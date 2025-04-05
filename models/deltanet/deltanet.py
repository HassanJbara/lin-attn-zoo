from utils import SwiGLU
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaNetConfig:
    model_type = "delta_net"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 2048,
        conv_size: int = 4,
        num_heads: int = 16,
        max_position_embeddings: int = 2048,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 24,
        norm_eps: float = 1e-6,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.006,
        vocab_size: int = 32000,
    ):
        self.hidden_size = hidden_size
        self.conv_size = conv_size
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings


class DeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 4,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, (
            "hidden_size must be divisible by num_heads"
        )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.norm_eps = norm_eps

        self.key_dim = hidden_size // num_heads
        self.value_dim = hidden_size // num_heads

        self.k_proj, self.q_proj, self.v_proj = self._build_projection_layers()
        self.k_conv, self.q_conv, self.v_conv = (
            self._build_conv(self.key_dim),
            self._build_conv(self.key_dim),
            self._build_conv(self.value_dim),
        )
        self.kq_norm = lambda x: F.normalize(x, p=2, dim=-1)  # l2 normalization
        self.activation = nn.SiLU()

        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.beta_activation = nn.Sigmoid()

        self.output_norm = nn.RMSNorm(self.hidden_size, eps=self.norm_eps)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _build_projection_layers(self):
        k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        return k_proj, q_proj, v_proj

    def _build_conv(self, grouping_dim: int):
        return nn.Conv1d(
            in_channels=grouping_dim,
            out_channels=grouping_dim,
            kernel_size=self.conv_size,
            groups=grouping_dim,
            padding=self.conv_size - 1,
            bias=self.conv_bias,
        )

    def _calculate_beta(self, x: torch.Tensor) -> torch.Tensor:
        x = self.beta_proj(x)
        x = self.beta_activation(x)
        return x

    def _calculate_conv(self, x: torch.Tensor, conv_layer: nn.Conv1d):
        # Reshape to apply convolution across the sequence dimension, treat features as channels
        x = x.transpose(1, 2)  # [B, T, D] --> [B, D, T]

        # Apply convolution and trim
        x = conv_layer(x)[..., : x.size(-1)]
        x = self.activation(x)

        return x.transpose(1, 2)  # [B, D, T] --> [B, T, D]

    def _delta_rule(
        self,
        k: torch.Tensor,
        q: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        last_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        last_state = (
            last_state
            if last_state is not None
            else torch.zeros((self.hidden_size, self.hidden_size))
        )
        hidden_state = last_state - beta * (last_state @ k - v) @ k.T
        o = hidden_state @ q

        return o, hidden_state

    def forward(
        self, x: torch.Tensor, last_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k, q, v = self.k_proj(x), self.q_proj(x), self.v_proj(x)

        k = self._calculate_conv(k, self.k_conv)
        q = self._calculate_conv(q, self.q_conv)
        v = self._calculate_conv(v, self.v_conv)

        k, q, v = self.activation(k), self.activation(q), self.activation(v)
        k, q = self.kq_norm(k), self.kq_norm(q)

        beta = self._calculate_beta(x)

        o, memory_state = self._delta_rule(k, q, v, beta, last_state)
        o = self.output_norm(o)
        o = self.output_proj(o)

        return o, memory_state


class DeltaNetBlock(nn.Module):
    def __init__(self, config: DeltaNetConfig):
        super().__init__()

        self.config = config

        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = DeltaNet(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            conv_size=config.conv_size,
            norm_eps=config.norm_eps,
        )
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        hidden_ratio = config.hidden_ratio if config.hidden_ratio is not None else 4
        intermediate_size = config.intermediate_size
        if intermediate_size is None:
            intermediate_size = int(config.hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)

        # Set up MLP projections
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.swiglu = SwiGLU(intermediate_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        last_memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, memory_state = self.attn(hidden_states, last_memory_state)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)

        # Apply the MLP
        gate, y = self.gate_proj(hidden_states), self.up_proj(hidden_states)
        hidden_states = self.down_proj(self.swiglu(gate, y))

        hidden_states = residual + hidden_states

        return hidden_states, memory_state


class DeltaNetModel:
    def __init__(self, config: DeltaNetConfig):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [DeltaNetBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        assert not (input_ids is not None and inputs_embeds is not None), (
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
        assert input_ids is not None or inputs_embeds is not None, (
            "You have to specify either input_ids or inputs_embeds"
        )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds
        memory_state = None

        for layer in self.layers:
            hidden_states, memory_state = layer(
                hidden_states, last_memory_state=memory_state
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states
