from typing import List, Optional, Tuple, Union
from models.utils import SwiGLU

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaNetConfig:
    model_type = "delta_net"

    def __init__(
        self,
        hidden_size: int = 2048,
        conv_size: int = 4,
        conv_bias: bool = False,
        num_heads: int = 16,
        max_position_embeddings: int = 2048,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 24,
        norm_eps: float = 1e-6,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        vocab_size: int = 32000,
    ):
        self.hidden_size = hidden_size
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


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

        self.k_proj, self.q_proj, self.v_proj = (
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
        )
        self.k_conv, self.q_conv, self.v_conv = (
            self._build_conv(),
            self._build_conv(),
            self._build_conv(),
        )
        self.kq_norm = lambda x: F.normalize(x, p=2, dim=-1)  # l2 normalization
        self.activation = nn.SiLU()

        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.beta_activation = nn.Sigmoid()

        self.output_norm = nn.RMSNorm(self.hidden_size, eps=self.norm_eps)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _build_conv(self):
        return nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.conv_size,
            groups=self.hidden_size,
            padding=self.conv_size - 1,
            bias=self.conv_bias,
        )

    def _calculate_beta(self, x: torch.Tensor) -> torch.Tensor:
        x = self.beta_proj(x)
        x = self.beta_activation(x)
        return x

    def _calculate_conv(
        self,
        x: torch.Tensor,  # [B, T, D]
        conv_layer: nn.Conv1d,
    ):
        # Reshape to apply convolution across the sequence dimension, treat features as channels
        x = x.transpose(1, 2)  # [B, T, D] --> [B, D, T]

        # Apply convolution and trim
        x = conv_layer(x)[..., : x.size(-1)]
        x = self.activation(x)

        return x.transpose(1, 2)  # [B, D, T] --> [B, T, D]

    def _delta_rule(
        self,
        k: torch.Tensor,  # [B, H, D]
        q: torch.Tensor,  # [B, H, D]
        v: torch.Tensor,  # [B, H, D]
        beta: torch.Tensor,  # [B, H, 1]
        last_state: torch.Tensor,  # [B, H, D, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_unsqueezed = k.unsqueeze(-1)  # [B, H, D, 1]
        v_unsqueezed = v.unsqueeze(-1)  # [B, H, D, 1]

        update = (last_state @ k_unsqueezed - v_unsqueezed) @ k_unsqueezed.transpose(
            -1, -2
        )  # [B, H, D, D]
        hidden_state = last_state - beta.unsqueeze(-1) * update
        o = hidden_state @ q.unsqueeze(-1)  # [B, H, D, 1]
        o = o.squeeze(-1)  # [B, H, D]

        return o, hidden_state

    def forward(
        self, x: torch.Tensor, last_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.size()  # x: [B, L, D]
        head_dim = self.key_dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        k = self._calculate_conv(k, self.k_conv).reshape(B, L, self.num_heads, head_dim)
        q = self._calculate_conv(q, self.q_conv).reshape(B, L, self.num_heads, head_dim)
        v = self._calculate_conv(v, self.v_conv).reshape(B, L, self.num_heads, head_dim)

        k, q, v = self.activation(k), self.activation(q), self.activation(v)
        k, q = self.kq_norm(k), self.kq_norm(q)

        beta = self._calculate_beta(x).reshape(B, L, self.num_heads, 1)
        last_state = (
            last_state
            if last_state is not None
            else torch.zeros((B, self.num_heads, head_dim, head_dim), device=x.device)
        )
        o_t = torch.zeros((B, self.num_heads, head_dim), device=x.device)

        for t in range(L):
            beta_t = beta[:, t]
            k_t = k[:, t]
            q_t = q[:, t]
            v_t = v[:, t]

            o_t, last_state = self._delta_rule(k_t, q_t, v_t, beta_t, last_state)

        o = o_t.reshape(B, self.num_heads * head_dim)  # [B, H, D] --> [B, H*D]
        o = self.output_norm(o)
        o = self.output_proj(o)

        return o, last_state


class DeltaNetBlock(nn.Module):
    def __init__(self, config: DeltaNetConfig):
        super().__init__()

        self.config = config

        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = DeltaNet(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            conv_size=config.conv_size,
            conv_bias=config.conv_bias,
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
        self.swiglu = SwiGLU(intermediate_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        last_memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states  # [B, T, D]

        normalized = self.attn_norm(hidden_states)
        attn_output, memory_state = self.attn(normalized, last_memory_state)  # [B, D]

        # Update only the last token in residual
        updated_residual = residual.clone()
        updated_residual[:, -1, :] = residual[:, -1, :] + attn_output

        # Apply MLP only to the last token
        last_token = updated_residual[:, -1:, :]  # Shape: [B, 1, D]
        normalized_last = self.mlp_norm(last_token)
        gate, y = (
            self.gate_proj(normalized_last),
            self.up_proj(normalized_last),
        )  # TODO: how is gate used?
        mlp_output = self.down_proj(self.swiglu(y))  # Shape: [B, 1, D]

        final_output = updated_residual.clone()
        final_output[:, -1:, :] = last_token + mlp_output

        return final_output, memory_state


class DeltaNetModel(nn.Module):
    def __init__(self, config: DeltaNetConfig):
        super().__init__()
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

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = nn.CrossEntropyLoss()

    def _process_causal_lm_output(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor,
        logits_to_keep: Optional[int] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process outputs when the model is used as a causal language model"""
        logits = self.lm_head(
            hidden_states
            if logits_to_keep is None
            else hidden_states[:, -logits_to_keep:]
        )

        labels = labels.to(hidden_states.device)  # pyright: ignore
        labels = torch.cat(
            (
                labels[..., 1:],
                torch.full_like(labels[:, :1], self.criterion.ignore_index),
            ),
            1,
        )  # pyright: ignore

        loss = self.criterion(logits.view(labels.numel(), -1), labels.view(-1))

        return (loss, logits, hidden_states)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[int] = 0,
        memory_states: Optional[List[Union[torch.Tensor, None]]] = None,
    ) -> Union[
        Tuple[torch.Tensor, ...],
        Tuple[torch.Tensor, torch.Tensor, List[Union[torch.Tensor, None]]],
    ]:
        assert not (input_ids is not None and inputs_embeds is not None), (
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
        assert input_ids is not None or inputs_embeds is not None, (
            "You have to specify either input_ids or inputs_embeds"
        )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        # Initialize memory states if None
        if memory_states is None:
            memory_states = [None for _ in self.layers]

        # Ensure we have the right number of memory states
        assert len(memory_states) == len(self.layers), (
            f"Expected {len(self.layers)} memory states, got {len(memory_states)}"
        )

        # Process through each layer with its own memory state
        for i, layer in enumerate(self.layers):
            hidden_states, memory_states[i] = layer(hidden_states, memory_states[i])

        hidden_states = self.norm(hidden_states)

        if labels is not None:
            return self._process_causal_lm_output(hidden_states, labels, logits_to_keep)

        logits = self.lm_head(
            hidden_states
            if logits_to_keep is None
            else hidden_states[:, -logits_to_keep:]
        )
        return (logits, hidden_states, memory_states)
