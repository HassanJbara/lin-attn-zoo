from math import log
from typing import List, Optional, Tuple, Union
from models.utils import GatedRMSNorm, SwiGLU

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaNetConfig:
    """Configuration class for GatedDeltaNet model."""

    def __init__(
        self,
        vocab_size=10000,
        hidden_size=128,
        num_hidden_layers=2,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        expand_v=2,
        head_dim=32,
        num_heads=3,
        mode="recurrent",  # Note: chunk mode is not implemented in GatedDeltaNet yet
        use_gate=True,
        use_short_conv=True,
        conv_size=4,
        conv_bias=False,
        norm_eps=1e-5,
        pad_token_id=0,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.mode = mode
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.norm_eps = norm_eps
        self.pad_token_id = pad_token_id


class GatedDeltaNet(nn.Module):
    """
    The layer implementaion for [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).

    Similar to Mamba2, each layer contains around 6*hidden_size*hidden_size parameters.
    Parameter alloation when use_gate=True:
        - 0.75 * hidden_size * hidden_size for the q_proj and k_proj each
        - 1.5 * hidden_size * hidden_size for the v_proj, g_proj and o_proj each
        - Others are ignorably small.
        - In total = 0.75 * 2 + 1.5 * 3 = 6 * hidden_size * hidden_size
    NOTE: num_heads * head_dim = 0.75 * hidden_size, please make sure to set the correct num_heads and head_dim.

    Parameter allocation when use_gate=False:
        - 1 * hidden_size * hidden_size for the q_proj and k_proj each
        - 2 * hidden_size * hidden_size for the v_proj and o_proj each
        - Others are ignorably small.
        - In total = 1 * 2 + 2 * 2 = 6 * hidden_size * hidden_size

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        mode (str, Optional):
            Which Gated DeltaNet kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_beta (bool, Optional):
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: int = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        mode: str = "chunk",
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        assert mode in ["chunk", "recurrent"], (
            "mode must be either 'chunk' or 'recurrent'"
        )

        self.mode = mode
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.key_dim * self.expand_v
        self.head_k_dim = head_dim
        self.head_v_dim = head_dim * self.expand_v
        self.layer_idx = layer_idx
        self.silu = nn.SiLU()

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # pyright: ignore

        self.dt_bias = self._build_dt_bias()
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True  # pyright: ignore

        self.k_conv1d, self.q_conv1d, self.v_conv1d = (
            self._build_conv(self.key_dim),
            self._build_conv(self.key_dim),
            self._build_conv(self.value_dim),
        )
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = GatedRMSNorm(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _build_dt_bias(self):
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (log(dt_max) - log(dt_min)) + log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        return nn.Parameter(inv_dt)

    def _build_conv(self, conv_dim):
        return nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=self.conv_size,
            groups=conv_dim,
            padding=self.conv_size - 1,
            bias=self.conv_bias,
        )

    def _calculate_conv(
        self,
        x: torch.Tensor,  # [B, L, D]
        conv_layer: nn.Conv1d,
    ):
        # reshape to apply convolution across the sequence dimension, treat features as channels
        x = x.transpose(1, 2)  # [B, L, D] --> [B, D, L]
        x = conv_layer(x)
        x = x[..., : x.shape[-1] - (self.conv_size - 1)]
        x = self.silu(x)
        return x.transpose(1, 2)  # [B, D, L] --> [B, L, D]

    def _calculate_beta(self, x: torch.Tensor) -> torch.Tensor:
        return self.b_proj(x).sigmoid()

    def _calculate_gate(self, x: torch.Tensor) -> torch.Tensor:
        return -self.A_log.exp() * F.softplus(self.a_proj(x) + self.dt_bias)

    def _gated_delta_rule(
        self,
        k_t: torch.Tensor,  # [B, H, D_K]
        q_t: torch.Tensor,  # [B, H, D_K]
        v_t: torch.Tensor,  # [B, H, D_V]
        g_t: torch.Tensor,  # [B, H, 1, 1]
        beta_t: torch.Tensor,  # [B, H, 1]
        recurrent_state: torch.Tensor,  # [B, H, D_K, D_V]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # L2 normalization to queries and keys
        q_t = F.normalize(q_t, p=2, dim=-1)
        k_t = F.normalize(k_t, p=2, dim=-1)

        q_t = q_t * (k_t.shape[-1] ** -0.5)  # scale
        recurrent_state = recurrent_state * torch.exp(g_t)

        correction = torch.einsum("bhk,bhkv->bhv", k_t, recurrent_state)
        v_t = (v_t - correction) * beta_t

        # Update hidden state with outer product using einsum
        recurrent_state = recurrent_state + torch.einsum("bhk,bhv->bhkv", k_t, v_t)
        o_t = torch.einsum("bhk,bhkv->bhv", q_t, recurrent_state)

        return o_t, recurrent_state

    def forward(
        self,
        x: torch.Tensor,
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = x.size()

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        k = (
            self._calculate_conv(k, self.k_conv1d)
            .reshape(B, L, self.num_heads, self.head_k_dim)
            .transpose(1, 2)
        )  # [B, H, L, D_K]
        q = (
            self._calculate_conv(q, self.q_conv1d)
            .reshape(B, L, self.num_heads, self.head_k_dim)
            .transpose(1, 2)
        )  # [B, H, L, D_K]
        v = (
            self._calculate_conv(v, self.v_conv1d)
            .reshape(B, L, self.num_heads, self.head_v_dim)
            .transpose(1, 2)
        )  # [B, H, L, D_V]

        beta = self._calculate_beta(x).unsqueeze(-1)  # [B, L, H] -> [B, L, H, 1]
        beta = beta.transpose(1, 2)  # [B, H, L, 1]

        g = self._calculate_gate(x).view(B, L, self.num_heads, 1, 1)  # [B, L, H, 1, 1]
        g = g.transpose(1, 2)  # [B, H, L, 1, 1]

        recurrent_state = (
            recurrent_state
            if recurrent_state is not None
            else torch.zeros(
                (B, self.num_heads, self.head_k_dim, self.head_v_dim),
                device=x.device,
                requires_grad=True,
                dtype=x.dtype,
            )
        )
        o = torch.zeros(
            (B, L, self.num_heads, self.head_v_dim),
            device=x.device,
            dtype=x.dtype,
            requires_grad=True,
        )

        if self.mode == "chunk":
            raise NotImplementedError(
                "Chunk mode is not implemented yet. Please use recurrent mode."
            )
        else:
            outputs = []

            # Inside the recurrent loop
            for t in range(L):
                q_t = q[:, :, t]  # [B, H, D_K]
                k_t = k[:, :, t]  # [B, H, D_K]
                v_t = v[:, :, t]  # [B, H, D_V]
                g_t = g[:, :, t]  # [B, H, 1, 1]
                beta_t = beta[:, :, t]  # [B, H, 1]

                o_t, recurrent_state = self._gated_delta_rule(
                    k_t, q_t, v_t, g_t, beta_t, recurrent_state
                )
                outputs.append(o_t)

            o = torch.stack(outputs, dim=1)  # [B, L, H, D_V]

        if self.use_gate:
            g = self.g_proj(x).reshape(B, L, self.num_heads, self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        o = o.reshape(
            B, L, self.num_heads * self.head_v_dim
        )  # [B, L, H, D_V] --> [B, L, H*D_V]
        o = self.o_proj(o)

        return o, recurrent_state


class GatedDeltaNetBlock(nn.Module):
    def __init__(self, config: GatedDeltaNetConfig):
        super().__init__()

        self.config = config

        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GatedDeltaNet(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            conv_size=config.conv_size,
            conv_bias=config.conv_bias,
            norm_eps=config.norm_eps,
            mode=config.mode,
            use_gate=config.use_gate,
            use_short_conv=config.use_short_conv,
            head_dim=config.head_dim,
            expand_v=config.expand_v,
        )
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        hidden_ratio = config.hidden_ratio if config.hidden_ratio is not None else 4
        intermediate_size = config.intermediate_size
        if intermediate_size is None:
            intermediate_size = int(config.hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)

        # self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.swiglu = SwiGLU(intermediate_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        last_memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_output = self.attn_norm(hidden_states)
        attn_output, memory_state = self.attn(
            attn_output, last_memory_state
        )  # [B, L, D]

        hidden_states = hidden_states + attn_output

        mlp_output = self.mlp_norm(hidden_states)
        # gate, y = (
        #     self.gate_proj(mlp_output),
        #     self.up_proj(mlp_output),
        # )  # TODO: how is gate used?
        y = self.up_proj(mlp_output)
        mlp_output = self.down_proj(self.swiglu(y))
        hidden_states = hidden_states + mlp_output

        return hidden_states, memory_state


class GatedDeltaNetModel(nn.Module):
    def __init__(self, config: GatedDeltaNetConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [GatedDeltaNetBlock(config) for _ in range(config.num_hidden_layers)]
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

        if memory_states is None:
            memory_states = [None for _ in self.layers]

        assert len(memory_states) == len(self.layers), (
            f"Expected {len(self.layers)} memory states, got {len(memory_states)}"
        )

        # process through each layer with its own memory state
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
