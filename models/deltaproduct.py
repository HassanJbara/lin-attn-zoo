import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from models.utils import GatedRMSNorm


class GatedDeltaProduct(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 256,
        num_heads: int = 6,
        mode: str = "chunk",
        use_gate: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        allow_neg_eigval: bool = True,
        num_householder: int = 2,
    ) -> None:
        super().__init__()
        assert mode in ["chunk", "recurrent"], f"Not suppoerted mode `{mode}`."

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.num_householder = num_householder
        self.use_gate = use_gate
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.key_dim = int(self.num_heads * self.head_dim)
        self.layer_idx = layer_idx
        self.silu = nn.SiLU()

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim * num_householder, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.key_dim * num_householder, bias=False)
        self.b_proj = nn.Linear(
            hidden_size, self.num_heads * num_householder, bias=False
        )
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # pyright: ignore

        self.dt_bias = self._build_dt_bias()
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True  # pyright: ignore

        self.conv_size = conv_size
        self.k_conv1d, self.q_conv1d, self.v_conv1d = (
            self._build_conv(self.key_dim * num_householder),
            self._build_conv(self.key_dim),
            self._build_conv(self.key_dim * num_householder),
        )

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.o_norm = GatedRMSNorm(self.head_dim, eps=norm_eps)
        else:
            self.o_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.key_dim, hidden_size, bias=False)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def _build_dt_bias(self):
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
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
        x: torch.Tensor,  # [B, L, D * num_householder]
        conv_layer: nn.Conv1d,
    ):
        # reshape to apply convolution across the sequence dimension, treat features as channels
        x = x.transpose(1, 2)
        x = conv_layer(x)
        x = x[..., : x.shape[-1] - (self.conv_size - 1)]
        x = self.silu(x)
        return x.transpose(1, 2)

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

        recurrent_state = (
            recurrent_state
            if recurrent_state is not None
            else torch.zeros(
                (B, self.num_heads, self.head_dim, self.head_dim),
                device=x.device,
                dtype=x.dtype,
            )
        )

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        k = self._calculate_conv(k, self.k_conv1d).reshape(
            B, L * self.num_householder, self.num_heads, self.head_dim
        )
        q = self._calculate_conv(q, self.q_conv1d).reshape(
            B, L, self.num_heads, self.head_dim
        )  # [B, H, L, D_K]
        v = self._calculate_conv(v, self.v_conv1d).reshape(
            B, L * self.num_householder, self.num_heads, self.head_dim
        )  # [B, H, L, D_V]

        beta = self._calculate_beta(x)
        beta = beta * 2.0 if self.allow_neg_eigval else beta
        beta = beta.reshape(B, L * self.num_householder, self.num_heads, 1)

        g = self._calculate_gate(x)
        o = torch.zeros(
            (B, L, self.num_heads, self.head_dim), device=x.device, dtype=x.dtype
        )

        # L2 normalization to queries and keys
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        q = q / (self.head_dim**0.5)

        if self.mode == "chunk":
            raise NotImplementedError(
                "Chunk mode is not implemented in this version. Use 'recurrent' mode for recurrent gated delta product."
            )

        elif self.mode == "recurrent":
            g_new = g.new_zeros(
                g.shape[0], g.shape[1], self.num_householder, g.shape[2]
            )
            g_new[:, :, 0] = g
            g = g_new.reshape(
                g.shape[0], g.shape[1] * self.num_householder, g.shape[2], 1, 1
            )

            q_new = q.new_zeros(
                q.shape[0], q.shape[1], self.num_householder, q.shape[2], q.shape[3]
            )
            q_new[:, :, -1] = q
            q = q_new.reshape(
                q.shape[0], q.shape[1] * self.num_householder, q.shape[2], q.shape[3]
            )

            outputs = []
            for t in range(L * self.num_householder):
                q_t = q[:, t]
                k_t = k[:, t]
                v_t = v[:, t]
                g_t = g[:, t]
                beta_t = beta[:, t]

                o_t, recurrent_state = self._gated_delta_rule(
                    k_t, q_t, v_t, g_t, beta_t, recurrent_state
                )
                outputs.append(o_t)

            o = torch.stack(outputs, dim=1)  # [B, L * num_householder, H, D_V]
            o = o.reshape(
                o.shape[0],
                o.shape[1] // self.num_householder,
                self.num_householder,
                self.num_heads,
                self.head_dim,
            )[..., -1, :, :].contiguous()

        if self.use_gate:
            g = self.g_proj(x).reshape(B, L, self.num_heads, self.head_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)

        o = o.reshape(B, L, self.num_heads * self.head_dim)
        o = self.o_proj(o)

        return o, recurrent_state
