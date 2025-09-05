from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import Swish


class GatedLinearAttention(nn.Module):
    def __init__(
        self,
        mode: str = "chunk",
        hidden_size: int = 1024,
        num_heads: int = 4,
        norm_eps: float = 1e-5,
        chunk_size: int = 64,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        clamp_min: Optional[float] = None,
        elementwise_affine: Optional[bool] = True,
        use_output_gate: Optional[bool] = True,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, (
            "hidden_size must be divisible by num_heads"
        )
        assert mode in ["chunk", "recurrent"], (
            "mode must be either 'chunk' or 'recurrent'"
        )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.mode = mode
        self.chunk_size = chunk_size
        self.gate_logit_normalizer = gate_logit_normalizer
        self.gate_low_rank_dim = gate_low_rank_dim
        self.clamp_min = clamp_min
        self.elementwise_affine = (
            elementwise_affine if elementwise_affine is not None else True
        )
        self.use_output_gate = use_output_gate if use_output_gate is not None else False

        self.head_dim = self.hidden_size // self.num_heads
        self.proj_dim = self.num_heads * self.head_dim  # Used for both key and value

        self.q_proj = nn.Linear(self.hidden_size, self.proj_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.proj_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.proj_dim, bias=False)

        self.gk_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.gate_low_rank_dim, bias=False),
            nn.Linear(self.gate_low_rank_dim, self.proj_dim, bias=True),
        )

        if self.use_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.proj_dim, bias=False)
            self.gate_fn = Swish()

        self.g_norm = nn.RMSNorm(
            self.head_dim,
            elementwise_affine=self.elementwise_affine,
            eps=norm_eps,
        )

        self.o_proj = nn.Linear(self.proj_dim, self.hidden_size, bias=False)

    def _gated_linear_attention(
        self,
        k: torch.Tensor,  # Shape: [B, 1, H, D, 1]
        q: torch.Tensor,  # Shape: [B, 1, H, D, 1]
        v: torch.Tensor,  # Shape: [B, 1, H, D, 1]
        S: torch.Tensor,  # Shape: [B, 1, H, D, D]
        gk: torch.Tensor,  # Shape: [B, 1, H, D, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        update = k @ v.transpose(-1, -2)  # [B, 1, H, D, D]
        S = torch.exp(gk) * S + update  # [B, 1, H, D, D]
        o = (S.transpose(-1, -2) @ q / (self.head_dim**0.5)).squeeze(-1)  # [B, 1, H, D]
        return o, S

    def forward(
        self, x: torch.Tensor, last_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.size()

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim, 1)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim, 1)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim, 1)
        gk = self.gk_proj(x).reshape(B, L, self.num_heads, self.head_dim, 1)

        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        if last_state is None:
            last_state = x.new_zeros(
                (B, 1, self.num_heads, self.head_dim, self.head_dim)
            )
        else:
            last_state = last_state.to(x.device, x.dtype)

        if self.mode == "chunk":
            raise NotImplementedError("Chunk mode is not implemented yet.")
        else:
            outputs = []
            for t in range(L):
                t_slice = slice(t, t + 1)
                gk_t, k_t, q_t, v_t = (
                    gk[:, t_slice],
                    k[:, t_slice],
                    q[:, t_slice],
                    v[:, t_slice],
                )
                o_t, last_state = self._gated_linear_attention(
                    k_t, q_t, v_t, last_state, gk_t
                )
                outputs.append(o_t)
            o = torch.cat(outputs, dim=1)  # [B, L, H, D]

        o = self.g_norm(o)
        o = o.reshape(B, L, self.num_heads * self.head_dim)

        if self.use_output_gate:
            g = self.g_proj(x)
            o = o * self.gate_fn(g)

        o = self.o_proj(o)

        return o, last_state
