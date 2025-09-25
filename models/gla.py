from math import ceil
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

    def _get_chunk(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        start = idx * self.chunk_size
        end = (idx + 1) * self.chunk_size
        return x[:, :, start:end]

    def _chunk_gated_linear_attention(
        self,
        q: torch.Tensor,  # [B, L, H, D, 1]
        k: torch.Tensor,  # [B, L, H, D, 1]
        v: torch.Tensor,  # [B, L, H, D_V, 1]
        S: torch.Tensor,  # [B, H, D, D_V]
        gk: torch.Tensor,  # [B, L, H, 1, 1]
        o: torch.Tensor,  # [B, L, H, D_V]
    ) -> torch.Tensor:
        (B, L, H, D, _) = q.shape
        n_chunks = ceil(L / self.chunk_size)
        last_size = L % self.chunk_size if L % self.chunk_size > 0 else self.chunk_size

        q, k, v, gk = map(lambda x: x.squeeze(-1), (q, k, v, gk))
        q, k, v, gk, o = map(lambda x: x.transpose(1, 2), (q, k, v, gk, o))

        padding_needed = self.chunk_size - last_size if last_size > 0 else 0
        if padding_needed > 0:
            q, k, v, gk = map(
                lambda x: F.pad(x, (0, 0, 0, padding_needed)), (q, k, v, gk)
            )

        # local cumulative sum of gk within each chunk
        gk_cumsum = (
            gk.reshape(B, H, n_chunks, self.chunk_size, D)
            .cumsum(3)
            .reshape(B, H, L + padding_needed, D)
        )
        scale = q.shape[-1] ** -0.5

        A_mask = torch.tril(
            torch.ones(
                (1, 1, self.chunk_size, self.chunk_size),
                dtype=torch.bool,
                device=q.device,
            )
        )

        for idx in range(n_chunks):
            Q, K, V, G = map(lambda x: self._get_chunk(x, idx), (q, k, v, gk_cumsum))

            # Inter-chunk part: (Q*scale)*exp(G) @ S
            qg = (Q * scale) * torch.exp(G)
            o_inter = torch.einsum("bhlk,bhkv->bhlv", qg.float(), S.float())

            # Intra-chunk part: A = Q_hat @ K_hat.T
            G_base = gk_cumsum[:, :, idx * self.chunk_size]
            Q_hat = Q * torch.exp(G - G_base.unsqueeze(2)) * scale
            K_hat = K * torch.exp(G_base.unsqueeze(2) - G)
            A = torch.matmul(Q_hat, K_hat.permute(0, 1, 3, 2))
            A = A.masked_fill(~A_mask, 0.0)
            o_intra = torch.einsum("bhls,bhsv->bhlv", A, V)

            o_chunk = (o_inter + o_intra).to(o.dtype)

            # update the state for the *next* chunk
            gk_last = G[:, :, -1]  # [B, H, D]
            S = S * torch.exp(gk_last).unsqueeze(-1)

            w = torch.exp(gk_last.unsqueeze(2) - G)
            k_scaled = K * w
            S += torch.einsum("bhlk,bhlv->bhkv", k_scaled, V)

            # store the final output for the current chunk
            start_idx = idx * self.chunk_size
            end_idx = start_idx + self.chunk_size
            if idx == n_chunks - 1 and last_size > 0:
                o[:, :, start_idx:end_idx] = o_chunk[:, :, :last_size]
            else:
                o[:, :, start_idx:end_idx] = o_chunk

        return S

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
            last_state = x.new_zeros((B, self.num_heads, self.head_dim, self.head_dim))
        else:
            last_state = last_state.to(x.device, x.dtype)

        o = torch.zeros(
            (B, L, self.num_heads, self.head_dim), device=x.device, dtype=x.dtype
        )

        if self.mode == "chunk":
            last_state = self._chunk_gated_linear_attention(q, k, v, last_state, gk, o)
        else:
            last_state = last_state.unsqueeze(1)  # [B, 1, H, D, D]
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
