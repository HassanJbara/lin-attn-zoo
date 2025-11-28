from math import ceil
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 4,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        mode: str = "chunk",
        chunk_size: int = 64,
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
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.norm_eps = norm_eps
        self.mode = mode
        self.chunk_size = chunk_size

        self.head_dim = hidden_size // num_heads
        self.proj_dim = self.num_heads * self.head_dim  # Used for both key and value

        self.q_proj = nn.Linear(self.hidden_size, self.proj_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.proj_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.proj_dim, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.k_conv1d = self._build_conv(self.proj_dim)
        self.q_conv1d = self._build_conv(self.proj_dim)
        self.v_conv1d = self._build_conv(self.proj_dim)

        self.activation = nn.SiLU()
        self.o_norm = nn.RMSNorm(self.head_dim, eps=self.norm_eps)
        self.o_proj = nn.Linear(self.proj_dim, self.hidden_size, bias=False)

    def _build_conv(self, conv_dim: int) -> nn.Conv1d:
        return nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=self.conv_size,
            groups=conv_dim,
            padding=self.conv_size - 1,
            bias=self.conv_bias,
        )

    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-2, keepdim=True)

    def _calculate_beta(self, x: torch.Tensor) -> torch.Tensor:
        return self.b_proj(x).sigmoid()

    def _calculate_conv(
        self,
        x: torch.Tensor,  # [B, L, D]
        conv_layer: nn.Conv1d,
    ) -> torch.Tensor:
        # Apply convolution across sequence dimension
        x = x.transpose(1, 2)  # [B, L, D] --> [B, D, L]
        x = conv_layer(x)
        x = x[..., : x.shape[-1] - (self.conv_size - 1)]
        x = self.activation(x)
        return x.transpose(1, 2)  # [B, D, L] --> [B, L, D]

    def _reshape_for_attention(
        self, x: torch.Tensor, conv_layer: nn.Conv1d, B: int, L: int
    ) -> torch.Tensor:
        """Process tensors through convolution and reshape for attention."""
        return self._calculate_conv(x, conv_layer).reshape(
            B, L, self.num_heads, self.head_dim, 1
        )

    def _delta_rule(
        self,
        k: torch.Tensor,  # Shape: [B, 1, H, D, 1]
        q: torch.Tensor,  # Shape: [B, 1, H, D, 1]
        v: torch.Tensor,  # Shape: [B, 1, H, D, 1]
        S: torch.Tensor,  # Shape: [B, 1, H, D, D]
        beta: torch.Tensor,  # Shape: [B, 1, H, 1, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        update = (S @ k - v) @ k.transpose(-1, -2)  # [B, 1, H, D, D]
        S = S - beta * update
        o = (S @ q / (self.head_dim**0.5)).squeeze(-1)  # [B, 1, H, D]
        return o, S

    def _get_chunk(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        start = idx * self.chunk_size
        end = (idx + 1) * self.chunk_size
        return x[:, :, start:end]

    def _chunk_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        S: torch.Tensor,
        beta: torch.Tensor,
        o: torch.Tensor,
    ) -> torch.Tensor:
        # dimensions same as in _delta_rule except for L at dim 1
        (B, L, H, D, _) = q.shape
        n_chunks = ceil(L / self.chunk_size)
        last_size = L % self.chunk_size
        q, k, v = map(lambda x: x.transpose(1, 2).reshape(B, H, L, D), (q, k, v))
        beta = beta.transpose(1, 2).reshape(B, H, L, 1)

        padding_needed = self.chunk_size - last_size if last_size > 0 else 0
        if padding_needed > 0:
            q, k, v, beta = map(
                lambda x: F.pad(x, (0, 0, 0, padding_needed)), (q, k, v, beta)
            )

        q = q / (self.head_dim**0.5)
        S = S.squeeze(dim=1)  # [B, H, D, D]
        I = torch.eye(self.chunk_size).repeat(B, H, 1, 1).to(q.device).type(q.dtype)
        M = (
            torch.tril(torch.ones((self.chunk_size, self.chunk_size)))
            .repeat(B, H, 1, 1)
            .to(q.device)
            .type(q.dtype)
        )

        for idx in range(n_chunks):
            Q = self._get_chunk(q, idx)
            K = self._get_chunk(k, idx)
            V = self._get_chunk(v, idx)
            B = self._get_chunk(beta, idx)

            K_beta = K * B
            V_beta = V * B

            T = torch.linalg.solve_triangular(
                (I + torch.tril(K_beta @ K.swapaxes(-1, -2), -1)).float(),
                I.float(),
                upper=False,
            ).type(q.dtype)

            W, U = T @ K_beta, T @ V_beta
            S_swapped = S.swapaxes(-1, -2)

            O = (
                Q @ S_swapped + (Q @ K.swapaxes(-1, -2) * M) @ (U - W @ S_swapped)
            ).movedim(2, 1)

            # Handle partial chunk
            if idx == n_chunks - 1 and last_size > 0:
                O = O[:, :last_size]

            start_idx = idx * self.chunk_size
            end_idx = min((idx + 1) * self.chunk_size, L)
            o[:, start_idx:end_idx, :, :] = O

            S = S + (U - W @ S_swapped).swapaxes(-1, -2) @ K

        return S

    def forward(
        self, x: torch.Tensor, last_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.size()
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Process through convolution and reshape
        k = self._reshape_for_attention(k, self.k_conv1d, B, L)
        q = self._reshape_for_attention(q, self.q_conv1d, B, L)
        v = self._reshape_for_attention(v, self.v_conv1d, B, L)

        k, q = self._l2_normalize(k), self._l2_normalize(q)
        beta = self._calculate_beta(x).reshape(B, L, self.num_heads, 1, 1)

        if last_state is None:
            last_state = x.new_zeros(
                (B, 1, self.num_heads, self.head_dim, self.head_dim)
            )
        else:
            last_state = last_state.to(x.device, x.dtype)

        if self.mode == "chunk":
            o = torch.empty(
                (B, L, self.num_heads, self.head_dim), device=x.device, dtype=x.dtype
            )
            last_state = self._chunk_delta_rule(q, k, v, last_state, beta, o)
        else:
            outputs = []
            for t in range(L):
                t_slice = slice(t, t + 1)
                beta_t, k_t, q_t, v_t = (
                    beta[:, t_slice],
                    k[:, t_slice],
                    q[:, t_slice],
                    v[:, t_slice],
                )
                o_t, last_state = self._delta_rule(k_t, q_t, v_t, last_state, beta_t)
                outputs.append(o_t)
            o = torch.cat(outputs, dim=1)  # [B, L, H, D]

        o = self.o_norm(o)
        o = o.reshape(B, L, self.num_heads * self.head_dim)
        o = self.o_proj(o)

        return o, last_state
