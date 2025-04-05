from typing import Optional
from utils import SwiGLU

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaNetBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        key_dim: int,
        value_dim: int,
        query_dim: int,
    ) -> None:
        super(DeltaNetBlock, self).__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_dim = query_dim

        self.k_proj, self.q_proj, self.v_proj = self._build_projection_layers()
        self.k_conv, self.q_conv, self.v_conv = self._build_convolutional_layers()
        self.activation = nn.SiLU()

        self.beta_proj = nn.Linear(self.input_dim, 1)
        self.beta_activation = nn.Sigmoid()

        self.output_norm = nn.RMSNorm(self.memory_dim)
        self.output_proj = nn.Linear(self.memory_dim, self.input_dim)

    def _build_projection_layers(self):
        k_proj = nn.Linear(self.input_dim, self.key_dim)
        q_proj = nn.Linear(self.input_dim, self.query_dim)
        v_proj = nn.Linear(self.input_dim, self.value_dim)

        return k_proj, q_proj, v_proj

    def _build_convolutional_layers(self):
        k_conv = nn.Conv1d(self.key_dim, self.memory_dim, kernel_size=3, padding=1)
        q_conv = nn.Conv1d(self.query_dim, self.memory_dim, kernel_size=3, padding=1)
        v_conv = nn.Conv1d(self.value_dim, self.memory_dim, kernel_size=3, padding=1)

        return k_conv, q_conv, v_conv

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:  # apply L2 normalization
        return F.normalize(x, p=2, dim=-1)

    def _calculate_beta(self, x: torch.Tensor) -> torch.Tensor:
        x = self.beta_proj(x)
        x = self.beta_activation(x)
        return x

    def _delta_rule(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        last_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        last_state = (
            last_state
            if last_state is not None
            else torch.zeros((self.memory_dim, self.memory_dim))
        )
        return last_state - beta * (last_state @ k - v) @ k.T

    def forward(self, x: torch.Tensor, last_state: Optional[torch.Tensor] = None):
        batch_size, _, seq_len = x.size()  # (batch_size, input_dim, seq_len)

        k, q, v = self.k_proj(x), self.q_proj(x), self.v_proj(x)
        # Reshape for convolutional layers
        k = k.view(batch_size, self.key_dim, seq_len, 1)
        q = q.view(batch_size, self.query_dim, seq_len, 1)
        v = v.view(batch_size, self.value_dim, seq_len, 1)

        k, q, v = self.k_conv(k), self.q_conv(q), self.v_conv(v)

        k = k.view(batch_size, self.memory_dim, seq_len)
        q = q.view(batch_size, self.memory_dim, seq_len)
        v = v.view(batch_size, self.memory_dim, seq_len)

        k, q, v = self.activation(k), self.activation(q), self.activation(v)
        k, q = self._normalize(k), self._normalize(q)

        beta = self._calculate_beta(x)

        delta = self._delta_rule(k, v, beta, last_state)
        output = self.output_norm(delta)
        output = self.output_proj(output)

        return output


class DeltaNet(nn.Module):
    def __init__(
        self, input_dim, memory_dim, key_dim, value_dim, query_dim, num_layers
    ):
        super(DeltaNet, self).__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_dim = query_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                DeltaNetBlock(input_dim, memory_dim, key_dim, value_dim, query_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
