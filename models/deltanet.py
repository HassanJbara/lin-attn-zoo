from math import ceil
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
        vocab_size: int = 32000,
        mode: str = "chunk",
        chunk_size: int = 64,
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
        self.mode = mode
        self.chunk_size = chunk_size


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
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.k_conv1d, self.q_conv1d, self.v_conv1d = (
            self._build_conv(self.key_dim),
            self._build_conv(self.key_dim),
            self._build_conv(self.value_dim),
        )

        self.kq_norm = lambda x: x / x.norm(dim=-2, keepdim=True)  # l2 normalization
        self.activation = nn.SiLU()

        self.o_norm = nn.RMSNorm(self.head_dim, eps=self.norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def _build_conv(self, conv_dim):
        return nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=self.conv_size,
            groups=conv_dim,
            padding=self.conv_size - 1,
            bias=self.conv_bias,
        )

    def _calculate_beta(self, x: torch.Tensor) -> torch.Tensor:
        return self.b_proj(x).sigmoid()

    def _calculate_conv(
        self,
        x: torch.Tensor,  # [B, L, D]
        conv_layer: nn.Conv1d,
    ):
        # reshape to apply convolution across the sequence dimension, treat features as channels
        x = x.transpose(1, 2)  # [B, L, D] --> [B, D, L]
        x = conv_layer(x)
        x = x[..., : x.shape[-1] - (self.conv_size - 1)]
        x = self.activation(x)
        return x.transpose(1, 2)  # [B, D, L] --> [B, L, D]

    def _delta_rule(
        self,
        k: torch.Tensor,  # [B, 1, H, D, 1]
        q: torch.Tensor,  # [B, 1, H, D, 1]
        v: torch.Tensor,  # [B, 1, H, D, 1]
        beta: torch.Tensor,  # [B, 1, H, 1, 1]
        last_state: torch.Tensor,  # [B, 1, H, D, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        update = (last_state @ k - v) @ k.transpose(-1, -2)  # [B, 1, H, D, D]
        hidden_state = last_state - beta * update
        o = (hidden_state @ q / (self.head_dim**0.5)).squeeze(
            -1
        )  # [B, 1, H, D, 1] --> [B, 1, H, D]

        return o, hidden_state

    def _chunk_delta_rule(
        self, q, k, v, S, beta, o
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (B, L, H, D, _) = q.shape
        n_chunks = ceil(L / self.chunk_size)
        last_size = L % self.chunk_size
        padding_size = 0 if last_size == 0 else self.chunk_size - last_size

        q, k, v = map(lambda x: x.transpose(1, 2).reshape(B, H, L, D), (q, k, v))
        beta = beta.transpose(1, 2).reshape(B, H, L, 1)
        if padding_size > 0:
            q, k, v, beta = map(
                lambda x: F.pad(x, (0, 0, 0, padding_size)), (q, k, v, beta)
            )
        q = q / (self.head_dim ** (1 / 2))
        S = S.squeeze(dim=1)  # [B, H, D, D]
        I = torch.eye(self.chunk_size).repeat(B, H, 1, 1).to(q.device).type(q.dtype)
        M = (
            torch.tril(torch.ones((self.chunk_size, self.chunk_size)))
            .repeat(B, H, 1, 1)
            .to(q.device)
            .type(q.dtype)
        )

        outputs = []
        for idx in range(n_chunks):
            Q = q[:, :, idx * self.chunk_size : (idx + 1) * self.chunk_size]
            K = k[:, :, idx * self.chunk_size : (idx + 1) * self.chunk_size]
            V = v[:, :, idx * self.chunk_size : (idx + 1) * self.chunk_size]
            Bd = (
                beta[:, :, idx * self.chunk_size : (idx + 1) * self.chunk_size].repeat(
                    1, 1, 1, self.chunk_size
                )
                * I
            )
            T = (
                torch.linalg.solve_triangular(
                    (I + torch.tril(Bd @ K @ K.swapaxes(-1, -2), -1)).float(),
                    I.float(),
                    upper=False,
                ).type(q.dtype)
                @ Bd
            )
            W, U = T @ K, T @ V
            O = (
                (
                    Q @ S.swapaxes(-1, -2)
                    + (Q @ K.swapaxes(-1, -2) * M) @ (U - W @ S.swapaxes(-1, -2))
                )
                .movedim(2, 1)
                .unsqueeze(dim=-1)
            )
            if idx == n_chunks - 1 and last_size > 0:
                O = O[:, :last_size]
            outputs.append(O)
        o = torch.cat(outputs, dim=1)

        return o, S.unsqueeze(1)

    def forward(
        self, x: torch.Tensor, last_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.size()  # x: [B, L, D]

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        k = self._calculate_conv(k, self.k_conv1d).reshape(
            B, L, self.num_heads, self.head_dim, 1
        )
        q = self._calculate_conv(q, self.q_conv1d).reshape(
            B, L, self.num_heads, self.head_dim, 1
        )
        v = self._calculate_conv(v, self.v_conv1d).reshape(
            B, L, self.num_heads, self.head_dim, 1
        )

        k, q = self.kq_norm(k), self.kq_norm(q)

        beta = self._calculate_beta(x).reshape(B, L, self.num_heads, 1, 1)

        if last_state is None:
            last_state = torch.zeros(
                (B, 1, self.num_heads, self.head_dim, self.head_dim),
                device=x.device,
                dtype=x.dtype,
            )
            # Only set requires_grad after initialization and outside the compiled region
            if self.training:
                last_state.requires_grad = True
        else:
            last_state = last_state.to(x.device, x.dtype)

        o = torch.empty(
            (B, L, self.num_heads, self.head_dim, 1),
            device=x.device,
            dtype=x.dtype,
            requires_grad=self.training,
        )

        if self.mode == "chunk":
            o, last_state = self._chunk_delta_rule(q, k, v, last_state, beta, o)
            o = o.squeeze(-1)  # [B, L, H, D, 1] --> [B, L, H, D]
        else:
            outputs = []

            for t in range(L):
                beta_t = beta[
                    :, t : t + 1
                ]  # [B, 1, H, 1, 1], second dimension for broadcasting
                k_t = k[
                    :, t : t + 1
                ]  # [B, 1, H, 1, 1], second dimension for broadcasting
                q_t = q[
                    :, t : t + 1
                ]  # [B, 1, H, 1, 1], second dimension for broadcasting
                v_t = v[
                    :, t : t + 1
                ]  # [B, 1, H, 1, 1], second dimension for broadcasting
                o_t, last_state = self._delta_rule(k_t, q_t, v_t, beta_t, last_state)
                outputs.append(o_t)

            o = torch.cat(outputs, dim=1)  # [B, L, H, D]

        o = self.o_norm(o)
        o = o.reshape(
            B, L, self.num_heads * self.head_dim
        )  # [B, L, H, D] --> [B, L, H*D]
        o = self.o_proj(o)

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
            mode=config.mode,
            chunk_size=config.chunk_size,
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


class DeltaNetModel(nn.Module):
    def __init__(self, config: DeltaNetConfig, device=None):
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

        if device is not None:
            self.device = device
            self.to(device)

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
