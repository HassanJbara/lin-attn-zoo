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
        x: torch.Tensor,  # [B, L, D]
        conv_layer: nn.Conv1d,
    ):
        # reshape to apply convolution across the sequence dimension, treat features as channels
        x = x.transpose(1, 2)  # [B, L, D] --> [B, D, L]

        x = conv_layer(x)[..., : x.size(-1)]
        x = self.activation(x)

        return x.transpose(1, 2)  # [B, D, L] --> [B, L, D]

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

    def _chunk_delta_rule(
        self,
        Q,  # [B, L, H, D]
        K,  # [B, L, H, D]
        V,  # [B, L, H, D]
        beta,  # [B, L, H, 1]
        C,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, H, D = Q.size()  # [B, L, H, D]
        num_chunks = ceil(L / C)
        padding = num_chunks * C - L

        if padding > 0:
            pad_size = (0, 0, 0, 0, 0, padding)
            Q = F.pad(Q, pad_size)
            K = F.pad(K, pad_size)
            V = F.pad(V, pad_size)
            beta = F.pad(beta, pad_size)

        Q, K, V = map(
            lambda x: x.view(B, num_chunks, C, H, D), (Q, K, V)
        )  # [B, L, H, D] --> [B, num_chunks, C, H, D]
        beta = beta.view(
            B, num_chunks, C, H, 1
        )  # [B, L, H, 1] --> [B, num_chunks, C, H, 1]

        K_beta = K * beta  # [B, num_chunks, C, H, D]
        V_beta = V * beta  # [B, num_chunks, C, H, D]

        K_beta_reshaped = K_beta.transpose(3, 4)  # [B, num_chunks, C, D, H]
        K_transposed = K.transpose(3, 4)  # [B, num_chunks, C, D, H]

        T = torch.zeros(B, num_chunks, C, C, device=Q.device)
        for h in range(H):
            T = T - (
                K_beta_reshaped[..., h] @ K_transposed[..., h].transpose(-1, -2)
            ).tril(-1)

        T = T + torch.eye(C, device=Q.device)

        for i in range(1, C):
            t_slice = T[:, :, i, :i] + (T[:, :, i, :, None] * T[:, :, :, :i]).sum(-2)
            T = T.clone()  # Avoid in-place operation
            T[:, :, i, :i] = t_slice

        W = torch.zeros_like(K_beta)  # [B, num_chunks, C, H, D]
        U = torch.zeros_like(V_beta)  # [B, num_chunks, C, H, D]

        for h in range(H):
            # T: [B, num_chunks, C, C]
            # K_beta[..., h, :]: [B, num_chunks, C, D]
            W[..., h, :] = T @ K_beta[..., h, :]
            U[..., h, :] = T @ V_beta[..., h, :]

        O = torch.empty_like(V)
        S = torch.zeros(B, H, D, D, device=Q.device)

        for i in range(num_chunks):
            q_i, k_i, w_i = (
                Q[:, i],
                K[:, i],
                W[:, i],
            )  # [B, C, H, D]

            # Eq. 8-9 - chunkwise Delta Rule forward update
            u_i = U[:, i] - torch.einsum("bchd,bhdd->bchd", w_i, S)  # [B, C, H, D]
            o_inter = torch.einsum("bchd,bhdd->bchd", q_i, S)  # [B, C, H, D]
            A_i = torch.einsum("bchd,bchd->bc", q_i, k_i).tril()  # [B, C, C]
            o_intra = torch.einsum("bc,bchd->bchd", A_i, u_i)  # [B, C, H, D]
            S = S + torch.einsum("bcdh,bche->bhde", k_i.transpose(-1, -2), u_i)
            O[:, i] = o_inter + o_intra

        if padding > 0:
            O = O.reshape(B, L + padding, H, D)[:, :L]

        return O.reshape(B, L, H, D), S

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
        o = torch.zeros((B, L, self.num_heads, head_dim), device=x.device)

        if self.mode == "chunk":
            o, last_state = self._chunk_delta_rule(q, k, v, beta, self.chunk_size)
        else:
            o_t = torch.zeros((B, self.num_heads, head_dim), device=x.device)
            outputs = []

            for t in range(L):
                beta_t = beta[:, t]
                k_t = k[:, t]
                q_t = q[:, t]
                v_t = v[:, t]

                o_t, last_state = self._delta_rule(k_t, q_t, v_t, beta_t, last_state)
                outputs.append(o_t)

            o = torch.stack(outputs, dim=1)  # [B, L, H, D]

        o = o.reshape(B, L, self.num_heads * head_dim)  # [B, L, H, D] --> [B, L, H*D]
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

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
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
        gate, y = (
            self.gate_proj(mlp_output),
            self.up_proj(mlp_output),
        )  # TODO: how is gate used?
        mlp_output = self.down_proj(self.swiglu(y))
        hidden_states = hidden_states + mlp_output

        return hidden_states, memory_state


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
