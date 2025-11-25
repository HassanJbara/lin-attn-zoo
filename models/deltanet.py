from math import ceil, sqrt
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# only if transformers is installed
try:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ImportError:
    PreTrainedModel = nn.Module

    class CausalLMOutputWithPast:
        def __init__(
            self,
            loss=None,
            logits=None,
            past_key_values=None,
            hidden_states=None,
            **kwargs,
        ):
            self.loss = loss
            self.logits = logits
            self.past_key_values = past_key_values
            self.hidden_states = hidden_states


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

        # Create MLP submodule to match FLA structure
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(
                    config.hidden_size, intermediate_size, bias=False
                ),
                "up_proj": nn.Linear(config.hidden_size, intermediate_size, bias=False),
                "down_proj": nn.Linear(
                    intermediate_size, config.hidden_size, bias=False
                ),
            }
        )
        self.activation = nn.SiLU()

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
        # SwiGLU: gate_proj(x) * silu(up_proj(x))
        gate = self.activation(self.mlp["gate_proj"](mlp_output))
        up = self.mlp["up_proj"](mlp_output)
        mlp_output = self.mlp["down_proj"](gate * up)
        hidden_states = hidden_states + mlp_output

        return hidden_states, memory_state


class DeltaNetPreTrainedModel(PreTrainedModel):  # type: ignore
    config_class = DeltaNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeltaNetBlock"]
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: str | None = None,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

        if prenorm_residual_strategy is not None:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, "o_proj"):
                p = module.o_proj.weight
            elif hasattr(module, "down_proj"):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                if prenorm_residual_strategy == "rescale":
                    nn.init.kaiming_uniform_(p, a=sqrt(5))
                    with torch.no_grad():
                        p /= sqrt(
                            num_residuals_per_layer * self.config.num_hidden_layers
                        )
                elif prenorm_residual_strategy == "zero":
                    nn.init.zeros_(p)
                else:
                    raise ValueError(
                        f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}"
                    )


class DeltaNetModel(DeltaNetPreTrainedModel):
    def __init__(self, config: DeltaNetConfig):
        super().__init__(config)
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
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        memory_states: Optional[List[Union[torch.Tensor, None]]] = None,
    ) -> Tuple[torch.Tensor, List[Union[torch.Tensor, None]]]:
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

        return (hidden_states, memory_states)


class DeltaNetForCausalLM(DeltaNetPreTrainedModel):
    def __init__(self, config: DeltaNetConfig):
        super().__init__(config)

        self.model = DeltaNetModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[int] = 0,
        memory_states: Optional[List[Union[torch.Tensor, None]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
            if hasattr(self.config, "use_return_dict")
            else True
        )

        hidden_states, memory_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            memory_states=memory_states,
        )

        logits = self.lm_head(
            hidden_states
            if logits_to_keep is None
            else hidden_states[:, -logits_to_keep:]
        )

        loss = None
        if labels is not None:
            labels = labels.to(hidden_states.device)
            labels = torch.cat(
                (
                    labels[..., 1:],
                    torch.full_like(labels[:, :1], self.criterion.ignore_index),
                ),
                1,
            )
            loss = self.criterion(
                logits.view(labels.numel(), -1).float(), labels.view(-1)
            )

        if not return_dict:
            output = (logits, hidden_states, memory_states)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=memory_states,
            hidden_states=hidden_states,
        )
