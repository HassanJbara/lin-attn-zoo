from math import sqrt
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from models.deltanet import DeltaNet
from models.gla import GatedLinearAttention
from models.gated_deltanet import GatedDeltaNet
from models.deltaproduct import GatedDeltaProduct

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig


ATTENTION_REGISTRY = {
    "gla": GatedLinearAttention,
    "deltanet": DeltaNet,
    "gated_deltanet": GatedDeltaNet,
    "deltaproduct": GatedDeltaProduct,
}


class ModelConfig(PretrainedConfig):
    def __init__(
        self,
        # core model dims
        hidden_size: int = 2048,
        num_heads: int = 16,
        num_hidden_layers: int = 24,
        norm_eps: float = 1e-6,
        # MLP dims
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        # vocab
        vocab_size: int = 32000,
        tie_word_embeddings: bool = False,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        # attention running mode
        mode: str = "chunk",  # "chunk" or "recurrent"
        chunk_size: int = 64,
        # attn selection
        attn_type: str = "deltanet",  # "deltanet" or "gla"
        attn_kwargs: Optional[Dict] = None,
        # optional extras used by init
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.mode = mode
        self.chunk_size = chunk_size

        self.attn_type = attn_type
        self.attn_kwargs = attn_kwargs or {}

        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class LMBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        assert config.attn_type in ATTENTION_REGISTRY, (
            f"Unknown attn_type={config.attn_type}"
        )
        attn_cls = ATTENTION_REGISTRY[config.attn_type]

        # only pass the common args; allow module-specific overrides via attn_kwargs
        self.attn = attn_cls(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            norm_eps=config.norm_eps,
            mode=config.mode,
            chunk_size=config.chunk_size,
            **config.attn_kwargs,
        )
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

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


class LMPreTrainedModel(PreTrainedModel):  # type: ignore
    config_class = ModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LMBlock"]
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


class LanguageModel(LMPreTrainedModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [LMBlock(config) for _ in range(config.num_hidden_layers)]
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


class ForCausalLM(LMPreTrainedModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.model = LanguageModel(config)
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
