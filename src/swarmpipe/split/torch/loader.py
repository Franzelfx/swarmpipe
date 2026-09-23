"""Introspect the decoder topology of a loaded causal LM.

Locates the ordered decoder block list, the input embedding, the final norm and
the output head **generically**, whatever the architecture (GPT-2 ``.h``,
Llama/Mistral/Qwen ``.layers``): the split has to work on a model nobody wrote
an adapter for. The topology is what the surgery consumes — it is where the
block range comes from, and the hidden size the boundary compressor is built
against.

Introspection only. Loading a model from the Hub is the caller's business, and
the split takes the model already loaded: see
:func:`~swarmpipe.split.torch.stage_builder.build_stage`.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

# Direct-child attribute names commonly used for the base decoder and final norm.
_BASE_DECODER_ATTRS = ("model", "transformer", "gpt_neox", "decoder", "backbone")
_FINAL_NORM_NAMES = ("norm", "ln_f", "final_layer_norm", "final_layernorm")


class ModelIntrospectionError(RuntimeError):
    """Raised when a model's decoder topology cannot be resolved generically."""


@dataclass
class DecoderTopology:
    """Architecture-agnostic handles into a loaded causal LM.

    Attributes
    ----------
    base : nn.Module
        The base decoder (e.g. ``LlamaModel``, ``GPT2Model``).
    blocks : nn.ModuleList
        The ordered transformer blocks.
    embed_tokens : nn.Module
        The input token embedding (``model.get_input_embeddings()``).
    final_norm : nn.Module or None
        The pre-head normalization, if present.
    lm_head : nn.Module
        The output projection (``model.get_output_embeddings()``).
    hidden_size, num_layers : int
        Pulled from the model config.
    tie_word_embeddings : bool
        Whether the head shares weights with the input embedding.
    """

    base: nn.Module
    blocks: nn.ModuleList
    embed_tokens: nn.Module
    final_norm: nn.Module | None
    lm_head: nn.Module
    hidden_size: int
    num_layers: int
    tie_word_embeddings: bool


def _config_int(config: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return None


def _find_base_decoder(model: nn.Module) -> nn.Module:
    get_decoder = getattr(model, "get_decoder", None)
    if callable(get_decoder):
        try:
            decoder = get_decoder()
            if isinstance(decoder, nn.Module):
                return decoder
        except (NotImplementedError, AttributeError):
            pass
    for attr in _BASE_DECODER_ATTRS:
        candidate = getattr(model, attr, None)
        if isinstance(candidate, nn.Module):
            return candidate
    return model


def _find_block_list(base: nn.Module, num_layers: int | None) -> nn.ModuleList:
    module_lists = [(name, m) for name, m in base.named_children() if isinstance(m, nn.ModuleList)]
    if not module_lists:
        raise ModelIntrospectionError(
            "Could not find a transformer block ModuleList on the decoder; "
            "this architecture may need a dedicated adapter."
        )
    if num_layers is not None:
        for _name, module_list in module_lists:
            if len(module_list) == num_layers:
                return module_list
    # Fall back to the longest ModuleList (the repeated blocks).
    return max(module_lists, key=lambda nm: len(nm[1]))[1]


def _is_norm(module: nn.Module, name: str) -> bool:
    cls = type(module).__name__
    return (
        isinstance(module, nn.LayerNorm)
        or "Norm" in cls
        or name in _FINAL_NORM_NAMES
    )


def _find_final_norm(
    base: nn.Module, blocks: nn.ModuleList, embed: nn.Module
) -> nn.Module | None:
    for name, module in base.named_children():
        if module is blocks or module is embed:
            continue
        if _is_norm(module, name):
            return module
    return None


def introspect_decoder(model: nn.Module) -> DecoderTopology:
    """Resolve the generic decoder topology of a loaded causal LM.

    Parameters
    ----------
    model : nn.Module
        A ``*ForCausalLM`` model.

    Returns
    -------
    DecoderTopology

    Raises
    ------
    ModelIntrospectionError
        If the block list cannot be located.
    """
    config = model.config
    num_layers = _config_int(config, "num_hidden_layers", "n_layer")
    hidden_size = _config_int(config, "hidden_size", "n_embd", "d_model")

    base = _find_base_decoder(model)
    blocks = _find_block_list(base, num_layers)
    embed_tokens = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    final_norm = _find_final_norm(base, blocks, embed_tokens)

    if num_layers is None:
        num_layers = len(blocks)

    return DecoderTopology(
        base=base,
        blocks=blocks,
        embed_tokens=embed_tokens,
        final_norm=final_norm,
        lm_head=lm_head,
        hidden_size=int(hidden_size) if hidden_size is not None else -1,
        num_layers=int(num_layers),
        tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
    )
