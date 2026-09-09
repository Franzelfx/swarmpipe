"""Load arbitrary HuggingFace causal LMs and introspect their decoder topology.

This is the heart of E3: instead of porting weights into a fixed custom model
(the Keras path), we load the real HF model with ``AutoModelForCausalLM`` and
**introspect** it generically — locating the ordered decoder block list, the
input embedding, the final norm, and the output head regardless of architecture
(GPT-2 ``.h``, LLaMA/Mistral/Qwen ``.layers``, etc.). Sharded safetensors are
handled natively by ``transformers``.

The topology is what later epics consume: E4 splits the block list across
devices/nodes, E5 inserts compression at a boundary, E6 wires the cross-node
pipeline. Introspection also resolves which modules must stay in full precision
when the weights are quantized (quantized-lora E1).

SilentSwarm — Copyright 2026 NexPatch AI UG.
Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE for details.
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


def full_precision_modules(model_name: str, *, revision: str | None = None) -> list[str]:
    """Qualified names of the modules that must not be quantized.

    The output head is kept in full precision: quantizing it costs little memory
    (one matrix) and measurably hurts output quality. When the head is tied to
    the input embedding the two share one tensor, so skipping the head also
    protects the embedding.

    Resolved by introspecting a meta-device model built from the config, so it
    works for any architecture and downloads no weights.

    Parameters
    ----------
    model_name : str
        HuggingFace repo id or local path.
    revision : str or None, optional
        Git revision to pin.

    Returns
    -------
    list of str
        Module names suitable for ``BitsAndBytesConfig(llm_int8_skip_modules=...)``.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name, revision=revision)
    with init_empty_weights():
        empty = AutoModelForCausalLM.from_config(config)
    topology = introspect_decoder(empty)

    for name, module in empty.named_modules():
        if module is topology.lm_head:
            return [name]
    return []


def build_quantization_config(
    quantization: Any,
    *,
    model_name: str | None = None,
    revision: str | None = None,
) -> Any:
    """Translate a quantization spec into a ``BitsAndBytesConfig``.

    Parameters
    ----------
    quantization : QuantizationSpec or dict or None
        Normalized by :func:`~silent_swarm.runtime.quantization.quantization_from_spec`
        if it is not already a spec. ``None`` returns ``None``.
    model_name, revision : optional
        Used to resolve the full-precision skip list by introspection when the
        spec does not name one explicitly.

    Returns
    -------
    BitsAndBytesConfig or None

    Raises
    ------
    RuntimeError
        If ``bitsandbytes`` is not installed.
    """
    from silent_swarm.runtime.quantization import QuantizationSpec, quantization_from_spec

    spec = quantization if isinstance(quantization, QuantizationSpec) else quantization_from_spec(quantization)
    if spec is None:
        return None

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without bitsandbytes
        raise RuntimeError(
            f"{spec.bits}-bit quantization needs bitsandbytes; run "
            "`pip install bitsandbytes` on the fellow (it is part of the "
            "'runtime' extra)."
        ) from exc

    import torch
    from transformers import BitsAndBytesConfig

    skip = spec.skip_modules
    if skip is None:
        skip = full_precision_modules(model_name, revision=revision) if model_name else []

    if spec.bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=spec.threshold,
            llm_int8_enable_fp32_cpu_offload=spec.cpu_offload,
            llm_int8_skip_modules=list(skip) or None,
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=spec.quant_type,
        bnb_4bit_use_double_quant=spec.double_quant,
        bnb_4bit_compute_dtype=getattr(torch, spec.compute_dtype),
        llm_int8_skip_modules=list(skip) or None,
    )


def load_causal_lm(
    model_name: str,
    *,
    device_map: Any = None,
    dtype: Any = None,
    revision: str | None = None,
    quantization: Any = None,
    offload_folder: str | None = None,
) -> nn.Module:
    """Load a pretrained causal LM from the HuggingFace Hub.

    Claims the torch framework for this process (see
    :func:`~silent_swarm.runtime.framework_guard.ensure_single_framework`) before
    importing the model classes.

    Parameters
    ----------
    model_name : str
        HuggingFace repo id.
    device_map : optional
        ``accelerate`` device map (e.g. ``"auto"`` or an explicit dict).
    dtype : optional
        Torch dtype for the weights. Under quantization this is the dtype of the
        modules left in full precision, not of the quantized ones.
    revision : str or None, optional
        Git revision to pin.
    quantization : QuantizationSpec or dict or None, optional
        Weight quantization (quantized-lora E1), e.g. ``{"bits": 8}``. ``None``
        loads full-precision weights, which is the default and byte-for-byte the
        previous behaviour.
    offload_folder : str or None, optional
        Where accelerate parks weights a ``device_map`` sends to ``"disk"``.
        Required whenever the map contains that target.

    Returns
    -------
    nn.Module
        The loaded ``*ForCausalLM`` model in eval mode.
    """
    from silent_swarm.runtime.framework_guard import ensure_single_framework

    ensure_single_framework("torch")

    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device_map is not None:
        kwargs["device_map"] = device_map
    if revision is not None:
        kwargs["revision"] = revision
    if offload_folder is not None:
        kwargs["offload_folder"] = offload_folder

    quant_config = build_quantization_config(
        quantization, model_name=model_name, revision=revision
    )
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model
