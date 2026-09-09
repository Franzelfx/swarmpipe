"""PEFT/LoRA fine-tuning (E7, wired into jobs by quantized-lora E3).

LoRA is the parameter-efficient path for fine-tuning large split models: the
pretrained weights are frozen and only small low-rank adapters train, which is
what makes a model far larger than the cluster's optimizer budget trainable at
all. Full fine-tuning costs roughly 16 bytes per parameter once AdamW's moments
and the fp32 master copy are counted; LoRA costs that only for the adapters.

PEFT is an optional dependency, imported lazily, so a fellow without it runs
full-precision, non-LoRA jobs unchanged and fails a LoRA one with a readable
message.

SilentSwarm — Copyright 2026 NexPatch AI UG.
Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE for details.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = [
    "apply_lora",
    "lora_fingerprint",
    "qualified_target_modules",
    "save_lora_adapters",
    "load_lora_adapters",
    "default_target_modules",
    "lora_parameters",
    "merge_lora",
]

# Attention projection names by architecture family, most specific first. Matched
# against the leaf module names inside one decoder block, so the right group is
# found by looking at the model rather than by trusting its config's model_type.
_ATTENTION_PROJECTIONS: tuple[tuple[str, ...], ...] = (
    ("q_proj", "k_proj", "v_proj", "o_proj"),  # Llama, Qwen, Mistral, Gemma
    ("query_key_value",),                      # GPT-NeoX, Falcon
    ("c_attn",),                               # GPT-2 (fused qkv; c_proj is
                                               # skipped because the MLP reuses
                                               # the name and peft matches on
                                               # the suffix)
    ("Wqkv",),                                 # MPT
    ("qkv_proj",),                             # Phi-3
)


def default_target_modules(topology: Any) -> list[str]:
    """Resolve which projections to adapt, from the model's own structure.

    Looks at the leaf module names inside the first decoder block and returns the
    first known attention-projection group fully present there. Falls back to
    every distinct leaf module name in the block that looks like a projection,
    so an unrecognised architecture still gets adapters rather than an error.

    Parameters
    ----------
    topology : DecoderTopology
        From :func:`~silent_swarm.runtime.torch.loader.introspect_decoder`.

    Returns
    -------
    list of str
        Module-name suffixes for ``LoraConfig(target_modules=...)``.

    Raises
    ------
    ValueError
        If the block exposes no adaptable submodule at all.
    """
    import torch.nn as nn

    blocks = list(topology.blocks)
    if not blocks:
        raise ValueError("Model has no decoder blocks to adapt.")

    leaf_names = {
        name.rpartition(".")[2]
        for name, module in blocks[0].named_modules()
        if not list(module.children()) and _is_projection(module, nn)
    }
    for group in _ATTENTION_PROJECTIONS:
        if set(group) <= leaf_names:
            return list(group)

    if not leaf_names:
        raise ValueError(
            "Could not find any adaptable projection in the decoder block; "
            "pass target_modules explicitly."
        )
    return sorted(leaf_names)


def _is_projection(module: Any, nn: Any) -> bool:
    """Whether a leaf module is a weight matrix LoRA can adapt.

    GPT-2 uses ``transformers.Conv1D`` rather than ``nn.Linear`` for its
    projections, so the check is on the shape of the module's weight, not on its
    class.
    """
    if isinstance(module, nn.Linear):
        return True
    weight = getattr(module, "weight", None)
    return weight is not None and getattr(weight, "ndim", 0) == 2


def apply_lora(
    model: Any,
    *,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> Any:
    """Wrap ``model`` with LoRA adapters via PEFT.

    The returned model has every pretrained parameter frozen and only the
    adapters trainable. ``introspect_decoder`` still resolves the topology
    through the wrapper, so the sharding and pipeline code keeps working.

    Parameters
    ----------
    model : nn.Module
        The model to adapt, modified in place and returned wrapped.
    r, alpha, dropout : optional
        LoRA rank, scaling, and dropout.
    target_modules : list of str or None, optional
        Module-name suffixes to adapt. ``None`` lets PEFT choose, which it does
        from the model type; prefer passing :func:`default_target_modules`.

    Raises
    ------
    RuntimeError
        If PEFT is not installed.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - exercised only without peft
        raise RuntimeError(
            "PEFT is not installed; run `pip install peft` to use LoRA fine-tuning."
        ) from exc

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def lora_parameters(model: Any) -> list[Any]:
    """The adapter parameters of a LoRA-wrapped model, for the optimizer.

    Returns
    -------
    list of nn.Parameter
        Every trainable parameter PEFT left unfrozen.

    Raises
    ------
    ValueError
        If nothing is trainable - an optimizer over an empty parameter list is a
        silent no-op that would train for the full run and change nothing.
    """
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError(
            "LoRA produced no trainable parameters; check that target_modules "
            "matches this architecture."
        )
    return params


def merge_lora(model: Any) -> Any:
    """Fold the adapters into the base weights and unwrap the PEFT model.

    Used before exporting a servable model: a merged model has the original
    architecture and parameter names, so the existing shard format and the
    inference path need to know nothing about adapters.

    Returns
    -------
    nn.Module
        The base model with adapters merged, or ``model`` unchanged when it
        carries none.
    """
    merge = getattr(model, "merge_and_unload", None)
    return merge() if callable(merge) else model


def lora_fingerprint(config: dict | None) -> str:
    """A stable short hash of the adapter configuration.

    Two pipeline stages train adapters over their own half of one model, and
    their shards are later merged into a single servable. That only makes sense
    if both stages adapted the same projections at the same rank, so each stage
    records this in its shard manifest and reconstruction refuses to merge
    shards that disagree - a mismatch produces a quietly wrong model otherwise.

    Computed from the **effective** configuration, after ``target_modules`` has
    been resolved from the architecture, so a genuine divergence is caught even
    when both stages were handed the same spec.

    Parameters
    ----------
    config : dict or None
        Effective LoRA config. Anything falsy, or not ``enabled``, fingerprints
        as the empty string, which is how a non-LoRA run is represented.

    Returns
    -------
    str
        16 hex characters, or ``""`` for a run with no adapters.
    """
    if not config or not config.get("enabled"):
        return ""
    payload = {
        "r": int(config.get("r", 8)),
        "alpha": int(config.get("alpha", 16)),
        "dropout": float(config.get("dropout", 0.05)),
        "target_modules": sorted(str(t) for t in (config.get("target_modules") or [])),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:16]


def qualified_target_modules(
    model: Any,
    blocks: Any,
    suffixes: list[str],
) -> list[str]:
    """Expand target suffixes into fully qualified names within ``blocks``.

    PEFT matches a list entry either exactly or as a dotted suffix, so full
    names restrict injection to precisely these blocks. A pipeline stage uses
    this to adapt only the blocks it runs - which matters once the other
    stage's blocks are on the meta device and must not be touched at all.

    The **suffixes** stay the architecture-level ones everywhere else, notably
    in the fingerprint: two stages adapting different halves of one model must
    still agree that they configured the same thing.

    Parameters
    ----------
    model : nn.Module
        The model whose module names are read.
    blocks : iterable of nn.Module
        The blocks to restrict injection to.
    suffixes : list of str
        Module-name suffixes, e.g. ``["q_proj", "v_proj"]``.

    Returns
    -------
    list of str
        Fully qualified module names, sorted.
    """
    owned = {id(block) for block in blocks}
    prefixes = [name for name, module in model.named_modules() if id(module) in owned]

    names: list[str] = []
    for prefix in prefixes:
        for child_name, _ in model.get_submodule(prefix).named_modules():
            if child_name.rpartition(".")[2] in suffixes:
                names.append(f"{prefix}.{child_name}")
    return sorted(names)


def save_lora_adapters(model: Any, out_dir: str | Any) -> dict:
    """Write a stage's adapter weights and their configuration.

    A LoRA run leaves the pretrained weights untouched, so a servable built from
    merged weights is the size of the whole model while the thing that actually
    changed is a few hundred times smaller. Saving the adapters on their own is
    what makes a fine-tune cheap to ship and to compose.

    Parameters
    ----------
    model : PeftModel
        The adapter-wrapped model.
    out_dir : path
        Directory to write ``adapter_model.safetensors`` and
        ``adapter_config.json`` into.

    Returns
    -------
    dict
        The adapter config that was written.

    Raises
    ------
    ValueError
        If ``model`` carries no adapters.
    """
    import json as _json
    from pathlib import Path as _Path

    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    config = getattr(model, "peft_config", None)
    if not config:
        raise ValueError("Model has no LoRA adapters to save.")

    out = _Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
    save_file(state, str(out / "adapter_model.safetensors"))

    active = next(iter(config.values()))
    payload = {
        "r": int(active.r),
        "lora_alpha": int(active.lora_alpha),
        "lora_dropout": float(active.lora_dropout),
        "target_modules": sorted(active.target_modules or []),
    }
    (out / "adapter_config.json").write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_lora_adapters(model: Any, adapter_dirs: list) -> Any:
    """Attach adapters saved by :func:`save_lora_adapters` to a base model.

    Takes several directories because a pipeline stage saves only the adapters
    for the blocks it ran: reassembling a servable means unioning the halves.
    Their target modules are disjoint by construction, so the union is exactly
    the adapter set of the whole model.

    Parameters
    ----------
    model : nn.Module
        The base model to adapt.
    adapter_dirs : list of path
        One directory per stage.

    Returns
    -------
    PeftModel
        ``model`` wrapped, with the adapter weights loaded.

    Raises
    ------
    ValueError
        If no adapter directory holds weights, or if the halves disagree on
        rank or scaling - which would make the reassembled model wrong.
    """
    import json as _json
    from pathlib import Path as _Path

    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    state: dict = {}
    targets: set[str] = set()
    shapes: set[tuple] = set()
    for directory in adapter_dirs:
        path = _Path(directory)
        weights = path / "adapter_model.safetensors"
        if not weights.exists():
            continue
        state.update(load_file(str(weights)))
        config = _json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        targets |= set(config.get("target_modules") or [])
        shapes.add((int(config["r"]), int(config["lora_alpha"]), float(config["lora_dropout"])))

    if not state:
        raise ValueError("No adapter weights found in any of the given directories.")
    if len(shapes) > 1:
        raise ValueError(
            f"Adapter halves disagree on rank/scaling/dropout: {sorted(shapes)}. "
            "They cannot be reassembled into one model."
        )

    r, alpha, dropout = next(iter(shapes))
    wrapped = apply_lora(model, r=r, alpha=alpha, dropout=dropout,
                         target_modules=sorted(targets))
    set_peft_model_state_dict(wrapped, state)
    return wrapped
