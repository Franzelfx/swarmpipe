"""Torch backend of Layer 1: the actual model surgery.

Carries out a :class:`~swarmpipe.split.spec.StageSpec` against a
loaded model and hands back a
:class:`~swarmpipe.split.api.StageBundle`: this stage's blocks, its
embeddings or head, its half of the boundary compressor, and exactly the
parameters it should optimize - all placed on the devices the spec asked for.

In the parent project this was the first ninety lines of a training function,
inseparable from the loop that followed it, which meant none of it could be
tested without a channel and a peer. Here it is a call: a model and a spec in,
one stage out. The loop stays with the caller — it is an opinion about
optimizers and schedules, and this is the layer beneath them.

Scope, as before: 2 stages, GPT-2 (learned positions) and Llama/Qwen (rotary).

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from swarmpipe.split.torch.framework_guard import ensure_single_framework

ensure_single_framework("torch")

from swarmpipe.split.api import StageBundle  # noqa: E402
from swarmpipe.split.spec import StageSpec  # noqa: E402
from swarmpipe.split.torch.arch_adapter import build_adapter  # noqa: E402
from swarmpipe.split.torch.compression.factory import compressor_from_spec  # noqa: E402
from swarmpipe.split.torch.compression.modules import LearnedBottleneck  # noqa: E402
from swarmpipe.split.torch.compression.split import split_bottleneck  # noqa: E402
from swarmpipe.split.torch.finetune.freeze import (  # noqa: E402
    freeze_backbone as _freeze_model_backbone,
)
from swarmpipe.split.torch.finetune.lora import (  # noqa: E402
    apply_lora,
    default_target_modules,
    qualified_target_modules,
)
from swarmpipe.split.torch.loader import introspect_decoder  # noqa: E402

__all__ = ["build_stage"]


def _require_trainable(params: list, stage_index: int, use_lora: bool) -> None:
    """Fail when a stage has nothing to optimize.

    An empty parameter list gives an optimizer that steps over nothing: the run
    completes, reports a loss, and leaves the weights exactly as they were. With
    LoRA this is reachable through a ``target_modules`` that matches nothing on
    this stage's blocks, so it is worth a hard failure rather than a silent one.
    """
    if params:
        return
    reason = (
        "check that target_modules matches this architecture"
        if use_lora
        else "check freeze_backbone and the compression boundary"
    )
    raise RuntimeError(f"Stage {stage_index} has no trainable parameters; {reason}.")


def _inject_lora(model, lora_cfg: dict[str, Any], topo, own_blocks: list):
    """Inject adapters into ``own_blocks`` and return the PEFT-wrapped model.

    Adapters are injected in place, so the block list (and the segmented forward
    that runs it) keeps working on the wrapped modules. PEFT is given the whole
    model, but each stage only ever adapts the blocks it owns - the same scoping
    the non-LoRA paths already use - so the two stages train disjoint halves.
    The reported config keeps the architecture-level suffixes (and so the
    fingerprint the two stages must agree on); only the injection is narrowed.
    Under a shard-only load the other stage's blocks hold meta tensors, so
    touching them would fail outright rather than merely waste adapters.
    """
    lora_cfg["target_modules"] = list(
        lora_cfg.get("target_modules") or default_target_modules(topo)
    )
    return apply_lora(
        model,
        r=int(lora_cfg.get("r", 8)),
        alpha=int(lora_cfg.get("alpha", 16)),
        dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=qualified_target_modules(model, own_blocks, lora_cfg["target_modules"]),
    )


def _build_compressors(hidden_size: int, compression_spec: dict, device: str):
    """Return the ``(sender, receiver)`` halves of the boundary compressor.

    A learned bottleneck splits across the boundary so the narrow *code* crosses
    the wire, not the reconstruction; anything else applies whole on the sender
    and leaves the receiver with nothing to do.
    """
    bottleneck = compressor_from_spec(hidden_size, compression_spec).to(device)
    if isinstance(bottleneck, LearnedBottleneck):
        sender, receiver = split_bottleneck(bottleneck)
        return sender.to(device), receiver.to(device)
    return bottleneck, None


def build_stage(model, spec: StageSpec) -> StageBundle:
    """Split ``model`` and materialize this stage.

    Introspects the decoder, slices out this stage's blocks, injects LoRA
    adapters scoped to them, builds and splits the boundary compressor, moves
    every module onto the device the spec names, and selects the trainable
    parameters. No optimizer, no training loop, no channel.

    Parameters
    ----------
    model : nn.Module
        The loaded model. Under a shard-only load it may hold only this stage's
        weights; the other stage's blocks are never touched.
    spec : StageSpec
        The plan: stage index, boundary layer, compression, LoRA, placement.

    Returns
    -------
    StageBundle
        This stage's modules, its trainable parameters, and the effective LoRA
        config. ``bundle.model`` is the model to export - with adapters enabled
        it is a PEFT wrapper, not the object passed in.

    Raises
    ------
    NotImplementedError
        For a pipeline that is not 2 stages, or a stage index outside it.
    RuntimeError
        If this stage ends up with nothing to train.

    Examples
    --------
    >>> spec = StageSpec(stage_index=0, split=SplitSpec(split_layer=2))  # doctest: +SKIP
    >>> bundle = build_stage(model, spec)  # doctest: +SKIP
    >>> len(bundle.blocks)  # doctest: +SKIP
    2
    """
    split = spec.split
    if split.num_stages != 2:
        raise NotImplementedError("torch distributed stage supports 2 stages.")
    if spec.stage_index not in (0, 1):
        raise NotImplementedError(
            f"my_index={spec.stage_index} unsupported for a 2-stage pipeline."
        )

    topo = introspect_decoder(model)
    # The adapter supplies the architecture-specific embed / causal-mask / rotary
    # glue so the segmented forward works for GPT-2 (learned positions) and
    # Llama/Qwen (rotary) alike.
    adapter = build_adapter(model, topo)

    blocks = list(topo.blocks)
    layer_start, layer_end = spec.layer_range(topo.num_layers)
    local_blocks = blocks[layer_start:layer_end]
    placement = spec.placement
    device = placement.device

    lora_cfg = dict(split.lora)
    use_lora = split.uses_lora
    if use_lora:
        model = _inject_lora(model, lora_cfg, topo, local_blocks)

    sender = receiver = None
    if split.uses_compression:
        sender, receiver = _build_compressors(topo.hidden_size, split.compression, device)
    # Each stage keeps its own half: the down-projection upstream, the
    # up-projection downstream.
    compressor = sender if spec.is_first else receiver

    embed_tokens = final_norm = lm_head = None
    if spec.is_first:
        embed_tokens = topo.embed_tokens
        # Move stage-0 modules to their device(s) - always, whether single-GPU or
        # multi-GPU. A `block_devices` guard here previously skipped this for
        # single-GPU distributed jobs (where the caller pins to one GPU), leaving
        # embed_tokens on CPU while inputs were moved to cuda:0 -> device mismatch
        # in F.embedding.
        embed_tokens.to(placement.embed)
        if adapter.wpe is not None:
            adapter.wpe.to(placement.embed)
        if adapter.rotary_emb is not None:
            adapter.rotary_emb.to(placement.embed)
    else:
        final_norm = topo.final_norm
        lm_head = topo.lm_head
        # Same single-GPU fix as stage 0: move modules unconditionally.
        if adapter.rotary_emb is not None:
            adapter.rotary_emb.to(placement.first_block_device)

    for index, block in enumerate(local_blocks):
        block.to(placement.block_device(index))
    if compressor is not None:
        # The sender encodes after the stage's last block; the receiver decodes
        # before its first one.
        compressor.to(
            placement.last_block_device if spec.is_first else placement.first_block_device
        )
    if final_norm is not None:
        final_norm.to(placement.head)
    if lm_head is not None:
        lm_head.to(placement.head)

    params = _select_parameters(
        model=model,
        adapter=adapter,
        spec=spec,
        local_blocks=local_blocks,
        embed_tokens=embed_tokens,
        final_norm=final_norm,
        lm_head=lm_head,
        compressor=compressor,
    )
    _require_trainable(params, spec.stage_index, use_lora)

    return StageBundle(
        spec=StageSpec(
            stage_index=spec.stage_index,
            # Report the resolved boundary, so a caller that asked for "halfway"
            # can see where halfway landed.
            split=replace(split, split_layer=split.boundary_layer(topo.num_layers)),
            placement=placement,
        ),
        model=model,
        topology=topo,
        adapter=adapter,
        blocks=local_blocks,
        layer_start=layer_start,
        layer_end=layer_end,
        embed_tokens=embed_tokens,
        final_norm=final_norm,
        lm_head=lm_head,
        compressor=compressor,
        trainable_parameters=params,
        lora=lora_cfg if use_lora else {},
    )


def _select_parameters(
    *,
    model,
    adapter,
    spec: StageSpec,
    local_blocks: list,
    embed_tokens,
    final_norm,
    lm_head,
    compressor,
) -> list:
    """Pick exactly what this stage optimizes, per fine-tuning mode."""
    split = spec.split
    if split.uses_lora:
        # Only the adapters injected into this stage's blocks carry
        # requires_grad; the compressor is not part of the PEFT model, so it
        # joins the optimizer separately and calibrates alongside them.
        params = [p for b in local_blocks for p in b.parameters() if p.requires_grad]
        if compressor is not None:
            params += list(compressor.parameters())
        return params

    if split.freeze_backbone and compressor is not None:
        # Genuinely freeze the backbone (requires_grad=False) so no gradients or
        # optimizer state are allocated for it and backward stops at the boundary
        # compressor - only the bottleneck trains. This is what lets a much
        # larger frozen model fit (~4 B/param vs ~16 for full FT).
        return _freeze_model_backbone(model, [compressor])

    params: list = []
    if spec.is_first:
        # Position-embedding params are architecture-specific: GPT-2 trains a
        # learned ``wpe``; rotary models (Llama/Qwen) have none.
        params += list(embed_tokens.parameters())
        if adapter.wpe is not None:
            params += list(adapter.wpe.parameters())
    params += [p for b in local_blocks for p in b.parameters()]
    if not spec.is_first:
        if final_norm is not None:
            params += list(final_norm.parameters())
        params += list(lm_head.parameters())
    if compressor is not None:
        params += list(compressor.parameters())
    return params
