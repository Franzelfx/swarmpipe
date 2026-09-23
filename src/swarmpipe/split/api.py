"""The Layer-1 call: split a model, get back one stage's modules.

``build_stage(model, spec) -> StageBundle`` is the seam the pipeline-library
extraction draws under the model surgery. It does the introspection, the block
slicing, the LoRA injection scoped to this stage's blocks, the compressor
construction and splitting, the device placement and the trainable-parameter
selection - and **nothing else**: no optimizer, no training loop, no channel.
That makes the surgery unit-testable without a cluster, and leaves the loop
(which is an opinion about optimizers and schedules) in the application.

This module is framework-free on purpose: the modules a bundle carries are typed
``Any`` here and made concrete by a backend. The only backend that exists is the
torch one, :func:`~swarmpipe.split.torch.stage_builder.build_stage`, and that is
a deliberate decision rather than an oversight - the ``Protocol`` seam stays
because it is what keeps this module, and the layers below it, torch-free.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from swarmpipe.split.spec import StageSpec


@dataclass
class StageBundle:
    """Everything one pipeline stage needs to run, and nothing it does not.

    Attributes
    ----------
    spec : StageSpec
        The plan this bundle realises, with ``split_layer`` resolved.
    model : Any
        The model this stage runs. Not the same object that went in when LoRA is
        enabled - PEFT returns a wrapper - which is why the caller must use this
        one for its export step.
    topology, adapter : Any
        The decoder handles and the architecture glue (embed / causal mask /
        rotary) the segmented forward needs.
    blocks : list
        This stage's blocks, in order, already placed. ``block_devices`` gives
        their devices when the stage spans several local GPUs.
    layer_start, layer_end : int
        The half-open block range ``blocks`` came from.
    embed_tokens : Any or None
        Input embedding - the first stage only.
    final_norm, lm_head : Any or None
        Pre-head norm (may be absent architecturally) and output projection -
        the last stage only.
    compressor : Any or None
        This stage's half of the boundary compressor: the down-projection on the
        sender, the up-projection on the receiver. ``None`` when the boundary is
        uncompressed.
    trainable_parameters : list
        Exactly the parameters this stage optimizes. Never empty - an optimizer
        over an empty list runs every step and changes nothing, so
        :func:`build_stage` fails instead.
    lora : dict
        The effective adapter config, with ``target_modules`` resolved, or ``{}``
        when adapters are off. Both stages must agree on it.
    """

    spec: StageSpec
    model: Any
    topology: Any
    adapter: Any
    blocks: list[Any]
    layer_start: int
    layer_end: int
    embed_tokens: Any = None
    final_norm: Any = None
    lm_head: Any = None
    compressor: Any = None
    trainable_parameters: list[Any] = field(default_factory=list)
    lora: dict[str, Any] = field(default_factory=dict)

    @property
    def stage_index(self) -> int:
        """Pipeline position of this stage."""
        return self.spec.stage_index

    @property
    def is_first(self) -> bool:
        """Whether this stage embeds the tokens."""
        return self.spec.is_first

    @property
    def is_last(self) -> bool:
        """Whether this stage computes the loss."""
        return self.spec.is_last

    @property
    def block_devices(self) -> list[str] | None:
        """Per-block devices, or ``None`` when the stage sits on one device."""
        return self.spec.placement.block_devices

    @property
    def num_layers(self) -> int:
        """How many blocks this stage owns."""
        return self.layer_end - self.layer_start


@runtime_checkable
class StageBuilder(Protocol):
    """A framework backend that can carry out a :class:`StageSpec`."""

    def __call__(self, model: Any, spec: StageSpec) -> StageBundle:
        """Split ``model`` and return this stage's modules and parameters."""
