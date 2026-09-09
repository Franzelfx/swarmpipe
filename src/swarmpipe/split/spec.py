"""Serialisable description of how a model is split across pipeline stages (L1).

The Layer-1 contract of the pipeline-library extraction: one object that says
*how* a model is cut in two and *where* this stage's modules go, instead of the
nine keyword arguments :func:`~swarmpipe.runtime.torch.distributed_stage.run_torch_stage`
used to take. Deliberately framework-free - plain dataclasses, JSON in and JSON
out - so the torch-free control plane can build a spec that a GPU worker
executes.

Nothing here touches a model: it is a *plan*, and
:func:`~swarmpipe.split.torch.stage_builder.build_stage` is what carries it
out.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clean(mapping: Any) -> dict[str, Any]:
    """Coerce an optional mapping to a plain dict (``None`` -> ``{}``)."""
    return dict(mapping) if mapping else {}


@dataclass(frozen=True)
class Placement:
    """Where one stage's modules live.

    Attributes
    ----------
    device : str
        Default device for everything without a more specific home.
    block_devices : list of str or None
        Per-block device for this stage's blocks (intra-node multi-GPU, P3), in
        block order. ``None`` puts every block on ``device``. Build one with
        :func:`~swarmpipe.split.plan.plan_block_devices`.
    embed_device, head_device : str or None
        Overrides for the embeddings and the final norm + head; each defaults to
        ``device``.
    """

    device: str = "cpu"
    block_devices: list[str] | None = None
    embed_device: str | None = None
    head_device: str | None = None

    @property
    def embed(self) -> str:
        """Resolved embedding device."""
        return self.embed_device or self.device

    @property
    def head(self) -> str:
        """Resolved final-norm/head device."""
        return self.head_device or self.device

    def block_device(self, index: int) -> str:
        """Resolved device for this stage's ``index``-th block."""
        if self.block_devices is None:
            return self.device
        return self.block_devices[index]

    @property
    def first_block_device(self) -> str:
        """Device the stage's first block sits on (where a receiver decodes)."""
        return self.block_devices[0] if self.block_devices else self.device

    @property
    def last_block_device(self) -> str:
        """Device the stage's last block sits on (where a sender encodes)."""
        return self.block_devices[-1] if self.block_devices else self.device

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "device": self.device,
            "block_devices": list(self.block_devices) if self.block_devices is not None else None,
            "embed_device": self.embed_device,
            "head_device": self.head_device,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> Placement:
        """Rebuild from :meth:`to_dict` output; missing keys take their default."""
        data = _clean(payload)
        devices = data.get("block_devices")
        return cls(
            device=str(data.get("device", "cpu")),
            block_devices=[str(d) for d in devices] if devices else None,
            embed_device=data.get("embed_device"),
            head_device=data.get("head_device"),
        )


@dataclass(frozen=True)
class SplitSpec:
    """How the model is cut - the half both stages must agree on.

    Attributes
    ----------
    num_stages : int
        Pipeline length. Only 2 is implemented.
    split_layer : int or None
        Boundary block index: stage 0 owns ``blocks[:split_layer]``, stage 1 the
        rest. ``None`` means "halfway", resolved against the model in
        :meth:`boundary_layer`.
    compression : dict
        Boundary compression profile (see
        :func:`~swarmpipe.split.compression.factory.compressor_from_spec`).
        A learned bottleneck is split across the boundary so the narrow code, not
        the reconstruction, crosses the wire.
    lora : dict
        Adapter config (``enabled``, ``r``, ``alpha``, ``dropout``,
        ``target_modules``). When enabled, this stage trains adapters over its
        own blocks only and the pretrained weights stay frozen.
    freeze_backbone : bool
        Train only the boundary compressor. Ignored when ``lora`` is enabled.
    dtype, quantization : optional
        The precision the leader chose for the run. Carried here so one spec
        describes the whole placement decision; consumed by the *loader*
        (:func:`~swarmpipe.split.torch.loader.load_stage_shard`), not
        by the stage builder, which takes the model already loaded.
    """

    num_stages: int = 2
    split_layer: int | None = None
    compression: dict[str, Any] = field(default_factory=dict)
    lora: dict[str, Any] = field(default_factory=dict)
    freeze_backbone: bool = False
    dtype: str | None = None
    quantization: Any = None

    @property
    def uses_lora(self) -> bool:
        """Whether adapters are trained instead of the base weights."""
        return bool(self.lora.get("enabled"))

    @property
    def uses_compression(self) -> bool:
        """Whether a compressor sits on the boundary at all."""
        if not self.compression:
            return False
        return str(self.compression.get("method", "none")).strip().lower() != "none"

    def boundary_layer(self, num_layers: int) -> int:
        """Resolve :attr:`split_layer` against a model with ``num_layers`` blocks."""
        return num_layers // 2 if self.split_layer is None else int(self.split_layer)

    def layer_range(self, stage_index: int, num_layers: int) -> tuple[int, int]:
        """Half-open block range owned by ``stage_index``."""
        boundary = self.boundary_layer(num_layers)
        return (0, boundary) if stage_index == 0 else (boundary, num_layers)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "num_stages": self.num_stages,
            "split_layer": self.split_layer,
            "compression": dict(self.compression),
            "lora": dict(self.lora),
            "freeze_backbone": self.freeze_backbone,
            "dtype": self.dtype,
            "quantization": self.quantization,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> SplitSpec:
        """Rebuild from :meth:`to_dict` output; missing keys take their default."""
        data = _clean(payload)
        split = data.get("split_layer")
        return cls(
            num_stages=int(data.get("num_stages", 2)),
            split_layer=None if split is None else int(split),
            compression=_clean(data.get("compression")),
            lora=_clean(data.get("lora")),
            freeze_backbone=bool(data.get("freeze_backbone", False)),
            dtype=data.get("dtype"),
            quantization=data.get("quantization"),
        )


@dataclass(frozen=True)
class StageSpec:
    """One stage's share of a :class:`SplitSpec`, plus where its modules go."""

    stage_index: int
    split: SplitSpec = field(default_factory=SplitSpec)
    placement: Placement = field(default_factory=Placement)

    @property
    def is_first(self) -> bool:
        """Whether this stage owns the embeddings and reads the token stream."""
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        """Whether this stage owns the final norm, the head and the loss."""
        return self.stage_index == self.split.num_stages - 1

    def layer_range(self, num_layers: int) -> tuple[int, int]:
        """Half-open block range this stage owns."""
        return self.split.layer_range(self.stage_index, num_layers)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "stage_index": self.stage_index,
            "split": self.split.to_dict(),
            "placement": self.placement.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StageSpec:
        """Rebuild from :meth:`to_dict` output."""
        data = _clean(payload)
        return cls(
            stage_index=int(data.get("stage_index", 0)),
            split=SplitSpec.from_dict(data.get("split")),
            placement=Placement.from_dict(data.get("placement")),
        )
