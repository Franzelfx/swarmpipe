"""Plan how a model's layers map to pipeline stages and GPUs.

Pure Python (no torch), so it is unit-testable without a GPU. Mirrors the
leader's placement policy: ``weighted`` distributes layers proportionally to free
VRAM, ``equal`` distributes them evenly. The leader's existing placement output
(``gpu_slots`` with per-GPU layer ranges) can be consumed directly via
:func:`spans_from_placement`.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuLayerSpan:
    """A contiguous block range assigned to one local GPU."""

    gpu_index: int
    layer_start: int
    layer_end: int

    @property
    def num_layers(self) -> int:
        return max(0, self.layer_end - self.layer_start)


def weighted_split(num_layers: int, weights: list[float]) -> list[int]:
    """Split ``num_layers`` across buckets proportional to ``weights``.

    Uses the largest-remainder method so the sizes sum exactly to ``num_layers``.
    Non-positive total weight falls back to an even split.
    """
    if not weights:
        return []
    if num_layers <= 0:
        return [0 for _ in weights]

    clamped = [max(0.0, float(w)) for w in weights]
    total = sum(clamped)
    if total <= 0:
        base, rem = divmod(num_layers, len(weights))
        return [base + (1 if i < rem else 0) for i in range(len(weights))]

    raw = [num_layers * (w / total) for w in clamped]
    sizes = [int(r) for r in raw]
    remaining = num_layers - sum(sizes)
    # Hand the leftover layers to the largest fractional remainders.
    order = sorted(range(len(weights)), key=lambda i: raw[i] - sizes[i], reverse=True)
    for i in range(remaining):
        sizes[order[i]] += 1
    return sizes


def stage_layer_ranges(num_layers: int, num_stages: int) -> list[tuple[int, int]]:
    """Split layers into ``num_stages`` contiguous pipeline ranges (even)."""
    if num_stages <= 0:
        raise ValueError("num_stages must be >= 1")
    per = num_layers // num_stages
    ranges: list[tuple[int, int]] = []
    start = 0
    for stage in range(num_stages):
        end = num_layers if stage == num_stages - 1 else start + per
        ranges.append((start, end))
        start = end
    return ranges


def plan_gpu_layers(
    layer_start: int,
    layer_end: int,
    gpu_free_gb: list[float],
    *,
    gpu_indices: list[int] | None = None,
    mode: str = "weighted",
) -> list[GpuLayerSpan]:
    """Distribute ``[layer_start, layer_end)`` across local GPUs.

    Parameters
    ----------
    layer_start, layer_end : int
        Half-open block range owned by this node/stage.
    gpu_free_gb : list of float
        Free VRAM per local GPU.
    gpu_indices : list of int or None, optional
        Physical GPU indices; defaults to ``range(len(gpu_free_gb))``.
    mode : {"weighted", "equal"}
        ``weighted`` scales by free VRAM; ``equal`` distributes evenly.

    Returns
    -------
    list of GpuLayerSpan
        One span per GPU that received at least one layer.
    """
    count = layer_end - layer_start
    indices = gpu_indices if gpu_indices is not None else list(range(len(gpu_free_gb)))
    weights = [1.0] * len(gpu_free_gb) if str(mode).lower() == "equal" else list(gpu_free_gb)
    sizes = weighted_split(count, weights)

    spans: list[GpuLayerSpan] = []
    cursor = layer_start
    for gpu_index, size in zip(indices, sizes):
        if size <= 0:
            continue
        spans.append(GpuLayerSpan(gpu_index, cursor, cursor + size))
        cursor += size
    return spans


def plan_block_devices(
    num_blocks: int,
    gpu_free_gb: list[float],
    *,
    gpu_indices: list[int] | None = None,
    mode: str = "weighted",
) -> list[str]:
    """Return a per-block CUDA device string, spreading blocks across local GPUs.

    Distributes ``num_blocks`` over the local GPUs (``equal`` or ``weighted`` by
    free VRAM) and returns ``["cuda:0", "cuda:0", "cuda:1", ...]`` (length
    ``num_blocks``).
    """
    spans = plan_gpu_layers(0, num_blocks, gpu_free_gb, gpu_indices=gpu_indices, mode=mode)
    devices: list[str] = [""] * num_blocks
    for span in spans:
        for i in range(span.layer_start, span.layer_end):
            devices[i] = f"cuda:{span.gpu_index}"
    return devices


def spans_from_placement(device_placement: dict) -> list[GpuLayerSpan]:
    """Build spans from the leader's placement payload (``gpu_slots``).

    Consumes the same structure the Keras distributed stage reads, so the torch
    runtime honors the leader's VRAM-based plan unchanged.
    """
    spans: list[GpuLayerSpan] = []
    for slot in (device_placement or {}).get("gpu_slots") or []:
        if not isinstance(slot, dict) or slot.get("gpu_index") is None:
            continue
        try:
            gpu_index = int(slot["gpu_index"])
            start = int(slot["layer_start"])
            end = int(slot["layer_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            spans.append(GpuLayerSpan(gpu_index, start, end))
    return sorted(spans, key=lambda s: (s.layer_start, s.gpu_index))
