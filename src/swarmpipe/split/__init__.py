"""Layer 1: model splitting and compression insertion.

L1 answers one question — *given a model and a plan, what does this stage run?*
It never touches a socket.

Two halves, deliberately separated:

* the **plan**, here and framework-free: :class:`~swarmpipe.split.spec.SplitSpec`
  (how the model is cut), :class:`~swarmpipe.split.spec.Placement` (where this
  stage's modules go), :class:`~swarmpipe.split.spec.StageSpec` (the two, for one
  stage), plus the layer→stage and layer→GPU planners in
  :mod:`swarmpipe.split.plan`. Importable with no torch installed, so a
  coordinator can build the plan a GPU worker executes;
* the **surgery**, in ``swarmpipe.split.torch`` (optional extra): ``build_stage``
  carries a spec out against a real model.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from swarmpipe.split.api import StageBuilder, StageBundle
from swarmpipe.split.plan import (
    GpuLayerSpan,
    plan_block_devices,
    plan_gpu_layers,
    spans_from_placement,
    stage_layer_ranges,
    weighted_split,
)
from swarmpipe.split.spec import Placement, SplitSpec, StageSpec

__all__ = [
    "GpuLayerSpan",
    "Placement",
    "SplitSpec",
    "StageBuilder",
    "StageBundle",
    "StageSpec",
    "plan_block_devices",
    "plan_gpu_layers",
    "spans_from_placement",
    "stage_layer_ranges",
    "weighted_split",
]
