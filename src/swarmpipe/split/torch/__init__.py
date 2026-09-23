"""The torch backend of Layer 1: the surgery that carries out a plan.

The one package in the library allowed to import torch, which is what lets the
rest of it — the plan objects, the wire format, the transport — install and run
on a box with no GPU stack. Ships with the optional extra::

    pip install swarmpipe[torch]

:func:`~swarmpipe.split.torch.stage_builder.build_stage` is the whole surface:
a model and a :class:`~swarmpipe.split.spec.StageSpec` go in, one stage's
modules and its trainable parameters come out. No optimizer, no training loop,
no channel — those are the caller's opinions, and keeping them out is what makes
the split testable on a laptop with no GPU and no second machine.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from swarmpipe.split.torch.stage_builder import build_stage

__all__ = ["build_stage"]
