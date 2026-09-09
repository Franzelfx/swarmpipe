"""Split a learned bottleneck across the pipeline boundary (E8).

For true bandwidth savings the *code* must cross the wire, not the
reconstruction. Splitting a :class:`LearnedBottleneck` puts the down-projection
on the upstream (sender) stage and the up-projection on the downstream
(receiver) stage, sharing the original module's parameters. Composed with the E6
:class:`~silent_swarm.runtime.pipeline.boundary.PipelineBoundary` — whose
"activation" becomes the narrow code — gradients train the down-projection on the
sender and the up-projection on the receiver, and the wire carries only the
narrow code.

SilentSwarm — Copyright 2026 NexPatch AI UG.
Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE for details.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .modules import LearnedBottleneck


class BottleneckSender(nn.Module):
    """Upstream half: project activations down to the wire code."""

    def __init__(self, bottleneck: LearnedBottleneck):
        super().__init__()
        self.down = bottleneck.down
        self.down_norm = bottleneck.down_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_norm(self.down(x))


class BottleneckReceiver(nn.Module):
    """Downstream half: reconstruct the activation from the wire code."""

    def __init__(self, bottleneck: LearnedBottleneck):
        super().__init__()
        self.up = bottleneck.up
        self.up_norm = bottleneck.up_norm

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.up_norm(self.up(z))


def split_bottleneck(bottleneck: LearnedBottleneck) -> tuple[BottleneckSender, BottleneckReceiver]:
    """Return ``(sender, receiver)`` halves sharing ``bottleneck``'s parameters.

    Note: the unquantized (fp16) code path is sent on the wire so gradients flow
    cleanly; int8-across-the-wire (STE through transport) is a later refinement.
    ``receiver(sender(x))`` equals ``bottleneck(x)`` when the bottleneck is not
    quantized.
    """
    return BottleneckSender(bottleneck), BottleneckReceiver(bottleneck)
