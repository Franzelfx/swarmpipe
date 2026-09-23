"""Layer 2: tensor frames. Sees shape, dtype and bytes — never a framework.

:class:`~swarmpipe.wire.frames.TensorFrame` is one tensor on its way across a
link: a dtype tag, a shape, a payload and the step id that lets a reply be
matched to what it answers. ``encode`` gives the two parts a
:class:`~swarmpipe.link.base.Link` carries; ``decode`` takes them back.

The payload is **bytes**, not an array, which is what lets bf16 travel at all —
numpy has no bfloat16, and widening it would double the traffic on the one link
that is the bottleneck. The torch side of that conversion lives in
:mod:`swarmpipe.split.torch.frames`, the only module allowed to know both.

Torch-free by rule. Still to come: the stateless wire codecs (int8/int4 packing,
an fp16 cast), and the session that owns the forward/backward exchange.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from swarmpipe.wire.frames import (
    DTYPES,
    WIRE_VERSION,
    FrameError,
    MalformedFrame,
    TensorFrame,
    UnsupportedDtype,
)

__all__ = [
    "DTYPES",
    "WIRE_VERSION",
    "FrameError",
    "MalformedFrame",
    "TensorFrame",
    "UnsupportedDtype",
]
