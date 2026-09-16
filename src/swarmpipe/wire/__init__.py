"""Layer 2: tensor frames and the session protocol. **Not yet implemented (T2).**

L2 sees a ``TensorFrame`` — shape, dtype, payload — and the role protocol on top
of it: an activation forward, a ``(loss, gradient)`` reply back. It is torch-free
by rule, numpy only; the torch↔numpy conversion is a thin adapter in L1's torch
backend.

What lands here, in order:

* ``frames.py`` — ``TensorFrame`` + encode/decode. **bf16 has no numpy dtype**
  and needs an explicit representation; that is the one real trap in T2.
* ``codec.py`` — the stateless wire codecs (int8/int4 packing, fp16 cast). These
  are *not* model modules: they have no parameters and no gradient (T3).
* ``session.py`` — the role object (upstream / downstream) that owns the
  exchange, including the **step id** that pairs an activation with its gradient.

See the README for the layer contracts.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

__all__: list[str] = []
