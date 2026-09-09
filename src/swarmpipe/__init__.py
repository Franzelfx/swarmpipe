"""swarmpipe — split a model across machines, and move the tensors between them.

Three layers, narrowest contract first:

* :mod:`swarmpipe.link` (L3) sees ``list[bytes]``. Nothing else.
* :mod:`swarmpipe.wire` (L2) sees ``TensorFrame`` — shape, dtype, payload — and
  the role protocol on top of it. Torch-free, numpy only.
* :mod:`swarmpipe.split` (L1) sees models. It never touches a socket.

L2 and L3 staying torch-free is not cosmetic: it is what lets a control plane
serve a relay, or a coordinator schedule a job, without installing PyTorch. Only
:mod:`swarmpipe.split.torch` imports torch, and it is an optional extra
(``pip install swarmpipe[torch]``).

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

__version__ = "0.0.0.dev0"

__all__ = ["__version__"]
