"""Layer 3: the transport path. Sees ``list[bytes]`` and nothing else.

:class:`~swarmpipe.link.base.Link` is the contract — ``send_multipart`` /
``recv_multipart`` / ``close``, plus the payload counters a caller's telemetry
reads. What it promises, and the several things it deliberately does not, is in
that module's docstring; every implementation passes one shared contract suite
unchanged.

Implemented:

* :func:`~swarmpipe.link.loopback.loopback_pair` — two ends in one process, so
  Layer 2 can be built and tested with no socket and no peer.

Still to come: a direct peer-to-peer link over ZMQ, a relayed link through an
HTTP coordinator for peers behind NAT, and the resolution between them.

Torch-free by rule, and numpy-free besides: a coordinator relaying bytes needs
neither.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from swarmpipe.link.base import Link, LinkClosed, LinkError, LinkTimeout
from swarmpipe.link.loopback import LoopbackLink, loopback_pair

__all__ = [
    "Link",
    "LinkClosed",
    "LinkError",
    "LinkTimeout",
    "LoopbackLink",
    "loopback_pair",
]
