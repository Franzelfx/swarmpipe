"""Layer 3: the transport path. Not yet implemented (T1).

L3 sees ``list[bytes]``. No tensors, no dtypes, no training steps. One protocol —
``send_multipart`` / ``recv_multipart`` / ``close``, plus the byte counters that
feed a caller's telemetry — with two implementations:

* ``zmq_link.py`` — direct peer-to-peer;
* ``relay_link.py`` — through an HTTP coordinator, for peers behind NAT;
* ``resolve.py`` — endpoint resolution and failover between the two.

Both must pass the *same* contract test. See the README for the layer
contracts.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

__all__: list[str] = []
