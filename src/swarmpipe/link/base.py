"""The Layer-3 contract: move a list of byte strings, and nothing more.

L3 sees ``list[bytes]``. Not tensors, not dtypes, not training steps — if a
signature here wants any of those, the layering is wrong and the concern belongs
in L2. Keeping this layer that narrow is what lets a relay coordinator run on a
small always-on box with no GPU stack: it moves bytes between peers and never
learns what they mean.

The protocol exists before any transport does, deliberately. The parent project
grew two transport stacks that drifted apart because the second was written
against the first one's habits rather than against a written contract; here both
the direct and the relayed implementation are written against
:class:`Link` and must pass one shared test suite unchanged.

What the contract promises
--------------------------
* A message is a list of parts. Every part arrives, in order, with its
  boundaries intact: send ``[b"ab", b"c"]`` and the peer receives
  ``[b"ab", b"c"]``, never ``[b"abc"]``. An empty part is a part.
* Messages arrive in the order they were sent.
* Payload bytes are counted, so a caller can report bandwidth without
  instrumenting the socket itself.
* ``close`` is idempotent, and using a closed link raises
  :class:`LinkClosed` rather than blocking or returning something plausible.

What it does not promise
------------------------
* **Delivery.** Nothing here retries, acknowledges or persists. A message
  handed to a link that later fails may be lost, and the caller finds out
  through a timeout or a closed link, not through a return value.
* **Reconnection**, and therefore nothing about ordering across one. A link
  that drops stays dropped; deciding whether to build a new one is the
  caller's business, and failover between a direct link and a relay belongs
  above this layer.
* **Concurrency.** One sender and one receiver per link. Two threads calling
  ``send_multipart`` on the same link is not defined behaviour.
* **Framing on the wire.** How parts are delimited is each implementation's
  business — except that it must not change, because the framing of the relay
  path is a compatibility surface that sits in the job records of running
  systems.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Link", "LinkClosed", "LinkError", "LinkTimeout"]


class LinkError(Exception):
    """Base class for every failure a link reports."""


class LinkClosed(LinkError):
    """Raised on use of a link that has been closed, at either end.

    Deliberately an error rather than a quiet empty result: a receiver that
    treats "nothing came back" as "nothing was sent" turns a dead peer into a
    training run that continues on stale data.
    """


class LinkTimeout(LinkError):
    """Raised when a send or receive did not complete within its timeout.

    A timeout never yields a partial message. Either every part of a message is
    returned or this is raised, because a short read on this layer becomes a
    wrongly-shaped tensor two layers up.
    """


@runtime_checkable
class Link(Protocol):
    """A bidirectional path that carries ``list[bytes]`` between two peers.

    Attributes
    ----------
    bytes_sent, bytes_recv : int
        Payload bytes moved so far — the sum of the part lengths, excluding
        whatever framing the implementation adds. They feed a caller's
        telemetry, which is why they count what the *caller* handed over rather
        than what went over the socket.
    """

    bytes_sent: int
    bytes_recv: int

    def send_multipart(self, parts: list[bytes], *, timeout: float | None = None) -> None:
        """Send one message, as an ordered list of parts.

        Parameters
        ----------
        parts : list of bytes
            The message. An empty part is preserved; an empty *list* is not a
            message and is rejected.
        timeout : float or None, optional
            Seconds to wait before giving up. ``None`` waits indefinitely.

        Raises
        ------
        LinkClosed
            If this link is closed.
        LinkTimeout
            If the message could not be handed over in time.
        ValueError
            If ``parts`` is empty or holds anything that is not ``bytes``.
        """
        ...

    def recv_multipart(self, *, timeout: float | None = None) -> list[bytes]:
        """Receive one whole message.

        Parameters
        ----------
        timeout : float or None, optional
            Seconds to wait before giving up. ``None`` waits indefinitely.

        Returns
        -------
        list of bytes
            Every part of one message, in the order it was sent.

        Raises
        ------
        LinkClosed
            If this link is closed, or was closed while waiting.
        LinkTimeout
            If no complete message arrived in time.
        """
        ...

    def close(self) -> None:
        """Release the link. Idempotent, and safe to call while a peer waits."""
        ...
