"""An in-process :class:`~swarmpipe.link.base.Link`, for building L2 without a socket.

Layer 2 has to be developed and tested against *something*, and waiting for a
transport would mean testing the wire format through a socket — which turns every
frame-encoding bug into a networking question. A pair of loopback links hands L2
a real :class:`~swarmpipe.link.base.Link` with no port, no peer process and no
timing: the frames go into a queue and come out the other side.

It is also the reference implementation of the contract. Whatever the ZMQ and
relay links do, they must do what this one does, verified by the same test suite.

Not for production use between processes: both ends live in one interpreter.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

import queue

from swarmpipe.link.base import LinkClosed, LinkTimeout

__all__ = ["LoopbackLink", "loopback_pair"]

# Pushed into a waiting receiver's inbox by close(), so a blocked recv wakes up
# and raises instead of waiting for a message that can no longer arrive.
_CLOSED = object()


def _validate(parts: list[bytes]) -> int:
    """Return the payload size of ``parts``, rejecting anything not a message."""
    if not parts:
        raise ValueError("a message needs at least one part; an empty list is not a message")
    for index, part in enumerate(parts):
        if not isinstance(part, bytes | bytearray | memoryview):
            raise ValueError(f"part {index} is {type(part).__name__}, not bytes")
    return sum(len(bytes(part)) for part in parts)


class LoopbackLink:
    """One end of an in-process link. Build a connected pair with :func:`loopback_pair`.

    Attributes
    ----------
    bytes_sent, bytes_recv : int
        Payload bytes moved through this end.
    """

    def __init__(self, inbox: queue.Queue, outbox: queue.Queue) -> None:
        self._inbox = inbox
        self._outbox = outbox
        self._closed = False
        self.bytes_sent = 0
        self.bytes_recv = 0

    @property
    def closed(self) -> bool:
        """Whether this end has been closed."""
        return self._closed

    def send_multipart(self, parts: list[bytes], *, timeout: float | None = None) -> None:
        """Hand one message to the peer's inbox."""
        if self._closed:
            raise LinkClosed("send on a closed link")
        size = _validate(parts)
        # A copy, so a caller reusing its buffer cannot mutate a message that is
        # already in flight — a real transport would have serialised it by now.
        message = [bytes(part) for part in parts]
        try:
            self._outbox.put(message, timeout=timeout)
        except queue.Full as exc:
            raise LinkTimeout(f"send timed out after {timeout}s") from exc
        self.bytes_sent += size

    def recv_multipart(self, *, timeout: float | None = None) -> list[bytes]:
        """Take one whole message off this end's inbox."""
        if self._closed:
            raise LinkClosed("receive on a closed link")
        try:
            message = self._inbox.get(timeout=timeout)
        except queue.Empty as exc:
            raise LinkTimeout(f"receive timed out after {timeout}s") from exc
        if message is _CLOSED:
            # Put it back: a second waiter must see the close too, and close()
            # only ever enqueues one sentinel.
            self._inbox.put(_CLOSED)
            raise LinkClosed("the link was closed while waiting for a message")
        self.bytes_recv += sum(len(part) for part in message)
        return message

    def close(self) -> None:
        """Close this end. Idempotent; wakes a receiver blocked on it."""
        if self._closed:
            return
        self._closed = True
        self._inbox.put(_CLOSED)


def loopback_pair(*, maxsize: int = 0) -> tuple[LoopbackLink, LoopbackLink]:
    """Return two connected :class:`LoopbackLink` ends.

    Parameters
    ----------
    maxsize : int, optional
        Messages that may be in flight in one direction before ``send_multipart``
        blocks. ``0`` means unbounded, which is the useful default for a test;
        a small value is how backpressure is exercised.

    Returns
    -------
    tuple of LoopbackLink
        ``(upstream, downstream)``. What one sends, the other receives.
    """
    to_downstream: queue.Queue = queue.Queue(maxsize=maxsize)
    to_upstream: queue.Queue = queue.Queue(maxsize=maxsize)
    upstream = LoopbackLink(inbox=to_upstream, outbox=to_downstream)
    downstream = LoopbackLink(inbox=to_downstream, outbox=to_upstream)
    return upstream, downstream
