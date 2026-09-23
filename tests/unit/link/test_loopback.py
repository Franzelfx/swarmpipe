"""The in-process link against the shared contract, plus what is specific to it.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

import pytest

from swarmpipe.link import LinkTimeout, loopback_pair
from tests.unit.link.contract import LinkContract


class TestLoopbackLink(LinkContract):
    @contextlib.contextmanager
    def make_pair(self) -> Iterator[tuple]:
        upstream, downstream = loopback_pair()
        try:
            yield upstream, downstream
        finally:
            upstream.close()
            downstream.close()


def test_a_bounded_pair_applies_backpressure() -> None:
    upstream, _ = loopback_pair(maxsize=1)
    upstream.send_multipart([b"first"])
    # The second message has nowhere to go until the first is taken, and a
    # sender that silently dropped it would lose a training step.
    with pytest.raises(LinkTimeout):
        upstream.send_multipart([b"second"], timeout=0.05)


def test_a_sender_in_another_thread_unblocks_a_waiting_receiver() -> None:
    upstream, downstream = loopback_pair()
    received: list[list[bytes]] = []

    def receive() -> None:
        received.append(downstream.recv_multipart(timeout=5.0))

    waiter = threading.Thread(target=receive)
    waiter.start()
    upstream.send_multipart([b"late"])
    waiter.join(timeout=5.0)

    assert not waiter.is_alive(), "the receiver was still blocked after a message arrived"
    assert received == [[b"late"]]


def test_the_sender_keeps_no_hold_on_the_caller_s_buffer() -> None:
    upstream, downstream = loopback_pair()
    buffer = bytearray(b"original")
    upstream.send_multipart([buffer])
    buffer[:] = b"mutated!"
    # A caller reusing one buffer per step must not rewrite a message that is
    # already in flight.
    assert downstream.recv_multipart(timeout=5.0) == [b"original"]
