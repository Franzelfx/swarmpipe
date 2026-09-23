"""The shared contract every :class:`~swarmpipe.link.base.Link` must satisfy.

Not a test module on its own — pytest collects nothing from here. A transport
subclasses :class:`LinkContract`, supplies a connected pair, and inherits the
suite. That is the whole point: the parent project ended up with two transport
stacks that drifted because the second was written against the first one's
habits rather than a written contract, and an implementation that needs its own
version of these tests has not met the contract.

Adding a transport::

    class TestZmqLink(LinkContract):
        @contextlib.contextmanager
        def make_pair(self):
            ...
            yield upstream, downstream

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest

from swarmpipe.link.base import Link, LinkClosed, LinkTimeout

# Short, so a broken implementation fails the suite instead of hanging it.
TIMEOUT = 5.0


class LinkContract:
    """Inherit this and implement :meth:`make_pair`."""

    @contextlib.contextmanager
    def make_pair(self) -> Iterator[tuple[Link, Link]]:
        """Yield two connected links, and clean them up afterwards."""
        raise NotImplementedError

    @pytest.fixture
    def pair(self) -> Iterator[tuple[Link, Link]]:
        with self.make_pair() as ends:
            yield ends

    # ---- shape of the thing --------------------------------------------

    def test_both_ends_satisfy_the_protocol(self, pair) -> None:
        # isinstance only, never issubclass: Link carries data members, and
        # issubclass() rejects those outright.
        for end in pair:
            assert isinstance(end, Link)

    # ---- what arrives --------------------------------------------------

    def test_part_boundaries_survive_the_trip(self, pair) -> None:
        upstream, downstream = pair
        upstream.send_multipart([b"ab", b"c"], timeout=TIMEOUT)
        # The whole reason L2 can put a header in one part and a payload in the
        # next: a transport that concatenates would silently reshape a tensor.
        assert downstream.recv_multipart(timeout=TIMEOUT) == [b"ab", b"c"]

    def test_an_empty_part_is_still_a_part(self, pair) -> None:
        upstream, downstream = pair
        upstream.send_multipart([b"", b"payload", b""], timeout=TIMEOUT)
        assert downstream.recv_multipart(timeout=TIMEOUT) == [b"", b"payload", b""]

    def test_payloads_are_binary_safe(self, pair) -> None:
        upstream, downstream = pair
        # Nulls, high bytes and a length-prefix lookalike: a transport that
        # delimits on a byte value rather than a length breaks on exactly this.
        blob = bytes(range(256)) + b"\x00\x00\x00\x04" + b"\n\r\n"
        upstream.send_multipart([blob], timeout=TIMEOUT)
        assert downstream.recv_multipart(timeout=TIMEOUT) == [blob]

    def test_messages_arrive_in_the_order_they_were_sent(self, pair) -> None:
        upstream, downstream = pair
        for i in range(10):
            upstream.send_multipart([b"m", str(i).encode()], timeout=TIMEOUT)
        seen = [downstream.recv_multipart(timeout=TIMEOUT)[1] for _ in range(10)]
        assert seen == [str(i).encode() for i in range(10)]

    def test_the_link_carries_traffic_both_ways(self, pair) -> None:
        upstream, downstream = pair
        upstream.send_multipart([b"activation"], timeout=TIMEOUT)
        assert downstream.recv_multipart(timeout=TIMEOUT) == [b"activation"]
        # The reply path is not a separate link: a gradient comes back the way
        # the activation went out.
        downstream.send_multipart([b"gradient"], timeout=TIMEOUT)
        assert upstream.recv_multipart(timeout=TIMEOUT) == [b"gradient"]

    def test_a_large_message_survives_whole(self, pair) -> None:
        upstream, downstream = pair
        big = bytes(1_000_000)
        upstream.send_multipart([b"header", big], timeout=TIMEOUT)
        received = downstream.recv_multipart(timeout=TIMEOUT)
        assert received[0] == b"header"
        assert len(received[1]) == len(big)

    # ---- counters ------------------------------------------------------

    def test_counters_start_at_zero_and_count_payload_bytes(self, pair) -> None:
        upstream, downstream = pair
        assert (upstream.bytes_sent, upstream.bytes_recv) == (0, 0)
        assert (downstream.bytes_sent, downstream.bytes_recv) == (0, 0)

        upstream.send_multipart([b"abc", b"de"], timeout=TIMEOUT)
        downstream.recv_multipart(timeout=TIMEOUT)
        # Payload only: whatever framing the implementation adds is its own
        # business and must not show up in a caller's bandwidth report.
        assert upstream.bytes_sent == 5
        assert downstream.bytes_recv == 5
        assert upstream.bytes_recv == 0
        assert downstream.bytes_sent == 0

    # ---- refusals ------------------------------------------------------

    def test_an_empty_message_is_refused(self, pair) -> None:
        upstream, _ = pair
        with pytest.raises(ValueError):
            upstream.send_multipart([], timeout=TIMEOUT)

    def test_a_part_that_is_not_bytes_is_refused(self, pair) -> None:
        upstream, _ = pair
        with pytest.raises(ValueError):
            upstream.send_multipart(["not bytes"], timeout=TIMEOUT)

    def test_receiving_with_nothing_to_receive_times_out(self, pair) -> None:
        _, downstream = pair
        # It raises rather than returning an empty list: "nothing came back"
        # read as "nothing was sent" is how a dead peer becomes a training run
        # that keeps going on stale data.
        with pytest.raises(LinkTimeout):
            downstream.recv_multipart(timeout=0.05)

    # ---- closing -------------------------------------------------------

    def test_close_is_idempotent(self, pair) -> None:
        upstream, _ = pair
        upstream.close()
        upstream.close()

    def test_using_a_closed_link_raises_rather_than_blocking(self, pair) -> None:
        upstream, downstream = pair
        upstream.close()
        with pytest.raises(LinkClosed):
            upstream.send_multipart([b"x"], timeout=TIMEOUT)
        with pytest.raises(LinkClosed):
            upstream.recv_multipart(timeout=TIMEOUT)
        downstream.close()

    def test_a_receiver_waiting_on_a_closed_link_is_woken(self, pair) -> None:
        _, downstream = pair
        downstream.close()
        # Not a timeout: the caller must be able to tell "the link is gone" from
        # "the peer is slow", because only one of them is worth retrying.
        with pytest.raises(LinkClosed):
            downstream.recv_multipart(timeout=TIMEOUT)
