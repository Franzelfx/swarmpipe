"""Frames as bytes in, bytes out: the whole of L2 without a framework or a socket.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

import numpy as np
import pytest

from swarmpipe.wire import DTYPES, MalformedFrame, TensorFrame, UnsupportedDtype

NUMPY_TAGS = [tag for tag, spec in DTYPES.items() if spec.numpy]


@pytest.mark.parametrize("tag", NUMPY_TAGS)
def test_every_numpy_dtype_round_trips(tag: str) -> None:
    array = (np.arange(6) % 2).astype(DTYPES[tag].numpy).reshape(2, 3)
    restored = TensorFrame.decode(TensorFrame.from_numpy(array, step=7).encode())

    assert restored.dtype == tag
    assert restored.shape == (2, 3)
    assert restored.step == 7
    assert np.array_equal(restored.to_numpy(), array)


def test_the_payload_is_exactly_as_big_as_the_shape_says() -> None:
    frame = TensorFrame.from_numpy(np.zeros((2, 3), dtype="<f4"))
    assert frame.nbytes == 24
    assert len(frame.payload) == 24


def test_nan_inf_and_negative_zero_survive_bit_for_bit() -> None:
    array = np.array([np.nan, np.inf, -np.inf, -0.0], dtype="<f4")
    restored = TensorFrame.decode(TensorFrame.from_numpy(array).encode()).to_numpy()
    # Compared as bits: NaN != NaN, and -0.0 == 0.0, so a value comparison here
    # would pass on a frame that quietly rewrote both.
    assert np.array_equal(array.view("<u4"), restored.view("<u4"))


def test_a_non_contiguous_view_is_sent_as_what_it_looks_like() -> None:
    base = np.arange(12, dtype="<f4").reshape(3, 4)
    transposed = base.T
    assert not transposed.flags["C_CONTIGUOUS"]

    frame = TensorFrame.from_numpy(transposed)
    assert frame.shape == (4, 3)
    assert np.array_equal(TensorFrame.decode(frame.encode()).to_numpy(), transposed)


def test_a_zero_size_tensor_is_a_tensor() -> None:
    frame = TensorFrame.decode(TensorFrame.from_numpy(np.zeros((0, 4), dtype="<f4")).encode())
    assert frame.payload == b""
    assert frame.to_numpy().shape == (0, 4)


def test_a_scalar_keeps_its_empty_shape() -> None:
    frame = TensorFrame.decode(TensorFrame.from_numpy(np.array(3.5, dtype="<f4")).encode())
    assert frame.shape == ()
    assert float(frame.to_numpy()) == 3.5


def test_the_step_id_rides_along_so_a_reply_can_be_matched_to_it() -> None:
    # Without it there is no way to notice that the gradient which arrived
    # answers a different step than the activation that was sent.
    frame = TensorFrame.from_numpy(np.zeros(2, dtype="<f4"), step=41)
    assert TensorFrame.decode(frame.encode()).step == 41


def test_meta_rides_in_the_header() -> None:
    frame = TensorFrame.from_numpy(np.zeros(2, dtype="<f4"), loss=0.25)
    assert TensorFrame.decode(frame.encode()).meta == {"loss": 0.25}


class TestBfloat16:
    """The dtype numpy cannot hold, which is why the payload is bytes."""

    BITS = np.array([0x3F80, 0xC000, 0x7FC0, 0x7F80, 0x0001], dtype="<u2")  # 1, -2, NaN, inf, denormal

    def frame(self) -> TensorFrame:
        return TensorFrame(dtype="bf16", shape=(5,), payload=self.BITS.tobytes(), step=3)

    def test_it_has_no_numpy_dtype_and_the_table_says_so(self) -> None:
        assert DTYPES["bf16"].numpy is None
        assert DTYPES["bf16"].itemsize == 2
        with pytest.raises(TypeError):
            np.dtype("bfloat16")

    def test_the_bits_survive_the_round_trip_untouched(self) -> None:
        restored = TensorFrame.decode(self.frame().encode())
        assert np.array_equal(restored.raw_bits(), self.BITS)
        assert restored.step == 3

    def test_it_costs_two_bytes_an_element_not_four(self) -> None:
        # The alternative was widening to fp32, which doubles the traffic on the
        # one link the whole project is bottlenecked on.
        assert len(self.frame().payload) == 10

    def test_to_numpy_refuses_rather_than_guessing(self) -> None:
        with pytest.raises(UnsupportedDtype, match="torch adapter"):
            self.frame().to_numpy()


class TestRefusals:
    """Everything that must not decode into a plausible-looking tensor."""

    def good(self) -> TensorFrame:
        return TensorFrame.from_numpy(np.arange(6, dtype="<f4"))

    def test_a_truncated_payload_never_becomes_a_smaller_tensor(self) -> None:
        header, payload = self.good().encode()
        with pytest.raises(MalformedFrame, match="20 bytes but shape"):
            TensorFrame.decode([header, payload[:-4]])

    def test_a_payload_with_too_much_in_it_is_refused(self) -> None:
        header, payload = self.good().encode()
        with pytest.raises(MalformedFrame, match="28 bytes but shape"):
            TensorFrame.decode([header, payload + b"xxxx"])

    @pytest.mark.parametrize("count", [0, 1, 3])
    def test_a_frame_is_two_parts(self, count: int) -> None:
        parts = ([*self.good().encode(), b"extra"])[:count] if count else []
        with pytest.raises(MalformedFrame, match="2 parts"):
            TensorFrame.decode(parts)

    def test_a_header_that_is_not_json_is_refused(self) -> None:
        with pytest.raises(MalformedFrame, match="not JSON"):
            TensorFrame.decode([b"\xff\xfe", self.good().payload])

    def test_a_header_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(MalformedFrame, match="not an object"):
            TensorFrame.decode([b"[1,2]", self.good().payload])

    def test_a_frame_from_a_future_version_is_refused(self) -> None:
        # The tags and the version sit in other people's job records; a reader
        # that guessed at an unknown version would be the thing that broke them.
        header = b'{"v":99,"dtype":"fp32","shape":[6],"step":0}'
        with pytest.raises(MalformedFrame, match="wire version"):
            TensorFrame.decode([header, self.good().payload])

    def test_an_unknown_dtype_tag_is_refused(self) -> None:
        header = b'{"v":1,"dtype":"fp8","shape":[6],"step":0}'
        with pytest.raises(UnsupportedDtype, match="fp8"):
            TensorFrame.decode([header, self.good().payload])

    def test_a_negative_dimension_is_refused(self) -> None:
        with pytest.raises(MalformedFrame, match="negative dimension"):
            TensorFrame(dtype="fp32", shape=(-1,), payload=b"")

    def test_a_numpy_dtype_with_no_wire_tag_is_refused(self) -> None:
        with pytest.raises(UnsupportedDtype, match="no wire tag"):
            TensorFrame.from_numpy(np.zeros(2, dtype=np.complex128))


def test_a_frame_crosses_a_real_link_unchanged() -> None:
    # L2 over L3 with nothing in between: the part boundaries L3 promises are
    # exactly what keeps the header out of the payload.
    from swarmpipe.link import loopback_pair

    upstream, downstream = loopback_pair()
    array = np.arange(24, dtype="<f4").reshape(2, 3, 4)

    upstream.send_multipart(TensorFrame.from_numpy(array, step=11).encode(), timeout=5.0)
    received = TensorFrame.decode(downstream.recv_multipart(timeout=5.0))

    assert received.step == 11
    assert np.array_equal(received.to_numpy(), array)
    assert downstream.bytes_recv == upstream.bytes_sent
