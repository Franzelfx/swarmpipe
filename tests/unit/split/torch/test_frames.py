"""The torch side of the frame seam: tensors in, frames out, tensors back.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.torch

TAGS = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp64": "float64",
    "int64": "int64",
    "int32": "int32",
    "int8": "int8",
    "uint8": "uint8",
    "bool": "bool",
}


@pytest.mark.parametrize(("tag", "torch_name"), sorted(TAGS.items()))
def test_every_dtype_round_trips_through_a_frame(tag: str, torch_name: str) -> None:
    import torch

    from swarmpipe.split.torch.frames import frame_from_tensor, tensor_from_frame

    dtype = getattr(torch, torch_name)
    original = (torch.arange(6) % 2).reshape(2, 3).to(dtype)

    frame = frame_from_tensor(original, step=5)
    assert frame.dtype == tag
    assert frame.shape == (2, 3)
    assert frame.step == 5
    assert len(frame.payload) == frame.nbytes

    restored = tensor_from_frame(frame)
    assert restored.dtype == dtype
    assert restored.shape == original.shape
    assert torch.equal(restored, original)


def test_bfloat16_crosses_as_its_own_bits_and_comes_back_exact() -> None:
    import torch

    from swarmpipe.split.torch.frames import frame_from_tensor, tensor_from_frame

    # Values bf16 represents exactly, plus the ones a lossy path would mangle.
    original = torch.tensor([1.0, -2.0, 0.0, float("inf"), float("nan")], dtype=torch.bfloat16)
    frame = frame_from_tensor(original)

    # Two bytes an element: not widened to fp32 on the way out.
    assert frame.dtype == "bf16"
    assert len(frame.payload) == 5 * 2

    restored = tensor_from_frame(frame)
    assert restored.dtype == torch.bfloat16
    # Compared as bits, so NaN counts as preserved rather than as unequal.
    assert torch.equal(restored.view(torch.int16), original.view(torch.int16))


def test_a_non_contiguous_tensor_is_sent_as_what_it_looks_like() -> None:
    import torch

    from swarmpipe.split.torch.frames import frame_from_tensor, tensor_from_frame

    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    view = base.t()
    assert not view.is_contiguous()

    restored = tensor_from_frame(frame_from_tensor(view))
    assert restored.shape == (4, 3)
    assert torch.equal(restored, view)


def test_a_zero_size_tensor_and_a_scalar_both_survive() -> None:
    import torch

    from swarmpipe.split.torch.frames import frame_from_tensor, tensor_from_frame

    empty = torch.zeros((0, 4), dtype=torch.float32)
    assert tensor_from_frame(frame_from_tensor(empty)).shape == (0, 4)

    scalar = torch.tensor(3.5)
    restored = tensor_from_frame(frame_from_tensor(scalar))
    assert restored.shape == ()
    assert float(restored) == 3.5


def test_a_frame_is_detached_from_the_graph_it_came_from() -> None:
    import torch

    from swarmpipe.split.torch.frames import frame_from_tensor, tensor_from_frame

    original = torch.ones(3, requires_grad=True) * 2
    assert original.requires_grad

    # Sending must not drag a graph across the wire, and the receiver decides
    # for itself whether its stage starts one.
    restored = tensor_from_frame(frame_from_tensor(original))
    assert not restored.requires_grad
    assert tensor_from_frame(frame_from_tensor(original), requires_grad=True).requires_grad


def test_the_rebuilt_tensor_does_not_share_memory_with_the_message() -> None:
    import torch

    from swarmpipe.split.torch.frames import frame_from_tensor, tensor_from_frame

    frame = frame_from_tensor(torch.arange(4, dtype=torch.float32))
    restored = tensor_from_frame(frame)
    restored += 1
    # A caller that reuses its receive buffer must not rewrite a tensor that is
    # already in use one layer up.
    assert torch.equal(tensor_from_frame(frame), torch.arange(4, dtype=torch.float32))


def test_a_dtype_the_wire_has_no_tag_for_is_refused() -> None:
    import torch

    from swarmpipe.split.torch.frames import frame_from_tensor
    from swarmpipe.wire import UnsupportedDtype

    with pytest.raises(UnsupportedDtype, match="no wire tag"):
        frame_from_tensor(torch.zeros(2, dtype=torch.complex64))


def test_a_tensor_goes_over_a_real_link_and_comes_back_a_tensor() -> None:
    import torch

    from swarmpipe.link import loopback_pair
    from swarmpipe.split.torch.frames import frame_from_tensor, tensor_from_frame
    from swarmpipe.wire import TensorFrame

    upstream, downstream = loopback_pair()
    activation = torch.randn(2, 3, 4, dtype=torch.float32)

    upstream.send_multipart(frame_from_tensor(activation, step=9).encode(), timeout=5.0)
    received = tensor_from_frame(TensorFrame.decode(downstream.recv_multipart(timeout=5.0)))

    assert torch.equal(received, activation)
