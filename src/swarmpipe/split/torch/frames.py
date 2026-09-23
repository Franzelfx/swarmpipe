"""The one place a torch tensor becomes a wire frame, and back.

Layer 2 may not import torch, and Layer 1's plan half may not either, so the
conversion lives here in the backend — the only package allowed to know about
both. A frame that arrives on a relay coordinator is bytes and a shape; a frame
that arrives on a worker becomes a tensor again, and this module is the seam.

The bf16 case is why the seam is worth naming. numpy cannot represent bfloat16,
so the frame carries the raw two bytes per element and this module reinterprets
them — ``view(torch.int16)`` on the way out, ``view(torch.bfloat16)`` on the way
back. Nothing is converted, widened or rounded: the bits that left the sender
are the bits the receiver gets.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from swarmpipe.wire.frames import DTYPES, TensorFrame, UnsupportedDtype

__all__ = ["frame_from_tensor", "tensor_from_frame"]

# The wire tag is the contract; this maps it onto the framework on each side.
_TORCH_BY_TAG: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
_TAG_BY_TORCH: dict[torch.dtype, str] = {v: k for k, v in _TORCH_BY_TAG.items()}

# bfloat16 has no numpy dtype, so it travels as its own bits. int16 is the
# same width and numpy can hold it, which is all the carrier has to do.
_BITS = torch.int16


def frame_from_tensor(tensor: torch.Tensor, *, step: int = 0, **meta: Any) -> TensorFrame:
    """Turn a tensor into a :class:`~swarmpipe.wire.frames.TensorFrame`.

    Detached, moved to host memory and made contiguous first: a frame's payload
    has to be the tensor the shape claims, whatever slicing or device the caller
    had it on.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to send. A view or a slice is fine.
    step : int, optional
        Pipeline step, so the reply can be matched to this message.
    **meta
        Small JSON-serialisable extras for the header - a loss value, say. Not
        a place for anything large; it is not the payload.

    Raises
    ------
    UnsupportedDtype
        For a torch dtype the wire has no tag for.
    """
    tag = _TAG_BY_TORCH.get(tensor.dtype)
    if tag is None:
        raise UnsupportedDtype(
            f"torch dtype {tensor.dtype} has no wire tag; known: {sorted(_TORCH_BY_TAG)}"
        )
    local = tensor.detach().cpu().contiguous()
    spec = DTYPES[tag]
    if spec.numpy is None:
        # Reinterpret, do not convert: same bytes, a dtype numpy will hold.
        payload = local.view(_BITS).numpy().astype("<i2", copy=False).tobytes()
    else:
        payload = local.numpy().astype(np.dtype(spec.numpy), copy=False).tobytes()
    return TensorFrame(
        dtype=tag, shape=tuple(local.shape), payload=payload, step=step, meta=dict(meta)
    )


def tensor_from_frame(
    frame: TensorFrame, *, device: Any = None, requires_grad: bool = False
) -> torch.Tensor:
    """Rebuild a tensor from a frame.

    Parameters
    ----------
    frame : TensorFrame
        A frame off the wire.
    device : optional
        Where to put the result. ``None`` leaves it on the host, which is what
        a coordinator wants and what a worker overrides.
    requires_grad : bool, optional
        Whether the rebuilt activation should start a graph on this stage. The
        receiving half of a boundary needs this; a coordinator never does.

    Raises
    ------
    UnsupportedDtype
        For a wire tag this backend has no torch dtype for.
    """
    torch_dtype = _TORCH_BY_TAG.get(frame.dtype)
    if torch_dtype is None:
        raise UnsupportedDtype(
            f"wire tag {frame.dtype!r} has no torch dtype; known: {sorted(_TORCH_BY_TAG)}"
        )
    spec = DTYPES[frame.dtype]
    carrier = "<i2" if spec.numpy is None else spec.numpy
    # A copy, because frombuffer over `bytes` is read-only and torch refuses to
    # take ownership of that - and a tensor that shares memory with a received
    # message would change under the caller when the buffer is reused.
    flat = torch.from_numpy(np.frombuffer(frame.payload, dtype=np.dtype(carrier)).copy())
    if spec.numpy is None:
        flat = flat.view(torch_dtype)
    tensor = flat.reshape(frame.shape)
    if device is not None:
        tensor = tensor.to(device)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor
