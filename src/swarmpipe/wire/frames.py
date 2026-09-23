"""Layer 2: one tensor on the wire, as shape, dtype tag, payload and step id.

A :class:`TensorFrame` is what L1 hands down and L3 carries: it knows the shape
of a tensor and the bytes of it, and nothing about the framework that produced
them. That is what lets a relay coordinator move activations without importing
torch, and what makes the frame testable as bytes in, bytes out.

**The payload is bytes, not an array.** That is the decision the whole module
turns on. numpy has no bfloat16 — still true as of numpy 2.5 — and bf16 is a
dtype this library must carry, because it is what mixed consumer GPUs actually
train in. Storing an array would mean either a new dependency for one dtype, or
widening bf16 to fp32 and doubling the traffic on the one link that is already
the bottleneck. Storing bytes means bf16 needs no numpy dtype at all: two bytes
per element are two bytes per element, and the tag says how to read them.

The wire is **little-endian**, always, whatever the host is. Both ends of a link
are ordinary machines today, but "we never tested that" is a poor reason for a
format to be host-dependent, and pinning it costs nothing.

The **step id** is not bookkeeping. Without it there is no way to tell that the
gradient which just arrived answers the activation that was just sent, and on a
relayed path a mismatch after a timeout produces a silently wrong weight update
rather than an error - a run that completes and reports a plausible loss.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import prod
from typing import Any

import numpy as np

__all__ = [
    "DTYPES",
    "FrameError",
    "MalformedFrame",
    "TensorFrame",
    "UnsupportedDtype",
    "WIRE_VERSION",
]

# Bumped only for a change that an old reader could not understand. The tag
# strings and this number sit in other people's job records: adding a dtype is
# backwards-compatible, changing what an existing tag means is not.
WIRE_VERSION = 1


@dataclass(frozen=True)
class _Dtype:
    """One row of the wire dtype table."""

    tag: str
    itemsize: int
    numpy: str | None
    note: str = ""


# The tag is the contract; the numpy column is a convenience for the dtypes
# numpy happens to have. ``numpy=None`` means "numpy cannot hold this, carry the
# bits" - which is exactly and only bfloat16.
DTYPES: dict[str, _Dtype] = {
    d.tag: d
    for d in (
        _Dtype("fp32", 4, "<f4"),
        _Dtype("fp16", 2, "<f2"),
        _Dtype("bf16", 2, None, "no numpy dtype; carried as raw 2-byte elements"),
        _Dtype("fp64", 8, "<f8"),
        _Dtype("int64", 8, "<i8"),
        _Dtype("int32", 4, "<i4"),
        _Dtype("int8", 1, "|i1"),
        _Dtype("uint8", 1, "|u1"),
        _Dtype("bool", 1, "|b1"),
    )
}

# Keyed on what numpy itself calls each dtype, not on the string in the table:
# a one-byte dtype canonicalises to "|i1" rather than "<i1", since byte order
# means nothing for a single byte, and a table written the other way silently
# fails to match every int8 array that arrives.
_NUMPY_TO_TAG: dict[str, str] = {
    np.dtype(d.numpy).str: d.tag for d in DTYPES.values() if d.numpy
}


class FrameError(Exception):
    """Base class for everything this module refuses."""


class UnsupportedDtype(FrameError):
    """Raised for a dtype the wire has no tag for, or numpy has no type for."""


class MalformedFrame(FrameError):
    """Raised when bytes off the wire cannot be a frame.

    Always with what was wrong in the message. A frame that decoded a truncated
    payload into a smaller tensor would hand L1 a correctly-typed, wrongly-shaped
    activation, and the failure would surface as a matrix multiply somewhere far
    from here.
    """


@dataclass(frozen=True)
class TensorFrame:
    """One tensor, ready for a :class:`~swarmpipe.link.base.Link`.

    Attributes
    ----------
    dtype : str
        A tag from :data:`DTYPES`, not a numpy or torch dtype.
    shape : tuple of int
        The tensor's shape. C-order, like the payload.
    payload : bytes
        ``prod(shape) * itemsize`` bytes, little-endian.
    step : int
        Which pipeline step this belongs to, so a reply can be matched to it.
    meta : dict
        Room for the small things a session needs to say - a loss value with a
        gradient, for instance. JSON-serialisable, and not a place for anything
        large: it rides in the header, which is not the payload.
    """

    dtype: str
    shape: tuple[int, ...]
    payload: bytes
    step: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a frame that cannot describe a tensor."""
        if self.dtype not in DTYPES:
            raise UnsupportedDtype(
                f"unknown dtype tag {self.dtype!r}; known tags: {sorted(DTYPES)}"
            )
        if any(int(d) < 0 for d in self.shape):
            raise MalformedFrame(f"shape {self.shape} has a negative dimension")
        expected = self.nbytes
        if len(self.payload) != expected:
            raise MalformedFrame(
                f"payload is {len(self.payload)} bytes but shape {self.shape} of "
                f"{self.dtype} needs {expected}"
            )

    @property
    def itemsize(self) -> int:
        """Bytes per element for this frame's dtype."""
        return DTYPES[self.dtype].itemsize

    @property
    def nbytes(self) -> int:
        """Bytes the payload must have for this shape and dtype."""
        return prod(self.shape) * self.itemsize

    # ---- the wire ------------------------------------------------------

    def encode(self) -> list[bytes]:
        """Return ``[header, payload]``, ready for ``send_multipart``.

        Two parts, not one concatenated blob: L3 preserves part boundaries, so
        the payload never has to be copied out of a larger buffer, and the
        header stays readable when something goes wrong on a live system.
        """
        header = {
            "v": WIRE_VERSION,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "step": self.step,
        }
        if self.meta:
            header["meta"] = self.meta
        return [json.dumps(header, separators=(",", ":")).encode(), self.payload]

    @classmethod
    def decode(cls, parts: list[bytes]) -> TensorFrame:
        """Rebuild a frame from what :meth:`encode` produced.

        Raises
        ------
        MalformedFrame
            If the parts are not a frame, the header is not readable, the
            version is one this reader does not understand, or the payload
            length disagrees with the shape.
        """
        if len(parts) != 2:
            raise MalformedFrame(f"a frame is 2 parts (header, payload), got {len(parts)}")
        try:
            header = json.loads(parts[0])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MalformedFrame(f"header is not JSON: {exc}") from exc
        if not isinstance(header, dict):
            raise MalformedFrame(f"header is {type(header).__name__}, not an object")

        version = header.get("v")
        if version != WIRE_VERSION:
            raise MalformedFrame(
                f"frame is wire version {version!r}, this reader speaks {WIRE_VERSION}"
            )
        try:
            dtype = header["dtype"]
            shape = tuple(int(d) for d in header["shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedFrame(f"header is missing or has a bad shape/dtype: {exc}") from exc

        return cls(
            dtype=dtype,
            shape=shape,
            payload=bytes(parts[1]),
            step=int(header.get("step", 0)),
            meta=dict(header.get("meta") or {}),
        )

    # ---- numpy, for the dtypes numpy has -------------------------------

    @classmethod
    def from_numpy(cls, array: np.ndarray, *, step: int = 0, **meta: Any) -> TensorFrame:
        """Build a frame from a numpy array.

        The array is made C-contiguous and little-endian first, so the payload
        is what the shape says it is regardless of how the caller sliced it.
        """
        tag = _NUMPY_TO_TAG.get(array.dtype.str) or _NUMPY_TO_TAG.get(
            array.dtype.newbyteorder("<").str
        )
        if tag is None:
            raise UnsupportedDtype(
                f"numpy dtype {array.dtype!r} has no wire tag; known: {sorted(DTYPES)}"
            )
        wire = np.ascontiguousarray(array, dtype=np.dtype(DTYPES[tag].numpy))
        return cls(dtype=tag, shape=tuple(array.shape), payload=wire.tobytes(),
                   step=step, meta=dict(meta))

    def to_numpy(self) -> np.ndarray:
        """Return the payload as a numpy array.

        Raises
        ------
        UnsupportedDtype
            For ``bf16``, which numpy cannot represent. Read it with the torch
            adapter in :mod:`swarmpipe.split.torch.frames`, or take
            :meth:`raw_bits` if the bit pattern is what you want.
        """
        spec = DTYPES[self.dtype]
        if spec.numpy is None:
            raise UnsupportedDtype(
                f"numpy has no {self.dtype} dtype ({spec.note}); use the torch "
                "adapter, or raw_bits() for the bit pattern"
            )
        array = np.frombuffer(self.payload, dtype=np.dtype(spec.numpy))
        return array.reshape(self.shape)

    def raw_bits(self) -> np.ndarray:
        """Return the payload as unsigned integers of this frame's item size.

        The escape hatch for a dtype numpy cannot hold: the bits are intact and
        the shape is right, but arithmetic on the result is meaningless.
        """
        width = {1: "<u1", 2: "<u2", 4: "<u4", 8: "<u8"}[self.itemsize]
        return np.frombuffer(self.payload, dtype=np.dtype(width)).reshape(self.shape)
