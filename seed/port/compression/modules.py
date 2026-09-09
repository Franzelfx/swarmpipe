"""Torch compression modules for the activation boundary (E5).

Each module preserves the activation shape under ``forward`` (for training/eval)
while ``encode``/``decode`` define the real wire path. The learned bottleneck and
int8 quantization genuinely reduce ``wire_nbytes`` versus an fp32 activation,
which is what makes compression worthwhile across a 1 Gbit link.

SilentSwarm — Copyright 2026 NexPatch AI UG.
Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE for details.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through gradient (identity on backward)."""
    return x + (torch.round(x) - x).detach()


class NoCompression(nn.Module):
    """Pass activations through unchanged (fp16 wire accounting)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bits_per_element = 16.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float16)

    def decode(self, payload: torch.Tensor) -> torch.Tensor:
        return payload.to(torch.float32)

    def wire_nbytes(self, x: torch.Tensor) -> int:
        return x.numel() * 2  # fp16


class FixedQuantization(nn.Module):
    """Symmetric per-tensor int quantization with a straight-through estimator."""

    def __init__(self, hidden_dim: int, bits: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bits = int(bits)
        self.bits_per_element = float(bits)
        self.levels = 2 ** self.bits

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        max_abs = x.abs().max()
        scale = max_abs / (self.levels // 2 - 1)
        return torch.clamp(scale, min=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self._scale(x)
        codes = torch.clamp(_round_ste(x / scale), -(self.levels // 2), self.levels // 2 - 1)
        return codes * scale

    def encode(self, x: torch.Tensor) -> dict[str, Any]:
        scale = self._scale(x)
        codes = torch.clamp(
            torch.round(x / scale), -(self.levels // 2), self.levels // 2 - 1
        ).to(torch.int8)
        return {"codes": codes, "scale": scale}

    def decode(self, payload: dict[str, Any]) -> torch.Tensor:
        return payload["codes"].to(torch.float32) * payload["scale"]

    def wire_nbytes(self, x: torch.Tensor) -> int:
        # int8 codes (8-bit assumed for accounting) + one fp32 scale.
        return x.numel() + 4


class LearnedBottleneck(nn.Module):
    """Down-project to a narrow code and back; optionally quantize the code.

    The wire payload is the **code** (``bottleneck_dim`` wide, optionally int8),
    not the reconstructed activation — so transport bytes scale with the
    bottleneck width, not the hidden width.
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int, quantize_bits: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.quantize_bits = quantize_bits

        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.down_norm = nn.LayerNorm(bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.up_norm = nn.LayerNorm(hidden_dim)
        self.quant = FixedQuantization(bottleneck_dim, bits=quantize_bits) if quantize_bits else None

        ratio = bottleneck_dim / hidden_dim
        bits = quantize_bits if quantize_bits is not None else 16
        self.bits_per_element = ratio * bits

    def _to_code(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_norm(self.down(x))

    def _from_code(self, z: torch.Tensor) -> torch.Tensor:
        return self.up_norm(self.up(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._to_code(x)
        if self.quant is not None:
            z = self.quant(z)  # STE
        return self._from_code(z)

    def encode(self, x: torch.Tensor) -> Any:
        z = self._to_code(x)
        if self.quant is not None:
            return self.quant.encode(z)  # int8 codes + scale
        return z.to(torch.float16)

    def decode(self, payload: Any) -> torch.Tensor:
        if self.quant is not None:
            z = self.quant.decode(payload)
        else:
            z = payload.to(torch.float32)
        return self._from_code(z)

    def wire_nbytes(self, x: torch.Tensor) -> int:
        code_elems = x.shape[:-1].numel() * self.bottleneck_dim
        if self.quantize_bits is not None:
            return code_elems + 4  # int8 codes + scale
        return code_elems * 2  # fp16 code
