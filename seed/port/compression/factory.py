"""Build a torch compressor from a JSON compression spec (E5).

Mirrors the Keras ``compression_factory_from_spec`` so existing experiment
configs (``none``, ``fixed_intN``, ``learned_dimN[_intM|_fp16]``,
``learned_bottleneck``) work unchanged on the torch runtime.

SilentSwarm — Copyright 2026 NexPatch AI UG.
Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE for details.
"""

from __future__ import annotations

import re

import torch.nn as nn

from .modules import FixedQuantization, LearnedBottleneck, NoCompression


def compressor_from_spec(hidden_dim: int, compression_spec: dict | None) -> nn.Module:
    """Create a compression ``nn.Module`` for one boundary.

    Parameters
    ----------
    hidden_dim : int
        Hidden width of the activation crossing the boundary.
    compression_spec : dict or None
        Compression profile (same schema as the Keras path).

    Raises
    ------
    ValueError
        For an unsupported method or a learned bottleneck missing its dimension.
    """
    spec = compression_spec or {"method": "none"}
    method = str(spec.get("method", "none")).strip().lower()
    quantized = bool(spec.get("quantized", False))
    quantization_bits = spec.get("quantization_bits")

    if method == "none":
        return NoCompression(hidden_dim)

    if method.startswith("fixed_int"):
        digits = re.findall(r"\d+", method)
        bits = int(quantization_bits) if quantization_bits is not None else (int(digits[0]) if digits else 8)
        return FixedQuantization(hidden_dim, bits=bits)

    if method.startswith("learned_dim"):
        match = re.match(r"learned_dim(\d+)(?:_(int(\d+)|fp16))?$", method)
        if not match:
            raise ValueError(f"Unsupported learned compression method: {method}")
        bottleneck_dim = int(match.group(1))
        suffix = match.group(2)
        quantize_bits = int(match.group(3)) if suffix and suffix.startswith("int") else None
        if quantized and quantization_bits is not None:
            quantize_bits = int(quantization_bits)
        return LearnedBottleneck(hidden_dim, bottleneck_dim, quantize_bits=quantize_bits)

    if method == "learned_bottleneck":
        bottleneck_dim = spec.get("learned_dim") or spec.get("bottleneck_dim")
        if bottleneck_dim is None:
            raise ValueError("Compression method 'learned_bottleneck' requires 'learned_dim'")
        quantize_bits = int(quantization_bits) if (quantized and quantization_bits is not None) else None
        return LearnedBottleneck(hidden_dim, int(bottleneck_dim), quantize_bits=quantize_bits)

    raise ValueError(f"Unsupported compression method: {method}")
