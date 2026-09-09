"""Insert a compressor at a layer boundary via a forward hook (E5).

Replacing a decoder block's output hidden state with the compressed/reconstructed
one needs no model rewrite — it attaches to any block (the E0 spike pattern,
generalized to tuple and tensor block outputs).

SilentSwarm — Copyright 2026 NexPatch AI UG.
Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE for details.
"""

from __future__ import annotations

import torch.nn as nn


def attach_compressor(block: nn.Module, compressor: nn.Module):
    """Register a forward hook on ``block`` that compresses its hidden output.

    Parameters
    ----------
    block : nn.Module
        The decoder block after which compression is applied.
    compressor : nn.Module
        A compression module whose ``forward`` maps hidden->hidden.

    Returns
    -------
    torch.utils.hooks.RemovableHandle
        Call ``.remove()`` to detach the compressor.
    """

    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (compressor(output[0]), *output[1:])
        return compressor(output)

    return block.register_forward_hook(_hook)
