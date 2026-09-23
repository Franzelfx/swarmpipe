"""Freeze a pretrained backbone, leaving only chosen modules trainable.

Used for compressor calibration: the inserted compression layers (which are
separate ``nn.Module`` objects, not part of the HF model) stay trainable while
all pretrained weights are frozen.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


def freeze_backbone(model: nn.Module, trainable: Iterable[nn.Module]) -> list[torch.nn.Parameter]:
    """Freeze ``model`` and enable grads only on ``trainable`` modules.

    Parameters
    ----------
    model : nn.Module
        The pretrained model to freeze.
    trainable : iterable of nn.Module
        Modules to keep trainable (e.g. the boundary compressors).

    Returns
    -------
    list of nn.Parameter
        The parameters left trainable (for the optimizer).
    """
    for param in model.parameters():
        param.requires_grad_(False)

    trainable_params: list[torch.nn.Parameter] = []
    for module in trainable:
        for param in module.parameters():
            param.requires_grad_(True)
            trainable_params.append(param)
    return trainable_params
