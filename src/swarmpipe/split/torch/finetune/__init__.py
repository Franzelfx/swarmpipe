"""Choosing what a stage trains: adapters, or a frozen backbone.

Which parameters a stage optimizes is part of the split, not of the training
loop: a stage that trains adapters over its own blocks must agree with its peer
on the adapter config, and a stage that freezes its backbone trains only the
boundary compressor. Both decisions are made while the model is being cut, so
they live with the surgery.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""
