"""Trainable compression for the stage boundary.

Compression is two concerns, and only one of them is here. A **trainable**
compressor — a learned bottleneck — has parameters, receives gradients and is
part of the model graph, so it belongs to L1 and to torch. The **stateless**
codecs (int8/int4 packing, an fp16 cast) are pure wire formats with no
parameters and no gradient; they belong to L2 and are torch-free.

The bottleneck is split across the boundary so the narrow *code* crosses the
wire rather than the reconstruction: the down-projection runs on the sender, the
up-projection on the receiver, and they share one module's parameters.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""
