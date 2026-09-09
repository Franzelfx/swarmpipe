"""Layer-to-stage and layer-to-GPU planning. Pure Python: no GPU, no torch.

Lifted from SilentSwarm's ``tests/unit/runtime/torch/test_sharding_plan.py``;
the ``build_device_map`` cases stayed behind with the torch backend (T0).
"""


def test_weighted_split_sums_and_proportions() -> None:
    from swarmpipe.split.plan import weighted_split

    # 12 layers, free VRAM 15:5 -> 9:3; equal -> 6:6.
    assert weighted_split(12, [15.0, 5.0]) == [9, 3]
    assert weighted_split(12, [1.0, 1.0]) == [6, 6]
    # Largest-remainder still sums exactly.
    assert sum(weighted_split(13, [7.0, 3.0, 1.0])) == 13


def test_stage_layer_ranges() -> None:
    from swarmpipe.split.plan import stage_layer_ranges

    assert stage_layer_ranges(12, 2) == [(0, 6), (6, 12)]
    # Remainder goes to the last stage.
    assert stage_layer_ranges(13, 2) == [(0, 6), (6, 13)]


def test_plan_gpu_layers_equal_vs_weighted() -> None:
    from swarmpipe.split.plan import plan_gpu_layers

    weighted = plan_gpu_layers(0, 12, [15.0, 5.0], mode="weighted")
    equal = plan_gpu_layers(0, 12, [15.0, 5.0], mode="equal")

    assert [(s.gpu_index, s.layer_start, s.layer_end) for s in weighted] == [(0, 0, 9), (1, 9, 12)]
    assert [(s.gpu_index, s.layer_start, s.layer_end) for s in equal] == [(0, 0, 6), (1, 6, 12)]


def test_plan_skips_zero_layer_gpus() -> None:
    from swarmpipe.split.plan import plan_gpu_layers

    # 1 layer, 2 GPUs -> only one span.
    spans = plan_gpu_layers(0, 1, [10.0, 10.0], mode="equal")
    assert len(spans) == 1


def test_plan_block_devices_equal_vs_weighted() -> None:
    from swarmpipe.split.plan import plan_block_devices

    # 4 blocks across 2 GPUs (free VRAM 3:1) -> weighted [0,0,0,1]; equal [0,0,1,1].
    assert plan_block_devices(4, [3.0, 1.0], mode="weighted") == [
        "cuda:0", "cuda:0", "cuda:0", "cuda:1",
    ]
    assert plan_block_devices(4, [10.0, 10.0], mode="equal") == [
        "cuda:0", "cuda:0", "cuda:1", "cuda:1",
    ]


def test_spans_from_leader_placement() -> None:
    from swarmpipe.split.plan import spans_from_placement

    placement = {
        "gpu_slots": [
            {"gpu_index": 0, "layer_start": 0, "layer_end": 8},
            {"gpu_index": 1, "layer_start": 8, "layer_end": 12},
        ]
    }
    spans = spans_from_placement(placement)
    assert [(s.gpu_index, s.num_layers) for s in spans] == [(0, 8), (1, 4)]
