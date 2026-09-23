"""Layer 1's plan object: defaults, resolution and JSON round-trip.

Deliberately torch-free — the leader builds a spec on a box that has no PyTorch,
so this package must import and pass without it. That contract itself is checked
in ``tests/unit/test_torch_free.py``, across every layer rather than this one.
"""

import json

from swarmpipe.split import Placement, SplitSpec, StageSpec


def test_boundary_defaults_to_halfway_and_ranges_are_contiguous() -> None:
    spec = SplitSpec()
    assert spec.boundary_layer(12) == 6
    assert spec.layer_range(0, 12) == (0, 6)
    assert spec.layer_range(1, 12) == (6, 12)


def test_an_explicit_boundary_wins_over_halfway() -> None:
    spec = SplitSpec(split_layer=2)
    assert spec.layer_range(0, 12) == (0, 2)
    assert spec.layer_range(1, 12) == (2, 12)


def test_compression_is_off_for_none_and_for_an_empty_spec() -> None:
    assert not SplitSpec().uses_compression
    assert not SplitSpec(compression={"method": "none"}).uses_compression
    assert not SplitSpec(compression={"method": "NONE"}).uses_compression
    assert SplitSpec(compression={"method": "learned_bottleneck", "learned_dim": 8}).uses_compression


def test_lora_is_off_unless_explicitly_enabled() -> None:
    assert not SplitSpec().uses_lora
    assert not SplitSpec(lora={"r": 8}).uses_lora
    assert SplitSpec(lora={"enabled": True}).uses_lora


def test_placement_resolves_every_device_from_one_default() -> None:
    placement = Placement(device="cuda:0")
    assert placement.embed == placement.head == "cuda:0"
    assert placement.block_device(3) == "cuda:0"
    assert placement.first_block_device == placement.last_block_device == "cuda:0"


def test_placement_spans_local_gpus_when_block_devices_are_given() -> None:
    placement = Placement(
        device="cuda:0", block_devices=["cuda:0", "cuda:0", "cuda:1"], head_device="cuda:1"
    )
    assert placement.block_device(2) == "cuda:1"
    assert placement.first_block_device == "cuda:0"
    assert placement.last_block_device == "cuda:1"   # the sender encodes here
    assert placement.head == "cuda:1"
    assert placement.embed == "cuda:0"               # falls back to `device`


def test_stage_spec_round_trips_through_json() -> None:
    """The leader stores this in a job record; the fellow reads it back."""
    spec = StageSpec(
        stage_index=1,
        split=SplitSpec(
            split_layer=3,
            compression={"method": "learned_dim8"},
            lora={"enabled": True, "r": 4, "target_modules": ["q_proj"]},
            freeze_backbone=True,
            dtype="bfloat16",
            quantization={"bits": 4},
        ),
        placement=Placement(device="cuda:0", block_devices=["cuda:0", "cuda:1"]),
    )

    restored = StageSpec.from_dict(json.loads(json.dumps(spec.to_dict())))

    assert restored == spec


def test_from_dict_fills_in_the_defaults_for_a_sparse_payload() -> None:
    restored = StageSpec.from_dict({"stage_index": 0})
    assert restored == StageSpec(stage_index=0)
    assert restored.split.num_stages == 2
    assert restored.placement.device == "cpu"


def test_stage_position_flags() -> None:
    assert StageSpec(stage_index=0).is_first
    assert not StageSpec(stage_index=0).is_last
    assert StageSpec(stage_index=1).is_last
