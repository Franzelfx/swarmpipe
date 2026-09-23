"""Layer 1: ``build_stage`` splits a model with no channel and no cluster.

The regression suite the surgery arrived with: it was written against the same
code in the parent project, where the split was inlined in a training function
and none of this could be checked without a channel and a peer. It passes here
unchanged, which is what makes the port reviewable. Everything runs on CPU
in-process: block ranges, module ownership, device placement and which
parameters came back trainable.

Placement is checked against the ``meta`` device - it is the one device besides
``cpu`` that exists on any machine, so "did this module actually move where the
spec said" is testable without a GPU.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

import pytest

# The surgery needs the torch extra; `pytest -m "not torch"` skips the lot.
pytestmark = pytest.mark.torch


def _gpt2(n_layer=4):
    import torch
    from transformers import AutoModelForCausalLM, GPT2Config

    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(
        GPT2Config(n_layer=n_layer, n_embd=32, n_head=4, n_positions=32, vocab_size=64)
    ).eval()


def _stage(index, **split_kwargs):
    from swarmpipe.split import Placement, SplitSpec, StageSpec

    placement = split_kwargs.pop("placement", Placement(device="cpu"))
    return StageSpec(
        stage_index=index, split=SplitSpec(**split_kwargs), placement=placement
    )


def _ids(params):
    return {id(p) for p in params}


# ---- what each stage owns -------------------------------------------------

def test_the_two_stages_own_disjoint_contiguous_block_ranges() -> None:
    from swarmpipe.split.torch.stage_builder import build_stage

    upstream = build_stage(_gpt2(), _stage(0, split_layer=1))
    downstream = build_stage(_gpt2(), _stage(1, split_layer=1))

    assert (upstream.layer_start, upstream.layer_end) == (0, 1)
    assert (downstream.layer_start, downstream.layer_end) == (1, 4)
    assert len(upstream.blocks) == 1 and len(downstream.blocks) == 3
    assert upstream.num_layers + downstream.num_layers == 4


def test_the_boundary_defaults_to_halfway_and_is_reported_back_resolved() -> None:
    """A caller that asked for "halfway" can see where halfway landed."""
    from swarmpipe.split.torch.stage_builder import build_stage

    bundle = build_stage(_gpt2(), _stage(0))

    assert bundle.layer_end == 2
    assert bundle.spec.split.split_layer == 2


def test_only_the_first_stage_owns_the_embeddings_and_only_the_last_the_head() -> None:
    from swarmpipe.split.torch.stage_builder import build_stage

    upstream = build_stage(_gpt2(), _stage(0))
    downstream = build_stage(_gpt2(), _stage(1))

    assert upstream.embed_tokens is not None
    assert upstream.lm_head is None and upstream.final_norm is None
    assert downstream.lm_head is not None and downstream.final_norm is not None
    assert downstream.embed_tokens is None
    assert upstream.is_first and downstream.is_last


def test_an_unsupported_pipeline_shape_is_refused() -> None:
    from swarmpipe.split.torch.stage_builder import build_stage

    with pytest.raises(NotImplementedError):
        build_stage(_gpt2(), _stage(0, num_stages=3))
    with pytest.raises(NotImplementedError):
        build_stage(_gpt2(), _stage(2))


# ---- device placement -----------------------------------------------------

def test_blocks_land_on_their_own_device_and_the_embeddings_on_theirs() -> None:
    """One stage's slice can span several local GPUs (P3)."""
    from swarmpipe.split import Placement
    from swarmpipe.split.torch.stage_builder import build_stage

    bundle = build_stage(
        _gpt2(),
        _stage(0, split_layer=2, placement=Placement(
            device="cpu", block_devices=["cpu", "meta"], embed_device="cpu",
        )),
    )

    assert next(bundle.blocks[0].parameters()).device.type == "cpu"
    assert next(bundle.blocks[1].parameters()).device.type == "meta"
    assert next(bundle.embed_tokens.parameters()).device.type == "cpu"


def test_the_head_follows_head_device_not_the_blocks() -> None:
    from swarmpipe.split import Placement
    from swarmpipe.split.torch.stage_builder import build_stage

    bundle = build_stage(
        _gpt2(),
        _stage(1, split_layer=2, placement=Placement(
            device="cpu", block_devices=["cpu", "cpu"], head_device="meta",
        )),
    )

    assert next(bundle.lm_head.parameters()).device.type == "meta"
    assert next(bundle.final_norm.parameters()).device.type == "meta"


def test_the_compressor_halves_sit_where_they_are_used() -> None:
    """The sender encodes after the last block, the receiver decodes before the first."""
    from swarmpipe.split import Placement
    from swarmpipe.split.torch.stage_builder import build_stage

    compression = {"method": "learned_bottleneck", "learned_dim": 8}
    upstream = build_stage(
        _gpt2(),
        _stage(0, split_layer=2, compression=compression, placement=Placement(
            device="cpu", block_devices=["cpu", "meta"],
        )),
    )
    downstream = build_stage(
        _gpt2(),
        _stage(1, split_layer=2, compression=compression, placement=Placement(
            device="cpu", block_devices=["meta", "cpu"],
        )),
    )

    assert next(upstream.compressor.parameters()).device.type == "meta"
    assert next(downstream.compressor.parameters()).device.type == "meta"


# ---- compression insertion ------------------------------------------------

def test_a_learned_bottleneck_is_split_across_the_boundary() -> None:
    """Only the narrow code crosses the wire, not the reconstruction."""
    import torch

    from swarmpipe.split.torch.compression.split import BottleneckReceiver, BottleneckSender
    from swarmpipe.split.torch.stage_builder import build_stage

    compression = {"method": "learned_bottleneck", "learned_dim": 8}
    upstream = build_stage(_gpt2(), _stage(0, compression=compression))
    downstream = build_stage(_gpt2(), _stage(1, compression=compression))

    assert isinstance(upstream.compressor, BottleneckSender)
    assert isinstance(downstream.compressor, BottleneckReceiver)

    hidden = torch.randn(1, 4, 32)
    code = upstream.compressor(hidden)
    assert code.shape[-1] == 8                       # genuine wire reduction
    assert downstream.compressor(code).shape == hidden.shape


def test_an_uncompressed_boundary_leaves_both_stages_without_a_compressor() -> None:
    from swarmpipe.split.torch.stage_builder import build_stage

    for spec in ({}, {"method": "none"}):
        assert build_stage(_gpt2(), _stage(0, compression=spec)).compressor is None
        assert build_stage(_gpt2(), _stage(1, compression=spec)).compressor is None


def test_a_non_bottleneck_compressor_applies_whole_on_the_sender() -> None:
    """Fixed quantization has no receiver half to reconstruct with."""
    from swarmpipe.split.torch.compression.modules import FixedQuantization
    from swarmpipe.split.torch.stage_builder import build_stage

    compression = {"method": "fixed_int8"}
    assert isinstance(
        build_stage(_gpt2(), _stage(0, compression=compression)).compressor, FixedQuantization
    )
    assert build_stage(_gpt2(), _stage(1, compression=compression)).compressor is None


# ---- trainable-parameter selection ----------------------------------------

def test_full_finetuning_trains_this_stage_and_nothing_of_the_other() -> None:
    from swarmpipe.split.torch.stage_builder import build_stage

    model = _gpt2()
    bundle = build_stage(model, _stage(0, split_layer=2))
    trainable = _ids(bundle.trainable_parameters)

    expected = _ids(bundle.embed_tokens.parameters())
    expected |= _ids(bundle.adapter.wpe.parameters())      # GPT-2 learned positions
    expected |= {id(p) for b in bundle.blocks for p in b.parameters()}
    assert trainable == expected

    # The downstream half is somebody else's problem.
    other_half = {id(p) for b in list(bundle.topology.blocks)[2:] for p in b.parameters()}
    assert trainable.isdisjoint(other_half)


def test_the_last_stage_trains_its_blocks_the_norm_and_the_head() -> None:
    from swarmpipe.split.torch.stage_builder import build_stage

    bundle = build_stage(_gpt2(), _stage(1, split_layer=2))
    trainable = _ids(bundle.trainable_parameters)

    assert trainable >= _ids(bundle.lm_head.parameters())
    assert trainable >= _ids(bundle.final_norm.parameters())
    assert trainable >= {id(p) for b in bundle.blocks for p in b.parameters()}


def test_freeze_backbone_trains_only_the_boundary_compressor() -> None:
    """The memory win: ~4 B/param for a frozen backbone vs ~16 for full FT."""
    from swarmpipe.split.torch.stage_builder import build_stage

    model = _gpt2()
    bundle = build_stage(
        model,
        _stage(0, freeze_backbone=True,
               compression={"method": "learned_bottleneck", "learned_dim": 8}),
    )

    assert all(not p.requires_grad for p in model.parameters())
    assert _ids(bundle.trainable_parameters) == _ids(bundle.compressor.parameters())


def test_freeze_backbone_without_a_compressor_falls_back_to_full_finetuning() -> None:
    """There would be nothing left to train otherwise, and an optimizer over an
    empty parameter list runs every step and changes nothing."""
    from swarmpipe.split.torch.stage_builder import build_stage

    model = _gpt2()
    bundle = build_stage(model, _stage(0, freeze_backbone=True))

    assert bundle.trainable_parameters
    assert any(p.requires_grad for p in model.parameters())


def test_lora_adapters_are_injected_into_this_stage_s_blocks_only() -> None:
    pytest.importorskip("peft")
    from swarmpipe.split.torch.stage_builder import build_stage

    bundle = build_stage(
        _gpt2(), _stage(0, split_layer=2, lora={"enabled": True, "r": 4})
    )

    assert bundle.lora["target_modules"] == ["c_attn"]       # resolved from the model
    assert bundle.trainable_parameters
    own = {id(p) for b in bundle.blocks for p in b.parameters()}
    assert _ids(bundle.trainable_parameters) <= own
    # Adapters only - the frozen base weights are not in the optimizer.
    assert all(p.requires_grad for p in bundle.trainable_parameters)
    assert len(bundle.trainable_parameters) < len(own)


def test_lora_adds_the_compressor_to_the_optimizer_alongside_the_adapters() -> None:
    pytest.importorskip("peft")
    from swarmpipe.split.torch.stage_builder import build_stage

    bundle = build_stage(
        _gpt2(),
        _stage(0, lora={"enabled": True, "r": 4},
               compression={"method": "learned_bottleneck", "learned_dim": 8}),
    )

    assert _ids(bundle.trainable_parameters) >= _ids(bundle.compressor.parameters())


def test_lora_hands_back_the_wrapped_model_for_the_export_step() -> None:
    pytest.importorskip("peft")
    from swarmpipe.split.torch.stage_builder import build_stage

    model = _gpt2()
    bundle = build_stage(model, _stage(0, lora={"enabled": True, "r": 4}))

    assert bundle.model is not model


def test_a_lora_target_that_matches_nothing_fails_loudly() -> None:
    """An optimizer over no parameters would run every step and change nothing.

    Whichever complains first wins: peft rejects a target that matches no module
    at all, and ``_require_trainable`` catches the cases it lets through.
    """
    pytest.importorskip("peft")
    from swarmpipe.split.torch.stage_builder import build_stage

    with pytest.raises(Exception) as excinfo:
        build_stage(
            _gpt2(),
            _stage(0, lora={"enabled": True, "r": 4, "target_modules": ["not_a_module"]}),
        )
    message = str(excinfo.value).lower()
    assert "no trainable parameters" in message or "target" in message


def test_a_stage_with_nothing_to_train_is_refused() -> None:
    """The guard itself, reached without peft: freeze everything by hand."""
    from swarmpipe.split.torch import stage_builder

    with pytest.raises(RuntimeError, match="no trainable parameters"):
        stage_builder._require_trainable([], stage_index=1, use_lora=False)
