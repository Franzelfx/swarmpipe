"""Architecture adapter for manual segmented decoder execution.

The distributed pipeline (training ``distributed_stage`` and inference
``pipeline_serving``) runs the decoder *block by block* by hand so a contiguous
slice can live on a different device or node. Doing that requires reproducing the
architecture-specific glue the full ``model.forward`` would normally apply:

* the input embedding (GPT-2 adds **learned** absolute position embeddings;
  Llama/Qwen use **rotary** embeddings applied inside attention),
* the per-block context — the causal ``attention_mask`` plus, for rotary models,
  the ``position_embeddings`` (cos/sin) and ``position_ids``,
* optional **KV-cache** threading for incremental decoding.

This adapter encapsulates those differences behind one interface so the pipeline
code is architecture-agnostic. In the parent project it was verified to reproduce
the full HuggingFace forward bit-for-bit for both the GPT-2 family (learned
positions) and the Llama/Qwen family (rotary) — which is exactly what makes a
hand-segmented forward correct, and the test that shows it has not been ported
yet.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from swarmpipe.split.torch.loader import DecoderTopology, introspect_decoder


class UnsupportedArchitectureError(RuntimeError):
    """Raised when a model can't be driven by the segmented forward."""


class DecoderAdapter:
    """Architecture glue for the hand-rolled, segmented decoder forward.

    Covers the two position-embedding families the runtime supports:

    * **learned absolute** (GPT-2 ``wte + wpe``) — blocks need only the causal
      mask; and
    * **rotary** (Llama/Qwen ``embed_tokens`` + ``rotary_emb``) — blocks also need
      ``position_embeddings`` (cos/sin) and ``position_ids``.

    Build one per loaded model (it holds references into the model's modules).
    """

    def __init__(self, model: nn.Module, topo: DecoderTopology | None = None) -> None:
        self.model = model
        self.config = model.config
        self.topo = topo or introspect_decoder(model)
        self.base = self.topo.base
        self.wpe = getattr(self.base, "wpe", None)
        self.rotary_emb = getattr(self.base, "rotary_emb", None)
        self.drop = getattr(self.base, "drop", None)
        # GPT-2 family uses learned absolute position embeddings (a ``wpe``
        # module); Llama/Qwen apply rotary embeddings inside attention (no wpe).
        self.uses_learned_positions = self.wpe is not None
        if not self.uses_learned_positions and self.rotary_emb is None:
            raise UnsupportedArchitectureError(
                f"model_type={getattr(self.config, 'model_type', '?')!r} exposes neither "
                "learned position embeddings (wpe) nor a rotary embedding module; "
                "the segmented forward needs one of them."
            )

    # ------------------------------------------------------------------
    def embed(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Input ids -> hidden states (stage-0 entry point)."""
        hidden = self.topo.embed_tokens(input_ids)
        if self.uses_learned_positions:
            hidden = hidden + self.wpe(position_ids)
        if self.drop is not None:
            hidden = self.drop(hidden)
        return hidden

    def block_context(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        past_key_values: Any = None,
    ) -> dict[str, Any]:
        """Per-step kwargs shared by every block in a stage.

        Builds the causal ``attention_mask`` with the model's own mask utility
        (so it matches the active attention implementation) and, for rotary
        models, the ``position_embeddings`` + ``position_ids``.
        """
        from transformers.masking_utils import create_causal_mask

        mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        ctx: dict[str, Any] = {"attention_mask": mask}
        if not self.uses_learned_positions and self.rotary_emb is not None:
            ctx["position_embeddings"] = self.rotary_emb(hidden, position_ids)
            ctx["position_ids"] = position_ids
        return ctx

    def run_blocks(
        self,
        blocks,
        hidden: torch.Tensor,
        ctx: dict[str, Any],
        *,
        block_devices: list[str] | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Run a sequence of decoder blocks with the shared ``ctx``.

        ``block_devices`` (intra-stage multi-GPU) moves the hidden state — and the
        context tensors — onto each block's device. Returns the final hidden
        state; when ``use_cache`` the ``past_key_values`` cache is updated in place.
        """
        cur_dev: str | None = None
        local_ctx = ctx
        for i, block in enumerate(blocks):
            if block_devices is not None:
                dev = block_devices[i]
                hidden = hidden.to(dev)
                if dev != cur_dev:
                    local_ctx = _ctx_to(ctx, dev)
                    cur_dev = dev
            kwargs = dict(local_ctx)
            if use_cache:
                kwargs["past_key_values"] = past_key_values
                kwargs["use_cache"] = True
            out = block(hidden, **kwargs)
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden

    @staticmethod
    def new_cache():
        """Return a fresh KV cache for incremental decoding (a DynamicCache)."""
        from transformers import DynamicCache

        return DynamicCache()


def build_adapter(model: nn.Module, topo: DecoderTopology | None = None) -> DecoderAdapter:
    """Construct the right :class:`DecoderAdapter` for a loaded causal LM."""
    return DecoderAdapter(model, topo)


def _ctx_to(ctx: dict[str, Any], device: str) -> dict[str, Any]:
    """Move tensor entries of a block context to ``device`` (tuples handled)."""
    moved: dict[str, Any] = {}
    for key, value in ctx.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, tuple):
            moved[key] = tuple(
                t.to(device) if isinstance(t, torch.Tensor) else t for t in value
            )
        else:
            moved[key] = value
    return moved
