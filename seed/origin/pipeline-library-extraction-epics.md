# Extracting the Pipeline Machinery into a Reusable Library — Epics

Working title of the library: **`swarmpipe`** (see [Open decisions](#open-decisions)).

> **Goal.** Separate the *machinery* — splitting a model, framing tensors, moving
> them between machines — from the *application* — SilentSwarm's leader, fellow,
> CLI and web UI. The machinery ships as a pip-installable distribution, first
> from this repo, later from its own, so other projects can depend on it without
> pulling in a control plane.

## Context

Today the three layers exist, but as **two competing half-implementations**, and
the most valuable one — the model surgery — is not a library at all: it is inline
inside a 300-line training function.

### Layer 1 — model splitting + compression insertion

| Piece | Where | State |
|---|---|---|
| Layer→stage and layer→GPU planning | [plan.py](../../src/silent_swarm/runtime/sharding/plan.py) | Clean, pure Python, already testable without a GPU. Lifts as-is. |
| `accelerate` device-map builder | [device_map.py](../../src/silent_swarm/runtime/sharding/device_map.py) | Torch-adjacent; belongs to the torch backend of L1. |
| Compressor modules + factory + hook | [compression/](../../src/silent_swarm/runtime/compression/) | Torch `nn.Module`s. Mixes two concerns — see below. |
| Bottleneck split across the boundary | [split.py](../../src/silent_swarm/runtime/compression/split.py) `split_bottleneck` | Exactly right, keep. |
| **The actual surgery** | [distributed_stage.py:135-230](../../src/silent_swarm/runtime/torch/distributed_stage.py) | **Inline.** `run_torch_stage()` does introspection, split, LoRA injection, compressor splitting, device placement, optimizer construction *and* the training loop in one function. There is no callable "split this model" API. |

### Layer 2 — tensor exchange, and Layer 3 — the transport path

Two stacks from two eras, both alive:

| Stack | Modules | Used by | Torch-free? |
|---|---|---|---|
| Legacy | [distributed/protocol.py](../../src/silent_swarm/distributed/protocol.py) (`ActivationPacket`, `CompressionMetadata`, `TransportAdapter`) + [zmq_transport.py](../../src/silent_swarm/transport/zmq_transport.py) (PUSH/PULL) | [runner.py:486](../../src/silent_swarm/fellow/runner.py) only | Yes |
| Current | [pipeline/serialization.py](../../src/silent_swarm/runtime/pipeline/serialization.py) + [pipeline/channel.py](../../src/silent_swarm/runtime/pipeline/channel.py) (`PipeChannel` → ZMQ `PAIR` / HTTP relay) + [pipeline/relay.py](../../src/silent_swarm/runtime/pipeline/relay.py) | the real training and serving path | No — `serialization.py` imports torch |

The legacy stack is *already* the three-layer design the extraction wants, and it
is the one nobody uses. The current stack works but blurs L2 into L3:
`ZmqPipeChannel` knows both the tensor format and the socket, and its frame
encoding imports torch — which is why L2/L3 cannot be shared with the leader as
they stand.

### The one boundary this document draws differently

**Compression is not one concern, it is two**, and they belong on different layers:

* **Trainable compression** (`LearnedBottleneck`, and its sender/receiver halves)
  has parameters, receives gradients, and is part of the model graph → **Layer 1**.
* **Stateless codecs** (`FixedQuantization`'s int4/int8 packing, an fp16 cast,
  optionally zstd) are pure wire formats → **Layer 2**.

[factory.py](../../src/silent_swarm/runtime/compression/factory.py)'s
`compressor_from_spec()` currently returns both kinds from one call, and
[modules.py](../../src/silent_swarm/runtime/compression/modules.py) gives every
module an `encode`/`decode`/`wire_nbytes` triple regardless. Splitting them makes
L2 testable as bytes-in/bytes-out and leaves L1 with no opinion about bandwidth.

`PipelineBoundary` ([boundary.py](../../src/silent_swarm/runtime/pipeline/boundary.py))
is then precisely the **L1→L2 adapter**, and the only piece in the chain that
needs to know about autograd.

---

## Target shape

```
swarmpipe/
  split/            L1   spec.py    SplitSpec / StageSpec (serialisable, framework-free)
                         plan.py    (from runtime/sharding/plan.py)
                         api.py     build_stage(model, spec) -> StageBundle   [Protocol]
                    L1   torch/     the torch realisation (surgery, LoRA, placement)
  wire/             L2   frames.py  TensorFrame + encode/decode (numpy, torch-free)
                         codec.py   int8 / int4 / fp16 / raw codecs
                         session.py role protocol: activation forward, (loss, grad) back
  link/             L3   base.py       Link: send_multipart / recv_multipart / close
                         zmq_link.py   direct P2P
                         relay_link.py HTTP relay through a coordinator
                         resolve.py    endpoint resolution + failover (mesh-aware, E7)
```

Layer contracts, narrowest first:

* **L3 sees `list[bytes]`.** Nothing else. No tensors, no dtypes, no steps.
* **L2 sees `TensorFrame`** (shape, dtype, payload) and the role protocol on top
  of it. **Torch-free**, numpy only.
* **L1 sees models.** It never touches a socket.

L2 and L3 staying torch-free is not cosmetic: it is what lets the leader serve
the relay without importing PyTorch, the rule
[test_leader_is_torch_free.py](../../tests/unit/leader/test_leader_is_torch_free.py)
already enforces for the control plane. The same guard test is extended to the
library in T1.

### What stays in SilentSwarm

The application keeps everything that knows about *this* product: the leader
control plane, job/pool/bucket state, the fellow's lifecycle and heartbeat, the
CLI, the web UI, the model catalogue, and the policy decisions (which node gets
which stage, which dtype the cluster can run). SilentSwarm becomes a **consumer**
of `swarmpipe`, passing it a `SplitSpec` and a `Link`.

---

## Task table

Branches follow the repo convention (`feat/…`, `chore/…`), one per task, each
merging to `main` on its own. Sizes: S ≈ a day, M ≈ 2-4 days, L ≈ a week+.

| # | Task | Branch | Depends on | Size | Ships |
|---|---|---|---|---|---|
| **T1** | `Link` protocol + one transport stack | `feat/pipe-link-layer` | — | M | L3 exists; the legacy `TransportAdapter`/`ZmqTransport` duplication is gone |
| **T2** | Torch-free wire frames + session | `feat/pipe-wire-layer` | T1 | M | L2 exists, importable without torch |
| **T3** | Stateless codecs moved out of the compressors | `feat/pipe-wire-codecs` | T2 | S | int4/int8 packing is a wire codec, not a model module |
| **T4** | `build_stage()` extracted from `run_torch_stage()` | `feat/pipe-split-layer` | T3 | **L** | L1 exists; model surgery is unit-testable without a cluster |
| **T5** | `SplitSpec` as a serialisable contract | `feat/pipe-split-spec` | T4 | M | leader and fellow exchange a spec instead of nine keyword arguments |
| **T6** | In-repo distribution + import guard | `feat/pipe-package-split` | T5 | M | `pip install swarmpipe` works; `silent_swarm` depends on it |
| **T7** | Docs, API reference, examples | `docs/pipe-library` | T6 | S | the library is usable by someone who has never seen SilentSwarm |
| **T8** | Second consumer proves reusability | `feat/pipe-reference-consumer` | T6 | M | a standalone example that uses none of `silent_swarm` |
| **T9** | Repo extraction with history + PyPI release | `chore/pipe-repo-extract` | T7, T8 | M | own repo, own CI, tagged release |

**Recommended order: T1 → T2 → T3 → T4, then reassess.**

T1-T4 are worth doing even if the library never leaves this repo — they remove a
duplicated transport stack and make the split testable. T5-T9 are the packaging
arc and only pay off once someone actually wants the second consumer; do not
start T9 before T8 has proven the API survives contact with a different caller.

---

## T1 — `Link` protocol and one transport stack

**Branch:** `feat/pipe-link-layer`

**Goal.** Define L3 and collapse the two implementations into one.

1. Add `Link` — `send_multipart(list[bytes])`, `recv_multipart() -> list[bytes]`,
   `close()`, plus the byte counters `ZmqPipeChannel` already carries
   (`on_bytes_sent` / `on_bytes_recv`, which feed pool telemetry).
2. Reshape `ZmqPipeChannel` and `RelayPipeChannel` as `Link` implementations that
   no longer encode tensors — encoding moves up to L2 in T2.
3. Decide the fate of the legacy stack. `ZmqTransport` is used only by
   [runner.py:486](../../src/silent_swarm/fellow/runner.py) for the receiver-side
   sockets; port that call site to the `Link` implementation and delete
   `distributed/protocol.py` + `transport/zmq_transport.py`, or keep exactly one
   of them as the PUSH/PULL `Link` variant. **Do not leave both alive** — they
   have already drifted once.

**Acceptance.** `tests/unit/runtime/link/` covers both implementations against the
same contract test. [test_tunnel_relay_transport.py](../../tests/integration/test_tunnel_relay_transport.py)
passes unchanged. No module under the link layer imports torch.

**Risk.** The relay wire format is shared with the leader endpoints
(`/api/v1/relay/{job_id}/{channel}/push|pop`, classified in
[route_policy.py:110](../../src/silent_swarm/leader/route_policy.py)) and with the
length-prefixed multipart encoding in `relay.py`. Changing framing here breaks a
live cluster mid-job; keep the bytes on the wire identical.

## T2 — Torch-free wire frames and the session protocol

**Branch:** `feat/pipe-wire-layer`

**Goal.** Define L2 so it can be imported without torch.

1. Move `tensor_to_frame` / `frame_to_tensor` to numpy, as `TensorFrame`. The
   torch↔numpy conversion becomes a thin adapter in the torch backend of L1.
2. Add `Session`: the role object (`upstream` / `downstream`) that owns the
   forward-activation / `(loss, gradient)`-reply protocol currently split between
   `channel.py` and `boundary.py`.
3. Carry a **step id** in the frame metadata. The relay path today has no way to
   pair an activation with its gradient; a mismatch after a timeout produces a
   silently wrong update rather than an error.
4. Keep `encode_loss_grad`'s habit of stuffing `loss` into the metadata dict, or
   give the reply its own frame — either is fine, but document which.

**Acceptance.** A test asserts `swarmpipe.wire` imports with `torch` absent from
`sys.modules`, mirroring `test_leader_is_torch_free.py`. Round-trip tests for
every dtype in use (fp32/fp16/bf16 — note bf16 has no numpy dtype and needs an
explicit representation; this is the one real trap in T2).

## T3 — Stateless codecs move to L2

**Branch:** `feat/pipe-wire-codecs`

**Goal.** Apply the split described in [Context](#the-one-boundary-this-document-draws-differently).

1. `FixedQuantization`'s scale-and-pack logic becomes `wire/codec.py` codecs
   operating on `TensorFrame`.
2. `compressor_from_spec()` keeps returning only trainable modules
   (`LearnedBottleneck`, `NoCompression`); the `fixed_intN` methods resolve to a
   wire codec instead. The **spec strings stay unchanged** — existing experiment
   configs and job records use them.
3. `wire_nbytes()` moves with the codecs; bandwidth accounting becomes an L2
   property.

**Acceptance.** Existing compression tests
([test_compression.py](../../tests/unit/runtime/torch/test_compression.py)) still
pass, with the fixed-quantization cases rewritten against the codec API. Numeric
equivalence: the same spec produces the same bytes before and after.

**Note.** The straight-through estimator in `_round_ste` only matters for
*trainable* quantization; a pure wire codec has no gradient to estimate. Check
whether any current config relies on `FixedQuantization` being differentiable
before moving it, or the change is silently a behaviour change.

## T4 — `build_stage()` extracted from `run_torch_stage()`

**Branch:** `feat/pipe-split-layer`

**Goal.** The largest and most valuable task: make the surgery a library call.

Split [run_torch_stage](../../src/silent_swarm/runtime/torch/distributed_stage.py)
in two:

* `build_stage(model, spec) -> StageBundle` — introspection, block slicing, LoRA
  injection scoped to this stage's blocks, compressor construction and splitting,
  device placement of embeddings/blocks/head, trainable-parameter selection.
  Returns the modules and the parameter list. **No optimizer, no loop, no channel.**
* `train_stage(bundle, session, ...)` — the loop, which stays in SilentSwarm at
  first and moves into the library only if T8 shows a second consumer wants it.

Keep `_require_trainable`'s hard failure: an optimizer over an empty parameter
list runs every step and changes nothing, and LoRA makes that reachable through a
`target_modules` that matches nothing on this stage.

**Acceptance.** [test_distributed_stage.py](../../tests/unit/runtime/torch/test_distributed_stage.py)
passes **unmodified** — it is the regression suite for this refactor. New tests
call `build_stage()` directly and assert the block ranges, the placement, and
which parameters came back trainable, with no channel and no cluster.

**Risk.** This touches the most heavily tested and most recently changed code in
the repo (the whole `feat/quantized-lora` series). Rebase pain is real; do T4 on
top of a merged `main`, not in parallel with more runtime work.

## T5 — `SplitSpec` as a serialisable contract

**Branch:** `feat/pipe-split-spec`

**Goal.** Replace the nine-keyword-argument call signature with one object that
the leader can construct, store in a job record, and hand to a fellow.

`SplitSpec` covers: stage index and count, the boundary layer, per-block device
placement, the compression spec, the LoRA config, dtype and quantization. It is a
dataclass or Pydantic model with no torch import, so it can live in the library
and still be built by the torch-free leader.

**Acceptance.** The leader builds a `SplitSpec`; the fellow consumes it; the job
record round-trips it through JSON. Both sides agree on the fingerprint the two
stages must match on (the LoRA config already has this requirement).

**Note.** This overlaps [contracts.py](../../src/silent_swarm/shared/contracts.py).
Decide whether `SplitSpec` lives in the library and SilentSwarm re-exports it, or
whether SilentSwarm keeps its own and converts — the former is cleaner, the latter
avoids a dependency from the control plane to the library.

## T6 — In-repo distribution and the import guard

**Branch:** `feat/pipe-package-split`

**Goal.** `swarmpipe` becomes its own installable distribution while still living
in this repo — the low-risk rehearsal for T9.

1. `src/swarmpipe/` with its own `pyproject.toml`; `torch` is an **optional
   extra** (`swarmpipe[torch]`), so L2/L3 install on a control-plane box.
2. `silent_swarm` declares `swarmpipe` as a dependency and imports it rather than
   its own copy.
3. A guard test asserting the dependency arrow points one way only: nothing under
   `swarmpipe/` may import `silent_swarm`. This is the test that keeps the
   extraction from rotting before T9.
4. `requirements.lock` regenerated via `scripts/lock_deps.sh`; the Docker leader
   image builds from a context that now needs both source trees.

**Acceptance.** `pip install -e src/swarmpipe` in a clean venv, then
`import swarmpipe.wire, swarmpipe.link` with no torch installed. Full test suite
green. `docker compose build leader` succeeds.

## T7 — Docs, API reference, examples

**Branch:** `docs/pipe-library`

Per the repo rule that features ship with docs: an architecture page for the three
layers, NumPy-style docstrings on every public symbol
([docstrings.md](../04_contributing/docstrings.md)), and a runnable two-process
example that splits a small model, trains a few steps over a local ZMQ link, and
prints the loss. `mkdocs build --strict` must stay clean.

## T8 — A second consumer

**Branch:** `feat/pipe-reference-consumer`

**Goal.** Prove the API is actually reusable before freezing it in its own repo.

Build something in `exp/` that uses `swarmpipe` and **imports nothing from
`silent_swarm`** — the honest test of whether the seams are real. The obvious
candidate: split a model across two processes with a hand-written coordinator,
no leader, no fellow, no job records.

Every place the example has to reach around the API is a T4/T5 bug. Fix those
before T9; after the repo split, fixing them costs a cross-repo release cycle.

## T9 — Repo extraction and PyPI release

**Branch:** `chore/pipe-repo-extract`

1. `git subtree split` (or `git filter-repo`) on `src/swarmpipe/` +
   `tests/unit/swarmpipe/` to carry the history — the LoRA and compression work
   has archaeology worth keeping.
2. Own CI: ruff, pytest, docs build. The frontend jobs do not follow.
3. License header and `LICENSE` per the [licensing decision](#open-decisions).
4. Tag `v0.1.0`, publish to PyPI, pin it in SilentSwarm's requirements, delete
   `src/swarmpipe/` here.

**Acceptance.** SilentSwarm's suite passes against the published wheel, not a path
install.

---

## Open decisions

These block T6/T9, not T1-T4. Decide them before T6.

| # | Decision | Notes |
|---|---|---|
| D1 | **Licence** | The repo is PolyForm Noncommercial 1.0.0 + a commercial licence. PolyForm-NC forbids commercial use, which makes the library unusable in a commercial project — including possibly your own. NexPatch holds the copyright and can release the library under different terms (Apache-2.0 or MPL-2.0 are the usual choices for infrastructure). **Decide this before publishing**, since relicensing after external contributions arrive requires their agreement. |
| D2 | **Name** | `swarmpipe` is a working title. Check PyPI availability. A name that does not say "swarm" ages better if the library outlives SilentSwarm. |
| D3 | **Does the training loop go in?** | T4 leaves `train_stage` in the application. T8 answers whether the library needs it. Shipping a loop means shipping an opinion about optimizers and schedules. |
| D4 | **Framework scope** | L1's torch backend is the only one that will exist. Keep the `Protocol` seam from [interfaces.py](../../src/silent_swarm/runtime/interfaces.py) anyway — it costs nothing and it is what makes L2/L3 torch-free — but do not build a second backend speculatively. |
| D5 | **Does `SplitSpec` live in the library?** | See T5. Determines whether the torch-free control plane depends on the library at all, or only the GPU workers do. |

## What this does not change

* The relay wire format and the leader's relay endpoints stay byte-identical.
* `plan_gpu_layers` and the leader's VRAM-based placement policy are untouched —
  policy stays in the application, the library only executes a plan.
* The fellow bootstrap contract is not in scope.
