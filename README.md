# swarmpipe

**Split a model across machines, and move the tensors between them.**

A library for pipeline-parallel execution on the hardware people actually have:
two or three ordinary computers, different GPUs, a consumer internet connection,
and no cluster manager.

> **Status: pre-alpha.** Layer 1 is implemented and tested: the plan objects and
> the torch backend that carries them out, `build_stage` included. Layers 2 and 3
> are specified but not written, so there is nothing to move a tensor between two
> machines yet. Follow along or open an issue if the problem is one you have.

[Deutsche Fassung](README.de.md) · [Contributing](CONTRIBUTING.md) ·
[Documentation policy](doc/README.md)

---

## The problem

A model too big for one graphics card leaves two options: rent a bigger card, or
adopt an entire distributed-training framework built for data centres. Both push
work toward centralised compute, and the second is usually heavier than the
problem — a full cluster stack to split one model across two desktop machines.

The consequences are not merely inconvenient. Researchers without an
institutional budget, small organisations, public bodies with data that may not
leave the building, and hobbyists with two gaming PCs are all shut out of work
their combined hardware could do. The hardware exists; idle GPUs are everywhere.
What is missing is the plumbing that makes several ordinary machines behave like
one larger one — packaged so that using it does not mean adopting somebody's
platform.

## The approach

A model is cut into *stages*: each machine runs a contiguous slice of the
decoder and passes the boundary activation forward and its gradient back. Three
strictly layered pieces, and each layer's contract is narrower than the one above
it:

```
swarmpipe/
  split/   L1  sees models.        Splits one, inserts the compression boundary,
                                   picks what trains. Never touches a socket.
  wire/    L2  sees tensor frames. Shape, dtype, payload, step id — and the
                                   forward/backward role protocol. Torch-free.
  link/    L3  sees list[bytes].   Direct P2P, or a relay for peers behind NAT.
                                   Nothing above bytes.
```

**Only L1's backend imports torch.** That is not tidiness for its own sake: it is
what lets a coordinator process — the thing that relays bytes between peers that
cannot reach each other — run on a small always-on box with no CUDA, no PyTorch,
and a five-second start-up.

```bash
pip install swarmpipe          # L2 + L3 + the L1 plan layer. No torch.
pip install swarmpipe[torch]   # ...plus the surgery, for a GPU worker.
```

The result is a library, not a platform: no scheduler, no control plane, no user
interface, and no opinion about who owns the machines or where the weights live.

## What is actually new here

Pipeline parallelism is not exotic, and the individual pieces are all mature.
What does not exist is this combination as a severable library: the working
implementations are welded into the systems that grew them — a training
framework, a cloud scheduler, a control plane. If you want to split a model
across two ordinary machines you either adopt someone's whole stack or write the
surgery again.

The gap is the small, heterogeneous, badly-connected case, and it needs different
things from the data-centre case: **compression on the stage boundary** because
bandwidth is the bottleneck rather than the interconnect, **a relay** because NAT
is universal on consumer connections, **per-machine precision** because the
hardware is mixed, and **no scheduler at all**, because the user is the
scheduler.

swarmpipe is that machinery on its own, extracted from a working system
([SilentSwarm](https://github.com/Franzelfx/nxpSilentSwarm)) rather than designed
in the abstract, with the seams drawn where experience says they belong. The
funded work is finishing the extraction properly.

## What L1 looks like today

The plan is a serialisable object, so the process that *decides* the split need
not be the process that *runs* it:

```python
from swarmpipe.split import Placement, SplitSpec, StageSpec

spec = StageSpec(
    stage_index=0,
    split=SplitSpec(
        split_layer=12,                                    # None = halfway
        compression={"method": "learned_bottleneck", "learned_dim": 256},
        lora={"enabled": True, "r": 8},
    ),
    placement=Placement(device="cuda:0", block_devices=["cuda:0", "cuda:1"]),
)
spec.to_dict()          # JSON, for a job record or an HTTP body
spec.layer_range(24)    # (0, 12) — this stage's blocks
```

and the surgery is one call that returns one stage, ready to run:

```python
from swarmpipe.split.torch import build_stage          # needs swarmpipe[torch]

bundle = build_stage(model, spec)
bundle.blocks                  # this stage's blocks, already on their devices
bundle.compressor              # the sender half here, the receiver half on stage 1
bundle.trainable_parameters    # exactly what this stage optimizes
bundle.model                   # the adapter-wrapped model, for your export step
```

No optimizer, no training loop, no channel. Those are the caller's opinions, and
keeping them out is what makes the split testable on a laptop with no GPU and no
second machine.

## Scope

**In scope**

- A Python library, embeddable in an existing stack
- The model surgery: split, compression boundary, LoRA, device placement
- A torch-free wire format and transport, including a NAT relay
- An end-to-end example that runs in two processes on one laptop
- A second, independent consumer of the API, as proof the seams are real

**Out of scope**

- A scheduler, a control plane, a user interface, model hosting
- Any opinion about optimizers, schedules or checkpointing — those are
  applications, and this is the layer beneath them
- A second deep-learning backend. The protocol seam stays (it is what forces
  L2/L3 to be torch-free), but nothing is built speculatively

**Claims we will not make.** Not "run a 70B model on two gaming PCs as fast as a
cloud instance." Pipeline parallelism across a consumer link is bounded by that
link, and the honest claim is narrower: work that was impossible on one machine
becomes possible on several, at a throughput we will publish rather than imply.
Compression on the boundary is lossy, and its effect on convergence is a
measurement we owe, not an assumption. Nothing here makes a slow network fast.

## Prior art

This project builds on existing work rather than replacing it. Short version:

| Project | What it does | Why this is still needed |
|---|---|---|
| [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [Colossal-AI](https://github.com/hpcaitech/ColossalAI) | Mature pipeline and tensor parallelism at scale | Built for homogeneous clusters with fast interconnects; a framework to adopt, not a library to import |
| [`torch.distributed.pipelining`](https://docs.pytorch.org/docs/stable/distributed.pipelining.html) (ex-PiPPy) | Native pipeline parallelism in PyTorch | Assumes a `torch.distributed` process group — mutually reachable ranks, no relay, no boundary compression |
| [Petals](https://github.com/bigscience-workshop/petals) / [hivemind](https://github.com/learning-at-home/hivemind) | Closest relative: model layers over a public swarm, NAT traversal | A network with a client and a protocol; swarmpipe is the splitting and transport machinery with no swarm attached |
| [exo](https://github.com/exo-explore/exo), [distributed-llama](https://github.com/b4rtaz/distributed-llama), llama.cpp RPC | Split **inference** across everyday devices | No gradients, no fine-tuning — the backward path is the hard half |
| [SilentSwarm](https://github.com/Franzelfx/nxpSilentSwarm) | The parent project this was extracted from | A platform with a scheduler and a control plane; noncommercially licensed. This is its severable core |

## Repository layout

| Path | What it holds |
|---|---|
| `src/swarmpipe/` | The library. Working today: all of `split/` — the plan objects, and `split/torch/` for the surgery. |
| `tests/unit/` | Mirrors `src/swarmpipe/`. Runs without a GPU; `pytest -m "not torch"` runs without torch installed at all. |
| `seed/origin/` | The extraction plan this repo came from, kept verbatim for provenance. |
| `doc/` | The documentation policy — what gets written down at this stage and what does not. |

## Documentation

Deliberately thin while the code is young: this page, [CONTRIBUTING.md](CONTRIBUTING.md),
the docstrings and the tests. What goes where, and what is left out on purpose,
is in [doc/README.md](doc/README.md).

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [THIRD-PARTY.md](THIRD-PARTY.md) for
dependency licensing. Note that the parent project, SilentSwarm, remains
dual-licensed under PolyForm Noncommercial and is *not* covered by this licence;
only the extracted library is.

## Contact

Issues and discussions welcome.
