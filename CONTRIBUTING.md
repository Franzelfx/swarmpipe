# Contributing

The project is pre-alpha: Layer 1 stands, Layers 2 and 3 are specified but not
written. The most useful contributions right now:

- **Counter-arguments to the cut.** If you have split a model across ordinary
  machines before and know where it fails, that is worth more than code.
- **Pointers to prior work we missed.** See the prior-art table in
  `README.md`. If something is missing there, please open an issue.
- **Reports from real hardware.** Mixed GPUs, consumer connections, NAT on both
  ends. That exact case is the one we cannot fully reproduce ourselves.

## What this is about

`swarmpipe` — the machinery for splitting a model across several machines and
moving the tensors between them, extracted from the SilentSwarm project. A
library, not a platform: no scheduler, no control plane, no user interface.

Before your first change, read `README.md`, then this page to the end — the
rules and pitfalls below are a list of things that have already gone wrong
once.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,torch,lora]"     # GPU machine, or CPU torch for tests
pytest -q                              # everything
pytest -m "not torch" -q               # only what must run without a DL framework
ruff check .
```

For work on the wire or transport layer, please use a **second environment
without torch installed** — that is the only way to notice when the torch-free
contract has been broken:

```bash
pip install -e ".[dev]"
pytest -m "not torch" -q
```

## Workflow

Work starts as an issue, not as a branch. Before the first commit, the issue
must have:

- a **done** checklist
- a milestone for the funded period, one milestone per two-month block
- exactly one workflow label: `feat`, `fix`, `docs`, `chore` or `spike`

If a task looks larger than about a week, split it first and branch only from a
piece with a clear checklist.

Branch names are:

```text
<type>(<domain>)/<number>-<slug>
```

`<type>` is one of `feat`, `fix`, `docs`, `chore`, `spike`. `<domain>` names
the layer or repository surface: `split`, `wire`, `link`, `seed`, `readme`,
`ci`, `release`. The only exceptions are the long-lived branches `main` and
`develop`.

Pull request titles and commit titles are:

```text
<type>(<domain>): #<number>: <description>
```

Use the imperative mood and keep the whole title at or below 72 characters. The
pull request title becomes the squash commit title, so write it once and keep
it final.

Everything lands through a pull request. `main` stays protected: no direct
pushes, no force pushes, only green CI, linear history, and squash merges.

## Layout

| Path | What lives there |
|---|---|
| `src/swarmpipe/split/` | L1. Sees models. `spec.py`/`api.py`/`plan.py` are torch-free; `torch/` is the backend. |
| `src/swarmpipe/wire/` | L2. Sees tensor frames. **Torch-free by rule.** Not yet implemented (T2). |
| `src/swarmpipe/link/` | L3. Sees `list[bytes]`. **Torch-free by rule.** Not yet implemented (T1). |
| `tests/unit/` | Mirrors `src/swarmpipe/`. |
| `seed/port/` | Code from the parent project, awaiting the T0 port. Not importable yet — see [seed/README.md](seed/README.md). |
| `seed/origin/` | The original extraction plan, verbatim. Provenance only; the valid plan is the milestones of the project sketch. |
| `doc/` | Documentation policy: what gets written down at this stage and what does not. |

## The rules that are not style

1. **Nothing below `wire/` or `link/` may import torch**, neither directly nor
   through another module. That is what the whole layering exists for: a
   coordinator must be able to relay traffic without a GPU stack. Enforced in
   CI; whoever breaks it fixes it with an adapter in L1, not with an exception.
   Checked through a subprocess with an import blocker, not in-process — the
   test suite imports torch elsewhere, and an in-process check would run
   against an already loaded module and pass.
2. **L1 never touches a socket.** If a splitting function needs a channel, the
   design is wrong.
3. **Dependencies point in one direction**: `split` → `wire` → `link`. The
   upper layer may import the lower one, never the reverse.
4. **Nothing here imports the parent project.** `grep -rn "silent_swarm" src/
   tests/` must stay empty.
5. **Wire formats are compatibility surfaces.** Framing and compression spec
   strings sit in other people's job records. Changing their *meaning* is a
   breaking change even when nothing fails locally.
6. **An empty list of trainable parameters is a hard error**, not a warning.
   An optimiser over nothing runs every step, reports a loss and changes
   nothing.
7. **Every feature ships with tests and documentation in the same change.**
   Inherited from the parent project, and the reason its refactorings are
   survivable.

## Tests

Mirror the source tree: `tests/unit/<package>/` for `src/swarmpipe/<package>/`.
Mark everything that needs the torch extra with `@pytest.mark.torch`.

Prefer tests that need no GPU. That goes further than it looks: device
placement can be checked against the `meta` device, which exists on every
machine, and the splitting logic needs no cluster at all.

## Style

Line length 100. `ruff` for linting and import order. NumPy docstrings on every
public symbol, starting with one line of purpose. Everything in English —
documents, code, docstrings, commit messages. `README.de.md` is the one
deliberate translation. What gets documented and what does not:
[doc/README.md](doc/README.md).

New dependencies only with a justification — the base install stays light, see
`THIRD-PARTY.md`. Extend a module rather than add an abstraction: the parent
project's transport layer acquired two competing implementations exactly that
way.

Comments explain the *why*, especially where the code is defensive: most guards
in this library exist because something failed in production, and a comment
naming the failure keeps the guard from being "simplified" away later.

## Wording

Not "as fast as a cluster" and not "unlimited scaling". Correct is: work that
was impossible on a single machine becomes possible on several, at a throughput
the network connection bounds and we measure. Compression at the stage boundary
is lossy. This applies to code comments, docstrings and error messages as much
as to the documentation.

## Pitfalls

- Always move modules to their target device unconditionally, not only with
  several GPUs — otherwise the embeddings stay on the CPU while the inputs are
  already on `cuda:0`.
- bfloat16 has no numpy dtype. A torch-free L2 needs an explicit
  representation for it; tests on fp32 only will not find this.
- Never hard-code the dtype — it is a property of the plan, not of the code.
- Do not change the relay framing or the compression spec strings: they sit in
  job records of running systems.

## What does not belong here

No model weights, no checkpoints, no personal data. Nothing from the parent
project beyond what already sits in `seed/` — and that stays unchanged until
the port.
