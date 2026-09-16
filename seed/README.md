# `seed/` — material from the parent project

Nothing here is on the import path. It is provided, not shipped.

## `port/`

Working, tested code from
[SilentSwarm](https://github.com/Franzelfx/nxpSilentSwarm) (branch
`feat/quantized-lora`, commit `1dec850`), **copied verbatim, with the original
headers and licence lines unchanged**. T0 moves it to `src/swarmpipe/` and
rewrites its imports; the file manifest is the tree under `port/` itself.

It is deliberately kept byte-identical: it is the reference for checking that
the port changed nothing but import paths. For the same reason it is excluded
from linting.

The licence headers name PolyForm Noncommercial — that is the parent project's
licence. The extracted library is licensed Apache-2.0; the headers are replaced
on moving, not before. Background: [../THIRD-PARTY.md](../THIRD-PARTY.md).

The centrepiece is `port/split_torch/stage_builder.py`: the model surgery,
already pulled out into a single library call with a plain data argument,
together with `port/split_torch/test_stage_builder.py`, its 19-test, CPU-only
suite.

## `origin/`

`pipeline-library-extraction-epics.md` — the extraction plan written inside the
parent project, verbatim, for provenance. Its internal links point at the
parent repository and do not resolve here.

**It is not the valid plan.** The valid plan is the milestones of the project
sketch and this repository's issues; there the work is reorganised for a
standalone repository. Read `origin/` for the rationale only.
