# `seed/` — material from the parent project

Nothing here is on the import path. It is provided, not shipped.

## `port/` — gone, and that is the point

It held working code from
[SilentSwarm](https://github.com/Franzelfx/nxpSilentSwarm) (branch
`feat/quantized-lora`, commit `1dec850`), copied verbatim with the original
PolyForm headers, so it could be shown that the port changed nothing but import
paths. That port has happened: the code lives under `src/swarmpipe/split/torch/`
with Apache-2.0 headers, and its suite runs in CI as
`tests/unit/split/torch/test_stage_builder.py`.

The verbatim copies remain in this repository's history. `git log --follow` on
any ported file reaches them, which is where the comparison goes now.

One piece did not come across: the model **loading** half of `loader.py`
(`load_causal_lm`, `build_quantization_config`, `full_precision_modules`). It
depends on a quantization module that was never copied into `seed/`, so there
was nothing to port it against — and the split takes a model that is already
loaded, so nothing in the library needs it.

## `origin/`

`pipeline-library-extraction-epics.md` — the extraction plan written inside the
parent project, verbatim, for provenance. Its internal links point at the parent
repository and do not resolve here.

**It is not the valid plan.** The valid plan is the milestones of the project
sketch and this repository's issues; there the work is reorganised for a
standalone repository. Read `origin/` for the rationale only.
