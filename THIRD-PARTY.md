# Dependencies and licences

This project is licensed under Apache-2.0 and builds on existing work instead
of rebuilding it.

## Base install

What `pip install swarmpipe` brings along. This list is deliberately short: it
is what a coordinating process without a GPU has to install.

| Project | Licence | Use |
|---|---|---|
| NumPy | BSD-3 | Tensor frames in L2, without a deep-learning framework |
| PyZMQ | BSD-3 (libzmq: MPL-2.0) | Transport in L3 |

## Optional extras

Only the model surgery in L1 needs a deep-learning framework.

| Project | Extra | Licence | Use |
|---|---|---|---|
| PyTorch | `[torch]` | BSD-3 | Model surgery, autograd boundary |
| Transformers | `[torch]` | Apache-2.0 | Decoder introspection |
| Accelerate | `[torch]` | Apache-2.0 | Device placement |
| PEFT | `[lora]` | Apache-2.0 | LoRA adapters |
| pytest, ruff | `[dev]` | MIT | Tests and linting |

## Provenance of the code

The code below `seed/port/` comes from
[SilentSwarm](https://github.com/Franzelfx/nxpSilentSwarm) (branch
`feat/quantized-lora`, commit `1dec850`) and still carries the parent project's
licence headers there.

**SilentSwarm is dual-licensed** (PolyForm Noncommercial 1.0.0 plus a commercial
licence) and is not covered by Apache-2.0. Relicensing the extracted part is
possible because NexPatch AI UG holds the copyright on this code and can
release it under different terms.

In practice:

- The files under `seed/port/` keep their old headers until T0 moves them to
  `src/swarmpipe/`. They are the reference for checking that the port changed
  nothing but import paths.
- On moving, the header is replaced by the Apache-2.0 notice that the files in
  `src/swarmpipe/` already carry.
- The parent project itself stays dual-licensed, unchanged. A permissive
  library and a noncommercial application are compatible with each other.

This file is not legal advice. When unsure, seek legal advice.
