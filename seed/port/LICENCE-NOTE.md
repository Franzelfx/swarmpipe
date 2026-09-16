# On the licence header in these files

The `.py` files in this directory carry the header

> SilentSwarm — Copyright 2026 NexPatch AI UG.
> Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE for details.

The "LICENSE" in it means the `LICENSE` of the **parent repository**
[SilentSwarm](https://github.com/Franzelfx/nxpSilentSwarm), not this
repository's. The copies here are kept byte-identical so that it can be shown
the T0 port changed nothing but import paths — which is why the headers stay
until then.

**The library itself is licensed Apache-2.0**, see `../../LICENSE` and
[../../THIRD-PARTY.md](../../THIRD-PARTY.md). NexPatch AI UG holds the copyright
on this code and releases it under Apache-2.0 for the extracted library; on
moving to `src/swarmpipe/` in T0 the header is replaced accordingly.

Nothing in this directory is on the import path, and none of it is included in
a built package. Anyone who wants to use the code now takes it from `src/` — or
waits for T0.
