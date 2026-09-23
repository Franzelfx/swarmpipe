"""The torch-free contract: what a coordinator can install without a GPU stack.

Rule 1 in ``CONTRIBUTING.md``: nothing below ``wire/`` or ``link/`` may import
torch, neither directly nor through another module. The same holds for the plan
half of ``split/`` — a control plane builds a spec that a GPU worker executes,
and it does so on a box with no CUDA and no PyTorch. Only the torch backend of
L1 is exempt, and it is an optional extra.

Two properties matter and both are checked here: that the modules *import*
without torch, and that the plan layer is *usable* without it — importing and
then failing on the first call would satisfy the letter of the rule and none of
its purpose.

Both run in a subprocess. The unit suite imports torch elsewhere, so an
in-process check would run against an already-loaded module and pass while the
rule is broken — which is exactly how the rule can rot unnoticed.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

# The blocker every probe installs. `find_spec` and not `find_module`: the
# latter was removed from the import system in Python 3.12, so a finder that
# only defines it is never consulted and blocks nothing at all.
#
# TorchImported deliberately does not inherit from ImportError. A module that
# guards its torch import with `try: import torch except ImportError` would
# otherwise swallow the block, and the violation would go unreported.
_BLOCKER = '''
    import sys

    class TorchImported(BaseException):
        pass

    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] == "torch":
                raise TorchImported(name)
            return None

    sys.meta_path.insert(0, _Blocker())
'''

# Walks the package from disk rather than importing it to enumerate: importing
# is what we are testing, and the torch backend would abort the walk. A module
# added tomorrow is covered the day it lands, without anyone updating a list.
_IMPORT_PROBE = textwrap.dedent(
    _BLOCKER
    + '''
    import importlib
    import json
    import pathlib

    import swarmpipe

    ROOT = pathlib.Path(swarmpipe.__path__[0])
    BACKEND = "swarmpipe.split.torch"

    def torch_free_modules():
        for layer in ("wire", "link", "split"):
            for path in sorted((ROOT / layer).rglob("*.py")):
                parts = path.relative_to(ROOT).with_suffix("").parts
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                name = ".".join(("swarmpipe", *parts))
                if name == BACKEND or name.startswith(BACKEND + "."):
                    continue
                yield name

    checked, violations = [], []
    for name in sorted(set(torch_free_modules())):
        checked.append(name)
        try:
            importlib.import_module(name)
        except TorchImported as exc:
            violations.append({"module": name, "reached_for": str(exc)})

    # Backstop for anything the blocker cannot see, such as a C extension that
    # pulls torch in sideways: torch must not be in sys.modules either way.
    leaked = sorted({m.split(".")[0] for m in sys.modules} & {"torch"})
    print(json.dumps({"checked": checked, "violations": violations, "leaked": leaked}))
    '''
)

# Importing without torch is not the promise; building a spec without torch is.
_USABILITY_PROBE = textwrap.dedent(
    _BLOCKER
    + '''
    from swarmpipe.split import Placement, SplitSpec, StageSpec
    from swarmpipe.split.plan import plan_block_devices

    spec = StageSpec(
        stage_index=1,
        split=SplitSpec(split_layer=12, compression={"method": "learned_bottleneck"}),
        placement=Placement(device="cuda:0", block_devices=plan_block_devices(4, [1.0, 1.0])),
    )
    assert spec.layer_range(24) == (12, 24)
    assert StageSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
    assert "torch" not in sys.modules
    print("ok")
    '''
)

# A floor, not a list: the check must not pass because it found nothing to
# check. New modules are still picked up without touching this.
_MUST_BE_COVERED = frozenset(
    {
        "swarmpipe.link",
        "swarmpipe.split",
        "swarmpipe.split.api",
        "swarmpipe.split.plan",
        "swarmpipe.split.spec",
        "swarmpipe.wire",
    }
)


def _run(probe: str) -> subprocess.CompletedProcess[str]:
    """Run ``probe`` in a clean interpreter and return the finished process."""
    return subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=180
    )


def test_the_torch_free_layers_import_with_torch_unavailable() -> None:
    result = _run(_IMPORT_PROBE)
    assert result.returncode == 0, f"the probe itself failed:\n{result.stderr}"

    report = json.loads(result.stdout)
    missing = _MUST_BE_COVERED - set(report["checked"])
    assert not missing, (
        f"the walk did not reach {sorted(missing)}, so a clean result means nothing; "
        "check that the package layout still matches this test"
    )

    offenders = [f"{v['module']} reached for {v['reached_for']}" for v in report["violations"]]
    assert not offenders, (
        "these modules import torch, directly or through another module:\n  "
        + "\n  ".join(offenders)
        + "\n\nRule 1 in CONTRIBUTING.md: nothing below wire/ or link/ — and nothing in "
        "the plan half of split/ — may import torch. A coordinator relays bytes with no "
        "GPU stack. Fix it with an adapter in L1's torch backend, not with an exception."
    )
    assert not report["leaked"], f"torch reached sys.modules anyway: {report['leaked']}"


def test_the_plan_layer_is_usable_with_torch_unavailable() -> None:
    result = _run(_USABILITY_PROBE)
    assert result.returncode == 0, f"the plan layer needs torch to be used:\n{result.stderr}"
    assert "ok" in result.stdout
