"""Check that the docstring rules reach public symbols and stop at private ones.

``CONTRIBUTING.md`` asks for a NumPy docstring on every public symbol and for
nothing on the private helpers, and ``pyproject.toml`` turns that into ruff's
``D`` rules. The part worth a test is the boundary: ``D1xx`` treats a leading
underscore as private and skips it, which is the whole reason the rule can be
enforced without turning every one-caller helper into a docstring chore. That
behaviour comes from ruff rather than from this repository, so it is checked
here rather than assumed — if a future version flagged private helpers too, the
convention would change under us without anyone deciding it.

swarmpipe — Copyright 2026 NexPatch AI UG.
Licensed under the Apache License 2.0. See LICENSE.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("ruff") is None, reason="needs ruff from the dev extra"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# One public and one private symbol of every kind D1xx distinguishes. The marker
# comment on a line is what the rules are expected to do with that symbol; the
# tests below read the markers rather than hard-coded line numbers.
FIXTURE = '''\
"""Fixture module."""


def public_function():  # flagged
    return 1


def _private_function():  # skipped
    return 1


class PublicClass:  # documented below, so that its methods are reached at all
    """A class."""

    def public_method(self):  # flagged
        return 1

    def _private_method(self):  # skipped
        return 1


class _PrivateClass:  # skipped, and so is every symbol nested inside it
    def public_method(self):  # skipped
        return 1
'''


def _lines_marked(marker: str) -> set[int]:
    """Return the 1-based fixture lines whose trailing comment is ``marker``."""
    return {
        number
        for number, line in enumerate(FIXTURE.splitlines(), start=1)
        if line.endswith(f"# {marker}") or f"# {marker}," in line
    }


@pytest.fixture
def findings(tmp_path: Path) -> list[dict]:
    """Run ruff over the fixture under this repository's own configuration.

    The fixture is written outside ``tests/`` on purpose: ``tests/**`` ignores
    the ``D`` rules, so a fixture living there would report nothing and the test
    would pass for the wrong reason. No ``--select`` is passed either, so the
    run also proves that ``pyproject.toml`` enables the rules at all.
    """
    module = tmp_path / "sample.py"
    module.write_text(FIXTURE)
    result = subprocess.run(
        [
            "ruff",
            "check",
            "--no-cache",
            "--config",
            str(PROJECT_ROOT / "pyproject.toml"),
            "--output-format",
            "json",
            str(module),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout, f"ruff produced no report: {result.stderr}"
    return [f for f in json.loads(result.stdout) if (f.get("code") or "").startswith("D")]


def test_public_symbols_are_flagged_when_undocumented(findings: list[dict]) -> None:
    codes = {f["code"] for f in findings}
    assert "D103" in codes, "a public function without a docstring must be reported"
    assert "D102" in codes, "a public method without a docstring must be reported"


def test_private_symbols_are_left_alone(findings: list[dict]) -> None:
    skipped = _lines_marked("skipped")
    assert skipped, "the fixture must contain private symbols to be meaningful"
    reported = {f["location"]["row"] for f in findings}
    assert not (reported & skipped), (
        "the D rules reached a private symbol; the convention in CONTRIBUTING.md "
        "assumes they stop at the leading underscore"
    )


def test_every_public_symbol_in_the_fixture_is_accounted_for(findings: list[dict]) -> None:
    reported = {f["location"]["row"] for f in findings}
    assert reported == _lines_marked("flagged")
