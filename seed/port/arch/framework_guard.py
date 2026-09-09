"""Enforce a single deep-learning framework per process.

TensorFlow (the legacy Keras stack) and PyTorch (the migration target) must not
coexist in one process: they fight over CUDA/cuDNN initialization and GPU memory.
During the migration both packages may be installed at once, so this guard makes
the *runtime* selection explicit and fails fast if a process tries to use both.

Usage (called at the top of any torch runtime entrypoint)::

    from silent_swarm.runtime.framework_guard import ensure_single_framework
    ensure_single_framework("torch")
    import torch  # safe: TensorFlow is now forbidden in this process

SilentSwarm — Copyright 2026 NexPatch AI UG.
Licensed under the PolyForm Noncommercial License 1.0.0. See LICENSE for details.
"""

from __future__ import annotations

import sys
from typing import Literal

Framework = Literal["torch", "tensorflow"]

_OTHER: dict[str, str] = {"torch": "tensorflow", "tensorflow": "torch"}

# Set once the first framework is claimed in this process.
_selected: str | None = None


class FrameworkConflictError(RuntimeError):
    """Raised when a process tries to use both torch and TensorFlow."""


def _is_imported(module_name: str) -> bool:
    """Return whether ``module_name`` is already imported in this process."""
    return module_name in sys.modules


def active_framework() -> str | None:
    """Return the framework claimed in this process, or ``None`` if unset."""
    return _selected


def ensure_single_framework(framework: Framework) -> None:
    """Claim ``framework`` for this process and forbid the other one.

    Parameters
    ----------
    framework : {"torch", "tensorflow"}
        The framework this process intends to use.

    Raises
    ------
    ValueError
        If ``framework`` is not a recognized value.
    FrameworkConflictError
        If the other framework has already been imported or claimed.
    """
    global _selected

    if framework not in _OTHER:
        raise ValueError(f"Unknown framework {framework!r}; expected one of {sorted(_OTHER)}.")

    other = _OTHER[framework]

    if _selected is not None and _selected != framework:
        raise FrameworkConflictError(
            f"This process already claimed framework {_selected!r}; cannot also use "
            f"{framework!r}. Run torch and TensorFlow workloads in separate processes."
        )

    if _is_imported(other):
        raise FrameworkConflictError(
            f"Cannot select {framework!r}: the conflicting framework {other!r} is already "
            f"imported in this process. Run them in separate processes."
        )

    _selected = framework


def reset_for_testing() -> None:
    """Clear the claimed framework. Intended for tests only."""
    global _selected
    _selected = None
