"""Kernel purity guards — machine-checked, not remembered.

Two guard classes, each answering a failure this project has actually suffered:

1. **World-agnosticism** (docs/PLAN.md: "世界無關"): the kernel may not name a
   world's institutions. When ECC's `co_share` / `debt_stress` / `sigma_soviet`
   were written straight into `FastChannelConfig`, the v1.6 promise — "the *form*
   is supplied by the substrate, the structure is world-agnostic" — was broken
   silently. Nothing failed; the docs and the code just diverged. This test fails.

2. **Layering**: `kernel` is the portable core; domain models live in
   `spectrum_os.extensions`. Kernel must never import upward. When GNP/oligopoly
   models were dropped into `kernel/`, they grew to exceed the engine itself.

A new world word in the kernel is not a style issue — it means some other
worldline cannot reuse the kernel without inheriting that world's concepts.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

KERNEL_DIR = Path(__file__).resolve().parents[1] / "spectrum_os" / "kernel"

# Vocabulary from the ECC (β₁) worldline and its vocabulary of institutions.
# Word-boundary anchored: short tokens like "rlo" would otherwise match "Carlo"
# (as in "Monte Carlo"), and a guard that cries wolf gets switched off.
WORLD_WORDS = (
    r"ECC",
    r"RLO",
    r"DSR",
    r"DSF",
    r"RSFSR",
    r"PSR",
    r"NSPV",
    r"dkk",
    r"soviet",
    r"soviet",
    r"coop\w*",          # cooperative / CoopShare / cooperation
    r"usury",
    r"kolkhoz",
    r"almanac",
    r"村庄|合作社|高利貸|哥達",
)

_COMPILED = [(p, re.compile(rf"\b{p}\b" if p.isascii() else p, re.IGNORECASE))
             for p in WORLD_WORDS]

# Files with a documented, reviewed exemption. Every entry needs a reason — a
# bare filename would make the guard decorative.
ALLOWED: dict[str, str] = {}


def _kernel_files() -> list[Path]:
    return sorted(p for p in KERNEL_DIR.glob("*.py") if p.name != "__pycache__")


def test_kernel_files_exist():
    assert _kernel_files(), f"no kernel modules found under {KERNEL_DIR}"


@pytest.mark.parametrize("path", _kernel_files(), ids=lambda p: p.name)
def test_kernel_names_no_world_institutions(path: Path):
    if path.name in ALLOWED:
        pytest.skip(f"documented exemption: {ALLOWED[path.name]}")
    offenders: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for label, rx in _COMPILED:
            m = rx.search(line)
            if m:
                offenders.append(f"  L{lineno}: {label!r} in: {line.strip()[:88]}")
    assert not offenders, (
        f"{path.name} names a world's institutions — the kernel must stay "
        "world-agnostic (the *form* comes from the substrate):\n"
        + "\n".join(offenders)
        + "\n\nIf the concept is genuinely universal, rename it by its mechanic "
        "(e.g. 'cooperative penetration' -> 'phi_drive'); if it is world-specific, "
        "it belongs in the substrate layer, not here."
    )


@pytest.mark.parametrize("path", _kernel_files(), ids=lambda p: p.name)
def test_kernel_does_not_import_upward(path: Path):
    """kernel must not import extensions (layering: extensions -> kernel only)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "spectrum_os.extensions"):
            offenders.append(f"  L{node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("spectrum_os.extensions"):
                    offenders.append(f"  L{node.lineno}: import {alias.name}")
    assert not offenders, (
        f"{path.name} imports the domain layer — layering violation "
        "(extensions may import kernel; never the reverse):\n" + "\n".join(offenders)
    )


def test_kernel_stays_small_enough_to_read():
    """Size budget: the kernel is the portable core, so growth must be a decision.

    When the GNP / oligopoly / frigorífico domain models were added to kernel/,
    they totalled 1,832 lines — more than `forces.py` itself (1,785). Domain
    models belong in `spectrum_os.extensions`. Exceeding the budget is not
    forbidden, but it must be argued for (raise the number deliberately).
    """
    BUDGET = 700  # per-module lines; forces.py is the documented exception
    EXEMPT = {"forces.py": "the R/C/P_R engine itself (spec-implementing, not a domain model)"}
    oversize = [
        f"  {p.name}: {len(p.read_text(encoding='utf-8').splitlines())} lines"
        for p in _kernel_files()
        if p.name not in EXEMPT
        and len(p.read_text(encoding="utf-8").splitlines()) > BUDGET
    ]
    assert not oversize, (
        f"kernel module(s) exceed the {BUDGET}-line budget — is this a domain "
        "model that belongs in spectrum_os.extensions?\n" + "\n".join(oversize)
    )
