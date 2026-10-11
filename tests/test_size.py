"""The package stays small: the line budget of the solver unification."""

from pathlib import Path

import pytest

BUDGET = 3000
SRC = Path(__file__).resolve().parents[1] / "src" / "dismech_newton"


@pytest.mark.xfail(reason="the old solvers are still in src until the unified one replaces them", strict=False)
def test_package_within_line_budget():
    lines = {p.relative_to(SRC).as_posix(): len(p.read_text(encoding="utf-8").splitlines()) for p in SRC.rglob("*.py")}
    total = sum(lines.values())
    assert total <= BUDGET, f"src has {total} lines (budget {BUDGET}): " + ", ".join(
        f"{k} {v}" for k, v in sorted(lines.items(), key=lambda kv: -kv[1]))
