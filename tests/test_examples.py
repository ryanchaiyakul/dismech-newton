"""The examples (but the fits, which take minutes) built with the null viewer and stepped a few frames: they still
run, and stay finite. ~30 s, so ``slow``: ``uv run pytest -m slow``."""

import importlib.util
import sys
from pathlib import Path

import newton
import newton.examples
import numpy as np
import pytest

EXAMPLES = ["cantilever", "flagella", "overhand_knot", "plectoneme", "shell_cantilever", "shell_fold",
            "newton/cable_bundle_hysteresis", "newton/cable_pile",
            "newton/cable_cross_slide_table", "newton/cable_plectoneme", "newton/cable_twist",
            "newton/cable_y_junction",
            "newton/cloth_bending", "newton/cloth_hanging", "newton/cloth_rigid_trampoline", "newton/cloth_twist"]
ROOT = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str):
    """The example's module by its path under ``examples/`` (``newton/`` would shadow the package by name)."""
    path = ROOT / f"{name}.py"
    sys.path.insert(0, str(path.parent))  # siblings it imports
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
@pytest.mark.usefixtures("needs_cuda")
@pytest.mark.parametrize("name", EXAMPLES)
def test_example_runs(name):
    module = _load(name)
    args = newton.examples.create_parser().parse_args(["--viewer", "null", "--num-frames", "3"])
    example = module.Example(newton.viewer.ViewerNull(num_frames=3), args)
    for _ in range(3):
        example.step()
        example.render()
    state = example.state_0 if hasattr(example, "state_0") else example.sim.state_0  # flagella: its simulation's
    assert np.isfinite(state.particle_q.numpy()).all()
