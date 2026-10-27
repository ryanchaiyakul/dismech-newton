"""The examples (but the fits, which take minutes) built with the null viewer and stepped a few frames: they still
run, and stay finite. ~30 s, so ``slow``: ``uv run pytest -m slow``."""

import importlib

import newton
import newton.examples
import numpy as np
import pytest

EXAMPLES = ["cable_pile", "cable_twist", "cantilever", "flagella", "overhand_knot", "plectoneme", "shell_cantilever",
            "shell_fold"]


@pytest.mark.slow
@pytest.mark.usefixtures("needs_cuda")
@pytest.mark.parametrize("name", EXAMPLES)
def test_example_runs(name):
    module = importlib.import_module(name)
    args = newton.examples.create_parser().parse_args(["--viewer", "null", "--num-frames", "3"])
    example = module.Example(newton.viewer.ViewerNull(num_frames=3), args)
    for _ in range(3):
        example.step()
        example.render()
    state = example.state_0 if hasattr(example, "state_0") else example.sim.state_0  # flagella: its simulation's
    assert np.isfinite(state.particle_q.numpy()).all()
