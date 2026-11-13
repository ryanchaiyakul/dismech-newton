"""End-to-end physical validation of every runnable scene under ``examples/`` and ``examples/newton/``: each example
runs through its physical horizon (invoking ``test_post_step()`` along the way when defined) and must pass its
analytical / physical ``test_final()`` checks. Marked ``slow``: ``uv run pytest -m slow``."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import newton
import newton.examples
import newton.viewer
import pytest
import warp as wp

ROOT = Path(__file__).resolve().parents[1] / "examples"

# (example path under examples/, frames, extra attributes on args, optional dependency)
EXAMPLES = [
    ("cantilever", 240, {}, None),
    ("flagella", 180, {}, None),
    ("midas_axial", 475, {}, "mujoco_warp"),
    ("mpm_rod_sand", 90, {}, None),
    ("mpm_spoon_sand", 96, {}, None),
    ("overhand_knot", 1140, {}, None),
    ("piper_midas_routing", 72, {}, "mujoco_warp"),
    ("plectoneme", 5040, {}, None),
    ("shell_cantilever", 240, {}, None),
    ("shell_fold", 120, {}, None),
    ("newton/cable_bundle_hysteresis", 180, {}, None),
    ("newton/cable_bundle_hysteresis", 180, {"no_dahl": True}, None),
    ("newton/cable_cross_slide_table", 150, {}, "mujoco_warp"),
    ("newton/cable_pile", 100, {}, None),
    ("newton/cable_plectoneme", 780, {}, None),
    ("newton/cable_twist", 100, {}, None),
    ("newton/cable_y_junction", 120, {}, None),
    ("newton/cloth_bending", 600, {}, None),
    ("newton/cloth_franka_fold", 720, {}, "mujoco_warp"),
    ("newton/cloth_hanging", 300, {}, None),
    ("newton/cloth_rigid_trampoline", 300, {}, "mujoco_warp"),
    ("newton/cloth_style3d", 100, {}, "git"),
    ("newton/cloth_twist", 300, {}, None),
    ("newton/franka_cable_ik_pick_place", 330, {}, "mujoco_warp"),
]


def _load(name: str):
    """The example's module by its path under ``examples/`` (``newton/`` would shadow the package by name)."""
    path = ROOT / f"{name}.py"
    for folder in (path.parent, ROOT):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_example_is_registered():
    """Every runnable script in ``examples/*.py`` and ``examples/newton/*.py`` has a ``test_final`` check here."""
    found = {p.relative_to(ROOT).with_suffix("").as_posix() for p in (*ROOT.glob("*.py"), *(ROOT / "newton").glob("*.py"))}
    registered = {name for name, *_ in EXAMPLES}
    assert registered == found, f"unregistered or missing examples: {found ^ registered}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "name, frames, overrides, extra",
    EXAMPLES,
    ids=[f"{n}[no-dahl]" if ov.get("no_dahl") else n for n, _, ov, _ in EXAMPLES],
)
def test_example_physical_checks(name, frames, overrides, extra):
    if extra is not None:
        pytest.importorskip(extra)
    if extra == "mujoco_warp" and not wp.get_device().is_cuda:
        pytest.skip("SolverMuJoCo coupled examples require CUDA")
    module = _load(name)
    base_args = newton.examples.create_parser().parse_args(["--viewer", "null", "--num-frames", str(frames)])
    args = SimpleNamespace(**{**vars(base_args), "test": True, **overrides})
    viewer = newton.viewer.ViewerNull(num_frames=frames)
    example = module.Example(viewer, args)
    post = getattr(example, "test_post_step", None)
    for _ in range(frames):
        example.step()
        if post is not None:
            post()
    example.test_final()
