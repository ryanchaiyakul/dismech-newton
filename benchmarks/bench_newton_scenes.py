"""Tier 2 end-to-end steady-state ``ms/frame`` benchmarks for canonical ``examples/newton/*.py`` ports."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import newton.examples
import newton.viewer
import pytest
import warp as wp

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

NEWTON_SCENES = [
    ("cable_twist", None),
    ("cable_pile", None),
    ("cable_plectoneme", None),
    ("cable_bundle_hysteresis", None),
    ("cable_y_junction", None),
    ("cloth_bending", None),
    ("cloth_hanging", None),
    ("cloth_twist", None),
    ("cloth_style3d", "git"),
    ("cable_cross_slide_table", "mujoco_warp"),
    ("franka_cable_ik_pick_place", "mujoco_warp"),
    ("cloth_rigid_trampoline", "mujoco_warp"),
]


def _load_newton_example(name: str):
    path = EXAMPLES_DIR / "newton" / f"{name}.py"
    for folder in (path.parent, EXAMPLES_DIR):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(f"_bench_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.benchmark_scene
@pytest.mark.parametrize("name, extra", NEWTON_SCENES, ids=[n for n, _ in NEWTON_SCENES])
def test_bench_newton_scene(benchmark, needs_cuda, name: str, extra: str | None):
    """Measure steady-state ``ms/frame`` over frames 2..6 (excluding frame 1 compile and graph capture)."""
    if extra is not None:
        pytest.importorskip(extra)
    module = _load_newton_example(name)
    num_frames = 20
    base_args = newton.examples.create_parser().parse_args(["--viewer", "null", "--num-frames", str(num_frames)])
    args = SimpleNamespace(**{**vars(base_args), "test": False})
    viewer = newton.viewer.ViewerNull(num_frames=num_frames)
    try:
        example = module.Example(viewer, args)
    except Exception as exc:
        if name == "cloth_style3d":
            pytest.skip(f"cloth_style3d asset unavailable: {exc}")
        raise

    # Frame 1: exclude NVRTC compile, CUDA graph capture, and first-step cuDSS analysis.
    example.step()
    wp.synchronize()
    frames_per_round = 3

    def _step_frames():
        for _ in range(frames_per_round):
            example.step()
        wp.synchronize()

    benchmark.pedantic(_step_frames, rounds=4, iterations=1, warmup_rounds=1)
    if benchmark.stats:
        ms_per_frame = (benchmark.stats["median"] * 1.0e3) / frames_per_round
        substeps = getattr(example, "substeps", 1)
        benchmark.extra_info["ms_per_frame"] = ms_per_frame
        benchmark.extra_info["ms_per_substep"] = ms_per_frame / max(substeps, 1)
