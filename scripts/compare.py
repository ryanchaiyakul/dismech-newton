"""Compare the DER ports with Newton's own VBD cable examples: speed and accuracy.

Scenes run Newton's ``Example`` unmodified next to ours, for the same simulated time, with no viewer.
Accuracy is measured identically on both from the capsules: the largest joint gap (relative to the
segment length; ours is the stretch), ground and self penetration (relative to the diameter). A
clamped cantilever (``examples/cantilever.py``) is checked against Euler-Bernoulli beam theory
(sag and first-mode frequency), and its numerical damping is measured.

    uv run scripts/compare.py
    uv run scripts/compare.py --scenes twist pile --frames 120
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))  # the example scenes

import cable_pile
import cable_plectoneme
import cable_twist
import cantilever as cantilever_example
import newton
import newton.viewer
import numpy as np
import warp as wp
from common import metrics
from newton.examples.cable import (
    example_cable_pile,
    example_cable_plectoneme,
    example_cable_twist,
)


_scratch = wp.zeros(1 << 20, dtype=float)


@wp.kernel
def _spin_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = wp.sin(a[i] + 1.0)


# name: (Newton's example, ours, segments per rod, default frames)
SCENES = {
    "twist": (example_cable_twist.Example, cable_twist.Example, 64, 300),
    "plectoneme": (example_cable_plectoneme.Example, cable_plectoneme.Example, 80, 780),
    "pile": (example_cable_pile.Example, cable_pile.Example, 40, 300),
}


def run_scene(cls, segments: int, frames: int, every: int = 10) -> dict:
    ex = cls(newton.viewer.ViewerNull(num_frames=frames), None)
    rods = np.arange(ex.model.body_count).reshape(-1, segments).tolist()
    ex.step()  # compile, set up and capture
    wp.synchronize()
    t = time.perf_counter()
    while time.perf_counter() - t < 1.0:  # bring the GPU up to clock on a throwaway kernel
        wp.launch(_spin_kernel, dim=1 << 20, inputs=[_scratch])
    wp.synchronize()
    worst, wall = {}, 0.0
    for f in range(frames):
        t = time.perf_counter()
        ex.step()
        wp.synchronize()
        wall += time.perf_counter() - t
        if f % every == every - 1:
            if not np.isfinite(ex.state_0.body_q.numpy()).all():
                return {"ms/frame": 1e3 * wall / (f + 1), "realtime": (f + 1) / 60.0 / wall, "finite": f"diverged at frame {f + 1}", **worst}
            for k, v in metrics(ex.model, ex.state_0, rods).items():
                worst[k] = max(worst.get(k, 0.0), v)
    return {"ms/frame": 1e3 * wall / frames, "realtime": frames / 60.0 / wall, "finite": "yes", **worst}


# -- cantilever (examples/cantilever.py) --------------------------------------------------


def cantilever(frames: int = 120, **options) -> dict[str, dict[str, float]]:
    """Tip sag, frequency and numerical damping of both cantilevers against Euler-Bernoulli."""
    ex = cantilever_example.Example(newton.viewer.ViewerNull(num_frames=frames), None, **options)
    for _ in range(frames):
        ex.step()
    return ex.summary()


def table(rows: dict[str, dict], title: str) -> None:
    keys = list(next(iter(rows.values())))
    print(f"\n### {title}\n")
    print("| | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for name, row in rows.items():
        cells = [f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v) for v in row.values()]
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="*", default=[*SCENES, "cantilever"])
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()
    wp.config.quiet = True
    for name in args.scenes:
        if name == "cantilever":
            rows = cantilever()
            table(rows, "cantilever (beam = Euler-Bernoulli, clamped midway between the fixed nodes)")
            continue
        newton_cls, ours_cls, segments, frames = SCENES[name]
        frames = args.frames or frames
        rows = {"Newton VBD": run_scene(newton_cls, segments, frames), "DER ADMM": run_scene(ours_cls, segments, frames)}
        table(rows, f"{name} ({frames} frames)")
