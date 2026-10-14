"""Reference runs of the examples, a regression check: record them on one branch, compare on another.

    uv run examples/utils/reference.py record [names...]   # into .cache/reference/<name>.npz
    uv run examples/utils/reference.py compare [names...]  # against them (the run: .cache/reference/current/)

Each example runs headless with ``--test`` in its own process; every ``EVERY`` frames (and at the end) it keeps
``state_0``'s node positions and twists (``fit_buckling``: the fitted ratios), plus the test's outcome and the
step time without the first frame (compilation, the graph capture).
"""

import argparse
import json
import runpy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".cache" / "reference"
EXAMPLES = ["cable_twist", "cable_pile", "cantilever", "overhand_knot", "plectoneme", "flagella", "fit_buckling",
            "fit_capture"]
EXTRA_ARGS = {name: ["--fresh"] for name in ("fit_buckling", "fit_capture")}  # re-run the fit, not the cached replay
EVERY = 10


def _snapshot(example) -> dict[str, np.ndarray]:
    out = {}
    state = getattr(example, "state_0", None)
    if state is not None:
        out["x"] = state.particle_q.numpy()
        out["theta"] = state.dismech.edge_q.numpy()
    if hasattr(example, "ratios"):
        out["ratios"] = np.asarray(example.ratios)
    return out


def _run_one(name: str, path: Path) -> None:
    """In this process: run ``name`` and write ``path``."""
    import newton.examples
    import warp as wp

    frames, result = [], {"error": "", "step_seconds": 0.0, "frames": 0}
    original = newton.examples.run

    def recording_run(example, args):
        step = example.step

        def recorded_step():
            wp.synchronize()
            t0 = time.perf_counter()
            step()
            wp.synchronize()
            result["frames"] += 1
            if result["frames"] > 1:
                result["step_seconds"] += time.perf_counter() - t0
            if result["frames"] % EVERY == 0:
                frames.append(_snapshot(example))

        example.step = recorded_step
        try:
            original(example, args)
        except Exception as e:  # the test's verdict is part of the reference
            result["error"] = f"{type(e).__name__}: {e}"
        frames.append(_snapshot(example))

    newton.examples.run = recording_run
    script = ROOT / "examples" / f"{name}.py"
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), "--viewer", "null", "--test", *EXTRA_ARGS.get(name, [])]
    runpy.run_path(str(script), run_name="__main__")

    arrays = {f"{k}_{i}": v for i, snap in enumerate(frames) for k, v in snap.items()}
    np.savez_compressed(path, result=json.dumps(result), count=len(frames), **arrays)


def _load(path: Path) -> tuple[dict, list[dict]]:
    with np.load(path) as f:
        result = json.loads(str(f["result"]))
        frames = [{k: f[f"{k}_{i}"] for k in ("x", "theta", "ratios") if f"{k}_{i}" in f} for i in range(int(f["count"]))]
    return result, frames


def _record(name: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, __file__, "_one", name, str(path)], check=True)
    return _load(path)[0]


def _status(r: dict) -> str:
    return f"{'FAIL ' + r['error'][:60] if r['error'] else 'pass'}, {1e3 * r['step_seconds'] / max(r['frames'] - 1, 1):.1f} ms/frame"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["record", "compare", "_one"])
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    if args.command == "_one":
        _run_one(args.names[0], Path(args.names[1]))
        return
    for name in args.names or EXAMPLES:
        ref = OUT / f"{name}.npz"
        if args.command == "record":
            print(f"{name}: {_status(_record(name, ref))}", flush=True)
            continue
        new = OUT / "current" / f"{name}.npz"
        r_new, f_new = _record(name, new), _load(new)[1]
        r_ref, f_ref = _load(ref)
        print(f"{name}: reference {_status(r_ref)}; now {_status(r_new)}")
        for key in ("x", "theta", "ratios"):
            pairs = [(a[key], b[key]) for a, b in zip(f_ref, f_new) if key in a and key in b]
            if pairs and pairs[-1][0].shape != pairs[-1][1].shape:  # a fit's history: compare the common rows
                print(f"    {key}: {len(pairs[-1][0])} rows, now {len(pairs[-1][1])}")
                pairs = [(a[: min(len(a), len(b))], b[: min(len(a), len(b))]) for a, b in pairs]
            if pairs:
                dev = [np.abs(a - b).max() for a, b in pairs]
                size = max(np.ptp(a, axis=0).max() if a.ndim > 1 else np.abs(a).max() for a, _ in pairs)
                print(f"    {key}: max deviation {max(dev):.3g} (final {dev[-1]:.3g}; scale {size:.3g})")


if __name__ == "__main__":
    main()
