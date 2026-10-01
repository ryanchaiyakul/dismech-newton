"""Flagella bundling (``examples/flagella.py``) timed against Table 1 of the IMC paper (arXiv:2205.10309).

Each run simulates ``--time`` seconds at the paper's ``dt = 1 ms``, with no viewer. The paper's
total runtime is for 250 s; ours is measured for ``--time`` and scaled to 250 s. Every
``--sample`` seconds we record the closest approach between flagella and within one (diameters),
the largest edge strain, the mean tip-to-tip distance (the reference code's bundling measure) and
how far the tip has turned about its clamp axis, for fidelity between the solvers.

    uv run scripts/flagella_bench.py --flagella 2 3 5 10 --time 250
    uv run scripts/flagella_bench.py --flagella 2 --solvers der vbd:10 vbd:100 --time 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import numpy as np
import warp as wp
from flagella import PAPER, DERFlagella, VBDFlagella

# M: (AIPTS, ATPTS [ms], total run time [hr]) for 250 s; ATPTS is over steps with a contact.
PAPER_TABLE = {
    "IMC": {2: (3.00, 10.2, 0.57), 3: (3.01, 21.3, 1.19), 5: (3.02, 67.5, 3.34), 10: (3.12, 389.4, 22.77)},
    "IPC": {2: (4.00, 18.75, 1.04), 3: (4.00, 39.5, 2.17), 5: (4.01, 95.3, 4.68), 10: (4.02, 477.47, 27.88)},
}


def make(spec: str, count: int):
    if spec == "der":
        return DERFlagella(count)
    kind, _, rest = spec.partition(":")
    it, _, sub = rest.partition("x")
    return VBDFlagella(count, iterations=int(it or 10), substeps=int(sub or 1))


def run(spec: str, count: int, sim_time: float, sample: float) -> dict:
    sim = make(spec, count)
    steps = int(round(sim_time / sim.dt))
    every = max(1, int(round(sample / sim.dt)))
    axis = np.array([pts[0, :2] for pts in sim.points])
    for _ in range(2):  # set up and capture both graphs; their steps count as simulated time
        sim.step()
    wp.synchronize()

    wall, trace, turned, last = 0.0, [], np.zeros(count), None
    for k in range(2, steps):
        t0 = time.perf_counter()
        sim.step()
        wp.synchronize()
        wall += time.perf_counter() - t0
        if (k + 1) % every == 0 or k == steps - 1:
            x = sim.nodes()
            if not np.isfinite(x).all():
                trace.append({"t": sim.time, "diverged": True})
                break
            tip = np.arctan2(x[:, -1, 1] - axis[:, 1], x[:, -1, 0] - axis[:, 0])
            if last is not None:
                turned += np.angle(np.exp(1j * (tip - last)))  # sampled well below half a turn
            last = tip
            between, within = sim.gaps()
            trace.append({"t": round(sim.time, 6), "gap_between": between, "gap_within": within,
                          "stretch": sim.max_stretch(), "tip_spread": sim.tip_spread(),
                          "tip_turns": float(turned.mean() / (2 * np.pi))})
    timed = max(1, sim.steps - 2)
    return {
        "solver": sim.name, "M": count, "sim_time": sim.time, "wall_s": wall, "ms_per_step": 1e3 * wall / timed,
        "iterations_per_step": sim.total_iterations / sim.steps,
        "hours_per_250s": wall / max(sim.time - 2 * sim.dt, 1e-9) * PAPER["total_time"] / 3600.0,
        "min_gap_between": min(r.get("gap_between", np.inf) for r in trace),
        "min_gap_within": min(r.get("gap_within", np.inf) for r in trace),
        "max_stretch": max(r.get("stretch", 0.0) for r in trace),
        "trace": trace,
    }


def table(rows: list[dict]) -> None:
    print("\n| solver | M | it/step | ms/step | hours for 250 s | paper IMC: it, ms, hours | min gap between / within [diam] | max stretch |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        imc = PAPER_TABLE["IMC"].get(r["M"])
        paper = f"{imc[0]:.2f}, {imc[1]:.1f}, {imc[2]:.2f}" if imc else "-"
        print(f"| {r['solver']} | {r['M']} | {r['iterations_per_step']:.1f} | {r['ms_per_step']:.2f} | "
              f"{r['hours_per_250s']:.3f} | {paper} | {r['min_gap_between']:.2f} / {r['min_gap_within']:.2f} | "
              f"{r['max_stretch']:.1e} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flagella", type=int, nargs="*", default=[2, 3, 5, 10])
    parser.add_argument("--solvers", nargs="*", default=["der", "vbd:10"],
                        help="der, or vbd:<iterations>[x<substeps>]")
    parser.add_argument("--time", type=float, default=PAPER["total_time"], help="simulated seconds per run")
    parser.add_argument("--sample", type=float, default=0.1, help="seconds between fidelity samples")
    parser.add_argument("--out", type=Path, default=None, help="append one JSON line per run")
    args = parser.parse_args()
    wp.config.log_level = wp.LOG_WARNING if hasattr(wp, "LOG_WARNING") else None
    rows = []
    for count in args.flagella:
        for spec in args.solvers:
            row = run(spec, count, args.time, args.sample)
            rows.append(row)
            print(f"{row['solver']} M={count}: {row['ms_per_step']:.2f} ms/step, "
                  f"{row['iterations_per_step']:.1f} it/step, {row['hours_per_250s']:.3f} h / 250 s", flush=True)
            if args.out:
                with args.out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
    table(rows)
