"""Gate 4: a rod thrown flat at a pad (no gravity) at v_i rebounds at v_o: Hunt-Crossley's F = k delta^n (1 + d v)
gives d v_o + ln(1 - d v_o) = -(d v_i - ln(1 + d v_i)) (m v dv / (1 + d v) = -k delta^n d delta, over approach and
rebound), whatever k and n. Sweep d [s/m] (DM_PAD_HC on a Newton hydroelastic pad)."""
import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path[:0] = ["tests", "examples", "exp/pads"]

VI, DT, STEPS = 0.5, 5.0e-5, 1200


def exact(d):
    if d == 0.0:
        return VI
    lhs = d * VI - np.log1p(d * VI)
    return brentq(lambda v: d * v + np.log1p(-d * v) + lhs, 1.0e-9, (1.0 - 1.0e-12) / d)


def sim():
    import newton
    import press
    press.KH = 3.0e6
    from dismech_newton import SolverDiSMech, flatten_state
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    press.pad(b)
    n = 21
    pts = np.column_stack([np.linspace(-0.2, 0.2, n), np.zeros(n), np.full(n, press.TOP + press.H + press.R + 1.0e-4)])
    SolverDiSMech.add_rod(b, rod=newton.Rod(pts, radius=press.R), stretch_stiffness=1.0e5, bend_stiffness=1.0,
                          twist_stiffness=1.0, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.0))
    model = b.finalize()
    solver = SolverDiSMech(model, iterations=30)
    s0, s1 = model.state(), model.state()
    qd = np.zeros((n, 3), np.float32)
    qd[:, 2] = -VI
    s0.particle_qd.assign(qd)
    flatten_state(s0)
    flatten_state(s1)
    vmax = 0.0
    for _ in range(STEPS):
        solver.step(s0, s1, None, None, DT)
        s0, s1 = s1, s0
        vmax = max(vmax, float(s0.particle_qd.numpy()[:, 2].mean()))
    return vmax


if __name__ == "__main__":
    d = float(os.environ.get("DM_PAD_HC", 0.0))
    v = sim()
    print(f"d {d:5.2f} s/m  rebound {v:.4f} / {exact(d):.4f} m/s  e {v / VI:.4f} / {exact(d) / VI:.4f}", flush=True)
