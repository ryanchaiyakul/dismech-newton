"""Gate 3: a stiff rod on a pad with an off-centre point load P at s = S: it tilts as the foundation's force and moment
balance says (delta(s) = delta0 + theta s, Winkler's line load kh (4/3) sqrt(2R) delta^(3/2) integrated exactly),
and the nodal quadrature converges with the node count."""
import sys

import newton
import numpy as np
import warp as wp
from scipy.optimize import fsolve

sys.path[:0] = ["tests", "examples", "exp/pads"]
from press import H, KH, R, TOP, pad  # noqa: E402

from dismech_newton import SolverDiSMech, flatten_state  # noqa: E402

L, G, P, S, EI = 0.4, 98.1, 2.0, 0.15, 50.0


def exact(w):
    k = KH * 4.0 / 3.0 * np.sqrt(2.0 * R)
    s = np.linspace(-L / 2, L / 2, 20001)

    def res(x):
        f = k * np.maximum(x[0] + x[1] * s, 0.0) ** 1.5
        return [np.trapezoid(f, s) / (w * L + P) - 1.0, np.trapezoid(s * f, s) / (P * S) - 1.0]
    return fsolve(res, [1.0e-4, 1.0e-4], xtol=1.0e-13)


def sim(n):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    pad(b)
    s = np.linspace(-L / 2, L / 2, n)
    pts = np.column_stack([s, np.zeros(n), np.full(n, TOP + H + R + 1.0e-4)])
    le = L / (n - 1)
    SolverDiSMech.add_rod(b, rod=newton.Rod(pts, radius=R), stretch_stiffness=1.0e7, bend_stiffness=EI / le,
                          twist_stiffness=EI / le, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5))
    model = b.finalize()
    solver = SolverDiSMech(model, iterations=30)
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    j = int(np.argmin(np.abs(s - S)))
    f = np.zeros((model.particle_count, 3), np.float32)
    f[j, 2] = -P
    for _ in range(720):
        s0.particle_f.assign(f)
        solver.step(s0, s1, None, None, 1.0 / 240.0)
        s0, s1 = s1, s0
    x = s0.particle_q.numpy()
    w = model.particle_mass.numpy().sum() * G / L
    delta = TOP + H + R - x[:, 2]
    th, d0 = np.polyfit(x[:, 0], delta, 1)
    return d0, th, s[j], w, np.abs(delta - (d0 + th * x[:, 0])).max()


if __name__ == "__main__":
    for n in (int(a) for a in sys.argv[1:]) if sys.argv[1:] else (6, 11, 21, 41, 81):
        d0, th, sj, w, bend = sim(n)
        S = sj
        e0, et = exact(w)
        print(f"n {n:3d}  delta0 {d0 * 1e6:8.3f} / {e0 * 1e6:8.3f} um  theta {th * 1e3:8.5f} / {et * 1e3:8.5f} mrad  "
              f"err {abs(d0 / e0 - 1):.1e} {abs(th / et - 1):.1e}  bend {bend * 1e6:.3f} um", flush=True)
