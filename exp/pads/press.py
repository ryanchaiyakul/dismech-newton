"""Gates 1-2: a rod (radius R) and a sheet resting on a flat hydroelastic pad (box, margin h = the pad, kh = E / h)
under a load sweep (gravity scaled). Winkler: a cylinder's line load w = kh (4/3) sqrt(2R) delta^(3/2); a sheet's
pressure p = kh delta. Prints the settled depth delta vs exact."""
import sys

import newton
import numpy as np
import warp as wp

sys.path[:0] = ["tests", "examples"]
from rollouts import cloth, grid  # noqa: E402

from dismech_newton import SolverDiSMech, flatten_state  # noqa: E402

R, H, KH, TOP = 0.005, 0.003, 3.0e7, 0.05  # rod radius, pad thickness, kh [N/m^3], the core's top


def pad(builder):
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, margin=H, is_hydroelastic=True, kh=KH)
    builder.add_shape_box(-1, xform=wp.transform((0.0, 0.0, TOP / 2), wp.quat_identity()), hx=0.3, hy=0.3,
                          hz=TOP / 2, cfg=cfg)


def settle(model, steps=480, dt=1.0 / 240.0):
    solver = SolverDiSMech(model, iterations=20)
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    for _ in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
    return s0.particle_q.numpy()


def rod(g):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -g))
    pad(b)
    pts = np.column_stack([np.linspace(-0.2, 0.2, 21), np.zeros(21), np.full(21, TOP + H + R + 1.0e-4)])
    SolverDiSMech.add_rod(b, rod=newton.Rod(pts, radius=R), stretch_stiffness=1.0e5, bend_stiffness=1.0,
                          twist_stiffness=1.0, collide=True,
                          cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5))
    model = b.finalize()
    x = settle(model)
    w = model.particle_mass.numpy().sum() * g / 0.4  # the line load (end nodes' half edges: 0.4 m of rod)
    exact = (3.0 * w / (4.0 * KH * np.sqrt(2.0 * R))) ** (2.0 / 3.0)
    delta = TOP + H + R - x[:, 2]
    return delta.mean(), delta.std(), exact


def sheet(g, rho=0.5, r=0.002):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -g))
    pad(b)
    v, f = grid(11, 11, 0.2, 0.2)
    v += [-0.1, -0.1, TOP + H + r + 1.0e-4]
    cloth(b, v, f, density=rho, tri_ke=1.0e3, tri_ka=1.0e3, edge_ke=1.0e-3, particle_radius=r, damping=1.0e-2)
    model = b.finalize()
    model.particle_mu = 0.5
    x = settle(model)
    exact = model.particle_mass.numpy().sum() * g / 0.04 / KH
    delta = TOP + H + r - x[:, 2]
    return delta.mean(), delta.std(), exact


if __name__ == "__main__":
    for name, f in (("rod", rod), ("sheet", sheet)):
        for g in (9.81, 98.1, 981.0, 3000.0):
            m, s, e = f(g)
            print(f"{name} g {g:7.1f}  delta {m * 1e6:9.3f} +- {s * 1e6:.3f} um  exact {e * 1e6:9.3f} um  "
                  f"ratio {m / e:.4f}", flush=True)
