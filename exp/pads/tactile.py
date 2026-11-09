"""Tactile readout preview: a PaXini DP-S2015-Elite fingertip (its STEP's outer surface, a static hydroelastic pad, kh
estimated: E ~1 MPa over ~2 mm) pressed by a stiff rod lying along its crest with a load P, then pulled along it
with T < mu P. Readout as the sensor's: each pad point's force on the sensor to its nearest taxel, in the taxel's
frame (rows: shear x, y, normal), 0.1 N per LSB (Fx, Fy int8, Fz uint8). Checks: sum Fz = P, sum shear = T,
centre of pressure under the load. Data (vendor's, local): $PAXINI (S2015.json, S2015_mesh.npz)."""
import json
import os
import sys

import newton
import numpy as np

sys.path[:0] = ["examples", "tests"]
from utils.common import pads  # noqa: E402

from dismech_newton import SolverDiSMech, flatten_state  # noqa: E402
from dismech_newton.contact import SLIP_SPEED  # noqa: E402

D = os.environ.get("PAXINI", "../../../scratch/paxini")
KH, R, MU, DT = float(os.environ.get("KH", 5.0e8)), 0.003, 0.5, 1.0 / 600.0
P, T = float(os.environ.get("P", 5.0)), float(os.environ.get("T", 1.0))
tax = json.load(open(f"{D}/S2015.json"))
TP, TR = np.array(tax["positions"]) * 1.0e-3, np.array(tax["frames"])
mesh = np.load(f"{D}/S2015_mesh.npz")

builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
top = mesh["vertices"][np.abs(mesh["vertices"][:, 0]) < 0.3][:, 2].max() * 1.0e-3  # the crest
if os.environ.get("BOX"):  # debug: a box with the crest's top instead of the mesh
    import warp as wp
    builder.add_shape_box(-1, xform=wp.transform((0.0, 0.009, top - 0.005), wp.quat_identity()), hx=0.0075, hy=0.011,
                          hz=0.005, cfg=newton.ModelBuilder.ShapeConfig(mu=MU))
else:
    builder.add_shape_mesh(-1, mesh=newton.Mesh(mesh["vertices"] * 1.0e-3, mesh["faces"].ravel()),
                           cfg=newton.ModelBuilder.ShapeConfig(mu=MU))
from rollouts import cloth, grid  # noqa: E402

v, f = grid(31, 31, 0.03, 0.03)  # a 30 mm sheet over the fingertip, its centre over the crest's middle
v += [-0.015, -0.006, top + 0.001 + 2.0e-4]
cloth(builder, v, f, density=0.2, tri_ke=1.0e3, tri_ka=1.0e3, edge_ke=1.0e-4, particle_radius=0.001, damping=1.0e-2)
n = len(v)
ys = v[:, 1]
model = builder.finalize()
model.particle_mu = MU
pads(model, [-1], KH)
solver = SolverDiSMech(model, iterations=30)
s0, s1 = model.state(), model.state()
flatten_state(s0)
flatten_state(s1)


def run(steps, fz, fy):
    global s0, s1
    f = np.tile([0.0, fy / n, -fz / n], (n, 1)).astype(np.float32)
    for _ in range(steps):
        s0.particle_f.assign(f)
        solver.step(s0, s1, None, None, DT)
        s0, s1 = s1, s0


def readout():
    """Per taxel the force on the sensor in its frame [N], from the pad points' forces (normal: the pad's
    pressure; tangential: the lagged friction), at the end of the last step."""
    s = solver.contact.s
    act, pad = s.active.numpy().astype(bool), s.pad.numpy()
    i = np.flatnonzero(act & (pad[:, 0] > 0.0))
    x = s0.particle_q.numpy()
    pair, w = s.pair.numpy()[i], s.w.numpy()[i]
    pt = np.einsum("pk,pkc->pc", w, x[np.maximum(pair, 0)] * (pair >= 0)[..., None])
    nrm, anchor, th = s.normal.numpy()[i], s.anchor.numpy()[i], s.thickness.numpy()[i]
    c = pt - anchor
    d = np.einsum("pc,pc->p", nrm, c)
    delta = np.maximum(th + pad[i, 2] - d, 0.0)
    fn = pad[i, 1] * pad[i, 0] * delta ** (pad[i, 1] - 1.0)  # the pressure's force on the rod, along n
    slip = c - s.c0.numpy()[i]
    slip -= np.einsum("pc,pc->p", nrm, slip)[:, None] * nrm
    y = np.linalg.norm(slip, axis=1)
    eps = SLIP_SPEED * DT
    f1 = np.where(y >= eps, 1.0 / np.maximum(y, 1e-30), 2.0 / eps - y / eps**2)
    on_rod = fn[:, None] * nrm - (s.mu.numpy()[i] * s.rho.numpy()[i] * f1)[:, None] * slip
    on_sensor = -on_rod
    near = np.argmin(np.linalg.norm(anchor[:, None] - TP[None], axis=2), axis=1)
    F = np.zeros((len(TP), 3))
    np.add.at(F, near, on_sensor)
    local = np.einsum("tij,tj->ti", TR, F)
    local[:, 2] *= -1.0  # pressing in reads positive Fz
    raw = np.column_stack([np.clip(np.round(local[:, :2] / 0.1), -128, 127),
                           np.clip(np.round(local[:, 2] / 0.1), 0, 255)]).astype(int)
    return F, local, raw, anchor, on_sensor


def report(tag, fz, fy):
    F, local, raw, anchor, on = readout()
    cop = (anchor * -on[:, 2:3]).sum(0) / max(-on[:, 2].sum(), 1e-30)
    print(f"{tag}: sum F on sensor {F.sum(0).round(4)} N (applied: 0, {-fy}, {-fz})  centre of pressure x "
          f"{cop[0] * 1e3:.3f} y {cop[1] * 1e3:.3f} mm (load at y {ys.mean() * 1e3:.1f})  taxels loaded "
          f"{(raw[:, 2] > 0).sum()}  max Fz {local[:, 2].max():.3f} N  sum raw Fz {raw[:, 2].sum() * 0.1:.1f} N",
          flush=True)
    return local, raw


if os.environ.get("DBG"):
    for k in range(5):
        run(1, P, 0.0)
        print("step", k, "rod z [mm]", (s0.particle_q.numpy()[[0, n // 2, -1], 2] * 1e3).round(3), "active",
              int(solver.contact.s.active.numpy().sum()), flush=True)
run(600, P, 0.0)
press = report("press", P, 0.0)
if os.environ.get("DBG"):
    for k in range(6):
        run(20 if k < 3 else 100, P, T)
        F = readout()[0]
        print("pull t", k, "mean x [mm]", (s0.particle_q.numpy().mean(0) * 1e3).round(3), "mean v", s0.particle_qd.numpy().mean(0).round(4),
              "sum F", F.sum(0).round(3), "counts", solver.counts.numpy(), flush=True)
run(120, P, T)  # 0.2 s: held (friction's smoothing below SLIP_SPEED lets a held contact creep ~4 mm/s)
pull = report("pull ", P, T)
np.savez(os.environ.get("OUT", "exp/pads/logs/tactile.npz"), taxels=TP, press=press[0], pull=pull[0],
         press_raw=press[1], pull_raw=pull[1], rod=s0.particle_q.numpy())
if os.environ.get("DBG"):
    s = solver.contact.s
    print("rod z [mm]", (s0.particle_q.numpy()[:, 2] * 1e3).round(3), "top", top * 1e3)
    print("active", int(s.active.numpy().sum()), "pad", int(((s.pad.numpy()[:, 0] > 0) & (s.active.numpy() > 0)).sum()),
          "shape flags", model.shape_flags.numpy(), "kh", model.shape_material_kh.numpy())
if os.environ.get("DBG"):
    print("max |v| [m/s]", float(np.abs(s0.particle_qd.numpy()).max()), "mean v", s0.particle_qd.numpy().mean(0).round(5))
    s = solver.contact.s
    act, pad = s.active.numpy().astype(bool), s.pad.numpy()
    i = np.flatnonzero(act & (pad[:, 0] > 0.0))
    x = s0.particle_q.numpy()
    pair, w = s.pair.numpy()[i], s.w.numpy()[i]
    pt = np.einsum("pk,pkc->pc", w, x[np.maximum(pair, 0)] * (pair >= 0)[..., None])
    nrm, anchor, th = s.normal.numpy()[i], s.anchor.numpy()[i], s.thickness.numpy()[i]
    d = np.einsum("pc,pc->p", nrm, pt - anchor)
    delta = th + pad[i, 2] - d
    fn = pad[i, 1] * pad[i, 0] * np.maximum(delta, 0) ** (pad[i, 1] - 1.0)
    print("pad pairs", len(i), "loaded", int((delta > 0).sum()), "sum normal on sheet", (fn[:, None] * nrm).sum(0).round(4),
          "delta max [um]", float(delta.max() * 1e6), "rho sum", float(s.rho.numpy()[i].sum()), "w sums", np.unique(w.sum(1).round(3)))
