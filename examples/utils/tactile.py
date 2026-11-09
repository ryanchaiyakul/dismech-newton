"""PaXini PX-6AX GEN3 tactile fingertips on the MIDAS hand: the sensors' outer surfaces as the fingertips' contact
shapes (hydroelastic pads in our contact, :func:`utils.common.pads`), the MJCF's fingertip capsules moved behind them.

The vendor's data stays local (the repo is public): ``PAXINI`` (default ``scratch/paxini``) holds per sensor its
taxel map (``S2015.json``, ``M2826.json``: positions [mm], frames), its meshed STEP (``*_mesh.npz``, mm) and
``mount/mount.json`` (each link's sensor and its pose, fitted to the mounting holes) and ``mount/dorsal.json`` (the
capsule left on the finger's back)."""

import json
import os
from pathlib import Path

import numpy as np
import warp as wp

from dismech_newton.collide import Bodies, _shape_pose
from dismech_newton.contact import Arm, Slots, contact_force, pair_point, shape_of

PAXINI = Path(os.environ.get("PAXINI", Path(__file__).resolve().parents[2] / "scratch" / "paxini"))
CAPSULES = {"index_dip_link": "index_fingertip_contact", "middle_dip_link": "middle_fingertip_contact",
            "ring_dip_link": "ring_fingertip_contact", "thumb_dip": "thumb_dip_contact"}


def mount(builder, root: Path = PAXINI) -> dict[str, int]:
    """After ``add_mjcf`` of the MIDAS hand: each sensor's surface a contact shape on its link (massless, the
    fingertip capsule's friction), the capsule itself no longer colliding, a smaller one on the finger's back.
    Returns per link its sensor shape."""
    import newton  # noqa: PLC0415
    from newton import ShapeFlags  # noqa: PLC0415

    poses = json.loads((root / "mount" / "mount.json").read_text())
    backs = json.loads((root / "mount" / "dorsal.json").read_text())
    body_of = {label.split("/")[-1]: b for b, label in enumerate(builder.body_label)}
    shapes = {label.split("/")[-1]: s for s, label in enumerate(builder.shape_label)}
    collide = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)
    out = {}
    for link, capsule in CAPSULES.items():
        p, body, old = poses[link], body_of[link], shapes[capsule]
        builder.shape_flags[old] &= ~collide
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=builder.shape_material_mu[old])
        mesh = np.load(root / f"{p['sensor']}_mesh.npz")
        q = wp.quat_from_matrix(wp.mat33(*np.asarray(p["R"], dtype=np.float32).ravel()))
        out[link] = builder.add_shape_mesh(body, xform=wp.transform(wp.vec3(*p["t"]), q), cfg=cfg, label=f"{link}_paxini",
                                           mesh=newton.Mesh(mesh["vertices"] * 1.0e-3, mesh["faces"].ravel()))
        b = backs[link]
        w, x, y, z = b["quat"]  # MJCF (w, x, y, z); the capsule's axis its +Z, as MuJoCo's
        builder.add_shape_capsule(body, xform=wp.transform(wp.vec3(*b["pos"]), wp.quat(x, y, z, w)), radius=b["radius"],
                                  half_height=b["half"], cfg=cfg, label=f"{link}_back")
    return out


SPACING = 1.25  # the receptive radius in taxel spacings (the median nearest-neighbour distance)
CONE = wp.constant(0.7071)  # a taxel takes a contact whose normal is within 45 degrees of its own (HydroelasticTouch)
LSB = wp.constant(0.1)  # [N] per count: Fx, Fy int8, Fz uint8 (the manual's)


class Tactile:
    """The sensors' taxel readings from our contact (``SolverDiSMech``): per step (:meth:`step`, after the solver's)
    each pair on a sensor shape its force on the sensor (pad pressure, barrier core, damping, friction: the step's
    contact force) at its point, in the sensor's frame, shared among the taxels within the receptive radius whose
    normals lie within 45 degrees of the contact's (weights ``(1 - d^2/r^2)^2``, normalized: the taxels' sum is the
    sensor's contact force; none in reach: its nearest). A coupler's passes each overwrite the step's forces;
    :meth:`commit` (once per coupled step) adds them over its ``dt``. :meth:`sample` (the sensor's 83.3 Hz: a frame
    of 6 steps of 2 ms, one graph) averages them, turns each into its taxel's frame (shear x, y, normal; pressing in
    reads positive) and quantizes (``raw``, :data:`LSB`). Device only: graph-capturable.

    ``shapes``: per link its sensor shape (:func:`mount`); ``solver``: the soft solver (a coupler's entry)."""

    def __init__(self, model, solver, shapes: dict[str, int], root: Path = PAXINI, spacing: float = SPACING):
        poses = json.loads((root / "mount" / "mount.json").read_text())
        pos, frame, start, radius = [], [], [], []
        of_shape = np.full(model.shape_count, -1, dtype=np.int32)
        self.links = list(shapes)
        for k, (link, shape) in enumerate(shapes.items()):
            data = json.loads((root / f"{poses[link]['sensor']}.json").read_text())
            p = np.asarray(data["positions"], dtype=np.float32) * 1.0e-3
            d = np.linalg.norm(p[:, None] - p[None], axis=2)
            np.fill_diagonal(d, np.inf)
            radius.append(spacing * float(np.median(d.min(1))))
            start.append(len(pos))
            pos += list(p)
            frame += list(np.asarray(data["frames"], dtype=np.float32).reshape(-1, 3, 3))
            of_shape[shape] = k
        start.append(len(pos))
        dev = model.device
        self.solver, self.count, self.start = solver, len(pos), np.array(start)
        self.of_shape = wp.array(of_shape, dtype=wp.int32, device=dev)
        self.first = wp.array(start, dtype=wp.int32, device=dev)
        self.radius = wp.array(radius, dtype=float, device=dev)
        self.pos = wp.array(np.array(pos), dtype=wp.vec3, device=dev)
        self.frame = wp.array(np.array(frame), dtype=wp.mat33, device=dev)
        self.cur = wp.zeros(self.count, dtype=wp.vec3, device=dev)  # the step's forces, sensor frame [N]
        self.cur_total = wp.zeros(len(shapes), dtype=wp.vec3, device=dev)  # each sensor's, unshared (a check)
        self.acc = wp.zeros(self.count, dtype=wp.vec3, device=dev)  # their time integrals [N s]
        self.total_acc = wp.zeros(len(shapes), dtype=wp.vec3, device=dev)
        self.time = wp.zeros(1, dtype=float, device=dev)
        self.force = wp.zeros(self.count, dtype=wp.vec3, device=dev)  # the last sample: taxel frames [N]
        self.raw = wp.zeros(self.count, dtype=wp.vec3i, device=dev)  # and in counts
        self.total = wp.zeros(len(shapes), dtype=wp.vec3, device=dev)  # each sensor's contact force, its frame [N]

    def step(self, state_out) -> None:
        """The forces of the solver's step just taken (``state_out``: its output), replacing the last pass's."""
        sv, c = self.solver, self.solver.contact
        self.cur.zero_()
        self.cur_total.zero_()
        if c.count:
            q = sv._qw if sv._qw is not None else state_out.dismech.q
            src = c.source
            wp.launch(_taxel_kernel, dim=c.s.active.shape[0],
                      inputs=[q, c.s, c.arm, c.slip, src.shape_key0, src.bodies, self.of_shape, self.first,
                              self.radius, self.pos, self.frame], outputs=[self.cur, self.cur_total],
                      device=self.acc.device)

    def hook(self) -> None:
        """Take every step of the solver (its last pass in a coupler's step stays)."""
        step = self.solver.step

        def stepped(state_in, state_out, *args, **kwargs):
            step(state_in, state_out, *args, **kwargs)
            self.step(state_out)

        self.solver.step = stepped

    def commit(self, dt: float) -> None:
        """Add the coupled step's forces (its last pass's) over ``dt``."""
        wp.launch(_commit_kernel, dim=self.count, inputs=[self.cur, self.cur_total, dt],
                  outputs=[self.acc, self.total_acc, self.time], device=self.acc.device)

    def sample(self) -> None:
        """The readings since the last sample into ``force``, ``raw`` and ``total``; restart the accumulation."""
        wp.launch(_sample_kernel, dim=self.count, inputs=[self.acc, self.total_acc, self.time, self.frame],
                  outputs=[self.force, self.raw, self.total], device=self.acc.device)
        self.time.zero_()  # (after: every taxel's thread reads it)

    def readings(self, link: str) -> tuple[np.ndarray, np.ndarray]:
        """``link``'s last sample (host): taxel forces [N] (shear x, y, normal) and counts."""
        k = self.links.index(link)
        a, b = self.start[k], self.start[k + 1]
        return self.force.numpy()[a:b], self.raw.numpy()[a:b]


@wp.func
def _reach(x: wp.vec3, n: wp.vec3, p: wp.vec3, frame: wp.mat33, r: float) -> float:
    """A taxel's weight for a contact at ``x`` (normal ``n``): ``(1 - d^2/r^2)^2`` within ``r`` and the cone."""
    d2 = wp.length_sq(x - p)
    if d2 >= r * r or wp.dot(wp.vec3(frame[2, 0], frame[2, 1], frame[2, 2]), n) <= CONE:
        return 0.0
    u = 1.0 - d2 / (r * r)
    return u * u


@wp.kernel
def _taxel_kernel(q: wp.array[float], s: Slots, arm: Arm, eps: float, key0: int, b: Bodies,
                  of_shape: wp.array[wp.int32], first: wp.array[wp.int32], radius: wp.array[float],
                  pos: wp.array[wp.vec3], frame: wp.array[wp.mat33], acc: wp.array[wp.vec3],
                  total: wp.array[wp.vec3]):
    i = wp.tid()
    if s.active[i] == 0 or s.key[i] < key0:
        return
    shape = shape_of(s.key[i], key0)
    k = of_shape[shape]
    if k < 0:
        return
    g, h, fr = contact_force(pair_point(q, s, i, arm, s.anchor[i]), s.normal[i], s.c0[i], s.thickness[i],
                             s.stiffness[i], s.lam[i], s.mu[i] * s.rho[i], eps, s.pad[i], s.damp[i])
    if wp.length(g) == 0.0:
        return
    inv = wp.transform_inverse(_shape_pose(b, shape, b.body_q))  # the sensor's frame at the step's start
    x = wp.transform_point(inv, s.anchor[i])
    f = wp.transform_vector(inv, g)  # the force on the sensor
    n = wp.transform_vector(inv, s.normal[i])  # outward from the sensor
    wp.atomic_add(total, k, f)
    wsum = float(0.0)
    near, dmin = int(first[k]), float(1.0e30)
    for t in range(first[k], first[k + 1]):
        wsum += _reach(x, n, pos[t], frame[t], radius[k])
        d2 = wp.length_sq(x - pos[t])
        if d2 < dmin:
            near, dmin = t, d2
    if wsum == 0.0:
        wp.atomic_add(acc, near, f)
        return
    for t in range(first[k], first[k + 1]):
        u = _reach(x, n, pos[t], frame[t], radius[k])
        if u > 0.0:
            wp.atomic_add(acc, t, (u / wsum) * f)


@wp.kernel
def _commit_kernel(cur: wp.array[wp.vec3], cur_total: wp.array[wp.vec3], dt: float, acc: wp.array[wp.vec3],
                   total: wp.array[wp.vec3], time: wp.array[float]):
    t = wp.tid()  # (a thread per taxel; the first few take the sensors' totals and the time)
    acc[t] = acc[t] + dt * cur[t]
    if t < total.shape[0]:
        total[t] = total[t] + dt * cur_total[t]
    if t == 0:
        time[0] += dt


@wp.kernel
def _sample_kernel(acc: wp.array[wp.vec3], total_acc: wp.array[wp.vec3], time: wp.array[float],
                   frame: wp.array[wp.mat33], force: wp.array[wp.vec3], raw: wp.array[wp.vec3i],
                   total: wp.array[wp.vec3]):
    t = wp.tid()
    span = wp.max(time[0], 1.0e-30)
    f = frame[t] * (acc[t] / span)
    f = wp.vec3(f[0], f[1], -f[2])  # pressing in (the force on the sensor against its normal) reads positive
    force[t] = f
    raw[t] = wp.vec3i(wp.clamp(int(wp.round(f[0] / LSB)), -128, 127), wp.clamp(int(wp.round(f[1] / LSB)), -128, 127),
                      wp.clamp(int(wp.round(f[2] / LSB)), 0, 255))
    acc[t] = wp.vec3(0.0)
    if t < total.shape[0]:
        total[t] = total_acc[t] / span
        total_acc[t] = wp.vec3(0.0)
