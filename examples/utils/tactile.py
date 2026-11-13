"""PaXini PX-6AX GEN3 tactile fingertips on the MIDAS hand: the sensors' outer surfaces as the fingertips' contact
shapes (hydroelastic pads in our contact, :func:`utils.common.pads`), the MJCF's fingertip capsules moved behind them.

``PAXINI`` (default ``examples/utils/paxini``) holds per sensor its taxel map (``S2015.json``, ``M2826.json``:
positions [mm], frames), its meshed STEP (``*_mesh.npz``, mm) and ``mount/mount.json`` (each link's sensor and its pose, fitted to the mounting holes) and ``mount/dorsal.json`` (the
capsule left on the finger's back)."""

import json
import os
from pathlib import Path

import numpy as np
import warp as wp

from dismech_newton.adjoint import _arm_com_kernel, _detached, _lever_off_kernel, _scan_kernel, _weights_kernel
from dismech_newton.collide import Bodies, _shape_gap, _shape_pose
from dismech_newton.contact import (
    LIFT,
    Arm,
    Slots,
    arm_body,
    arm_flexibility,
    arm_move,
    barrier,
    body_flexibility,
    contact_force,
    contact_point,
    flexibility,
    pad_energy,
    pair_point,
    roll_point,
    shape_of,
    slot_roll,
)
from dismech_newton.coupling import _rotvec
from dismech_newton.state import dof_mass, suspended_tape

PAXINI = Path(os.environ.get("PAXINI", Path(__file__).resolve().parent / "paxini"))
CAPSULES = {"index_dip_link": "index_fingertip_contact", "middle_dip_link": "middle_fingertip_contact",
            "ring_dip_link": "ring_fingertip_contact", "thumb_dip": "thumb_dip_contact"}
COLORS = {"index_dip_link": "#3987e5", "middle_dip_link": "#38ad7a", "ring_dip_link": "#a83265", "thumb_dip": "#c98500"}
"""Each sensor's colour, in the viewer (:func:`mount`) and in plots."""


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
    out, meshes = {}, {}
    for link, capsule in CAPSULES.items():
        p, body, old = poses[link], body_of[link], shapes[capsule]
        builder.shape_flags[old] &= ~collide
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=builder.shape_material_mu[old])
        if p["sensor"] not in meshes:  # its SDF (our contact samples it): out past a thick rod's reach
            mesh = np.load(root / f"{p['sensor']}_mesh.npz")
            meshes[p["sensor"]] = newton.Mesh(mesh["vertices"] * 1.0e-3, mesh["faces"].ravel())
            meshes[p["sensor"]].build_sdf(narrow_band_range=SDF_BAND, target_voxel_size=SDF_VOXEL,
                                          margin=SDF_BAND[1])
        q = wp.quat_from_matrix(wp.mat33(*np.asarray(p["R"], dtype=np.float32).ravel()))
        rgb = tuple(int(COLORS[link][i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        out[link] = builder.add_shape_mesh(body, xform=wp.transform(wp.vec3(*p["t"]), q), cfg=cfg, label=f"{link}_paxini",
                                           mesh=meshes[p["sensor"]], color=rgb)
        b = backs[link]
        w, x, y, z = b["quat"]  # MJCF (w, x, y, z); the capsule's axis its +Z, as MuJoCo's
        builder.add_shape_capsule(body, xform=wp.transform(wp.vec3(*b["pos"]), wp.quat(x, y, z, w)), radius=b["radius"],
                                  half_height=b["half"], cfg=cfg, label=f"{link}_back")
    return out


SDF_BAND, SDF_VOXEL = (-0.005, 0.04), 0.5e-3  # [m]
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
        n_links = len(shapes)
        spw = model.shape_count // max(model.world_count, 1)
        multi = model.world_count > 1 and model.shape_count % model.world_count == 0 and all(
            s < spw for s in shapes.values()
        )
        self.num_worlds = int(model.world_count) if multi else 1
        for k, (link, shape) in enumerate(shapes.items()):
            data = json.loads((root / f"{poses[link]['sensor']}.json").read_text())
            p = np.asarray(data["positions"], dtype=np.float32) * 1.0e-3
            d = np.linalg.norm(p[:, None] - p[None], axis=2)
            np.fill_diagonal(d, np.inf)
            radius.append(spacing * float(np.median(d.min(1))))
            start.append(len(pos))
            pos += list(p)
            frame += list(np.asarray(data["frames"], dtype=np.float32).reshape(-1, 3, 3))
            if multi:
                for w in range(self.num_worlds):
                    of_shape[w * spw + shape] = w * n_links + k
            else:
                of_shape[shape] = k
        start.append(len(pos))
        dev = model.device
        self.solver = solver
        self.sensors_per_world = n_links
        self.count_per_world = len(pos)
        self.count = self.num_worlds * self.count_per_world
        self.start = np.array(start)
        self.of_shape = wp.array(of_shape, dtype=wp.int32, device=dev)
        self.first = wp.array(start, dtype=wp.int32, device=dev)
        self.radius = wp.array(radius, dtype=float, device=dev)
        self.pos = wp.array(np.array(pos), dtype=wp.vec3, device=dev)
        self.frame = wp.array(np.array(frame), dtype=wp.mat33, device=dev)
        total_sensors = self.num_worlds * n_links
        self.cur = wp.zeros(self.count, dtype=wp.vec3, device=dev, requires_grad=True)  # sensor frame [N]
        self.cur_total = wp.zeros(total_sensors, dtype=wp.vec3, device=dev, requires_grad=True)
        self.acc = wp.zeros(self.count, dtype=wp.vec3, device=dev, requires_grad=True)  # time integrals [N s]
        self.total_acc = wp.zeros(total_sensors, dtype=wp.vec3, device=dev, requires_grad=True)
        self.time = wp.zeros(1, dtype=float, device=dev)
        self.force = wp.zeros(self.count, dtype=wp.vec3, device=dev, requires_grad=True)  # taxel frames [N]
        self.raw = wp.zeros(self.count, dtype=wp.vec3i, device=dev)  # and in counts
        self.total = wp.zeros(total_sensors, dtype=wp.vec3, device=dev, requires_grad=True)
        self._acc_dt = 0.0
        self._dummy_v3 = wp.zeros(max(solver.contact.count, 1), dtype=wp.vec3, device=dev)
        self._dummy_f = wp.zeros(1, dtype=float, device=dev)
        self._slot_cap = 0

    def _slot_bufs(self, n_c: int, dev, grad: bool):
        if grad:
            with suspended_tape():
                return (
                    wp.full(n_c, -1, dtype=wp.int32, device=dev),
                    wp.empty(n_c, dtype=wp.int32, device=dev),
                    wp.zeros(n_c, dtype=wp.vec3, device=dev, requires_grad=True),
                    wp.zeros(n_c, dtype=wp.vec3, device=dev, requires_grad=True),
                    wp.zeros(n_c, dtype=wp.vec3, device=dev, requires_grad=True),
                    wp.zeros(n_c, dtype=float, device=dev, requires_grad=True),
                    wp.zeros(n_c, dtype=wp.int32, device=dev),
                )
        if self._slot_cap < n_c:
            self._slot_cap = n_c
            self._s_k = wp.empty(n_c, dtype=wp.int32, device=dev)
            self._s_off = wp.empty(n_c, dtype=wp.int32, device=dev)
            self._s_x = wp.empty(n_c, dtype=wp.vec3, device=dev)
            self._s_n = wp.empty(n_c, dtype=wp.vec3, device=dev)
            self._s_f = wp.empty(n_c, dtype=wp.vec3, device=dev)
            self._s_wsum = wp.empty(n_c, dtype=float, device=dev)
            self._s_near = wp.empty(n_c, dtype=wp.int32, device=dev)
        self._s_wsum.zero_()
        return self._s_k, self._s_off, self._s_x, self._s_n, self._s_f, self._s_wsum, self._s_near

    def _pack_diff_q(self, state, poses, body_com, pa, dev, *, include_arm: bool = True):
        sv = self.solver
        with suspended_tape():
            q_diff = wp.zeros(sv.num_dofs, dtype=float, device=dev, requires_grad=True)
        np_ = min(sv.model.particle_count, state.particle_q.shape[0])
        if np_ > 0:
            wp.launch(_tactile_node_q_kernel, dim=np_, inputs=[state.particle_q], outputs=[q_diff], device=dev)
        ns_d = state.dismech
        eq = getattr(ns_d, "edge_q", None)
        ne_ = eq.shape[0] if eq is not None else 0
        if ne_ > 0:
            wp.launch(_tactile_scalar_q_kernel, dim=ne_, inputs=[eq, 3 * np_], outputs=[q_diff], device=dev)
        hq = getattr(ns_d, "hinge_q", None)
        nh_ = hq.shape[0] if hq is not None else 0
        if nh_ > 0:
            wp.launch(_tactile_scalar_q_kernel, dim=nh_, inputs=[hq, 3 * np_ + ne_], outputs=[q_diff], device=dev)
        if include_arm and pa is not None:
            gb = getattr(pa, "_global_body", None)
            use_global = int(gb is not None and state.body_q.shape[0] != sv.model.body_count)
            wp.launch(_tactile_arm_q_kernel, dim=pa._body.shape[0],
                      inputs=[pa._body, gb if gb is not None else pa._body, use_global, pa._arm.base,
                              pa._was_valid, 0, state.body_q, 0, state.body_q, poses, body_com],
                      outputs=[q_diff], device=dev)
        return q_diff

    def step(self, state_out, state_in=None) -> None:
        """The forces of the solver's step just taken (``state_out``: its output), replacing the last pass's."""
        sv, c = self.solver, self.solver.contact
        dev = self.acc.device
        tape = wp._src.context.runtime.tape
        if tape is not None:
            self.cur = wp.zeros_like(self.cur, requires_grad=True)
            self.cur_total = wp.zeros_like(self.cur_total, requires_grad=True)
        else:
            self.cur.zero_()
            self.cur_total.zero_()
        if not c.count:
            pa = getattr(sv, "_proxy_arm", None)
            if tape is not None and pa is not None and state_out.body_q is not None:
                pa._tape_last = state_out.body_q
                pa._tape_id = tape
            return
        src = c.source
        shape_mu = sv.model.shape_material_mu if sv.model.shape_count else wp.zeros(1, dtype=float, device=dev)
        if tape is None:
            q = sv._qw if sv._qw is not None else state_out.dismech.q
            n_s = c.s.active.shape[0]
            s_k, s_off, s_x, s_n, s_f, s_ws, s_near = self._slot_bufs(n_s, dev, grad=False)
            wp.launch(_taxel_kernel, dim=n_s,
                      inputs=[q, q, q, 0, 0, c.s, c.s.w, c.arm, c.slip, src.shape_key0, src.bodies,
                              src.bodies.body_q, sv.fixed, self._dummy_f, self._dummy_f, c.arm.com,
                              self._dummy_v3, src._mu_shape, shape_mu, _detached(shape_mu),
                              float(sv.model.particle_mu), c.trib, c.aniso, self.of_shape, self.sensors_per_world],
                      outputs=[s_k, s_off, s_x, s_n, s_f, self.cur_total, self._dummy_v3], device=dev)
            wp.launch(_wsum_kernel, dim=n_s, inputs=[s_k, s_x, s_n, self.first, self.radius, self.pos, self.frame],
                      outputs=[s_ws, s_near], device=dev)
            wp.launch(_scatter_kernel, dim=n_s,
                      inputs=[s_k, s_off, self.count_per_world, s_x, s_n, s_f, s_ws, s_near,
                              self.first, self.radius, self.pos, self.frame],
                      outputs=[self.cur], device=dev)
            return
        with suspended_tape():
            s_snap = c.snapshot()
            if s_snap is None:
                return
            poses_snap = wp.clone(src.bodies.body_q)
            poses_diff = wp.clone(poses_snap, requires_grad=True)
            arm_snap = Arm()
            for name in Arm.vars:
                v = getattr(c.arm, name)
                setattr(arm_snap, name, wp.clone(v) if isinstance(v, wp.array) else v)
            pa = getattr(sv, "_proxy_arm", None)
            n_c = s_snap.active.shape[0]
            mass = wp.zeros(sv.num_state_dofs, dtype=float, device=dev, requires_grad=True)
            com_diff = wp.zeros(sv.model.body_count, dtype=wp.vec3, device=dev, requires_grad=True)
            t0 = wp.empty(n_c, dtype=float, device=dev)
            weights = wp.zeros(n_c, dtype=wp.vec4, device=dev, requires_grad=True)
            off = wp.zeros(n_c, dtype=wp.vec3, device=dev)
            anchors = wp.zeros(n_c, dtype=wp.vec3, device=dev)
            anchors_out = wp.zeros(n_c, dtype=wp.vec3, device=dev)
            dummy_tot = wp.zeros_like(self.cur_total)
        prev_q = None
        if pa is not None:
            prev_q = getattr(pa, "_tape_last", None) if getattr(pa, "_tape_id", None) is tape else None
            wp.launch(_tactile_poses_kernel, dim=pa._body.shape[0],
                      inputs=[pa._body, pa._was_valid, prev_q if prev_q is not None else state_in.body_q,
                              int(prev_q is not None), state_in.body_q if state_in is not None else poses_snap],
                      outputs=[poses_diff], device=dev)
            wp.launch(_arm_com_kernel, dim=pa._body.shape[0],
                      inputs=[pa._body, sv.model.body_com, poses_diff], outputs=[com_diff], device=dev)
        elif state_in is not None and state_in.body_q is not None and state_in.body_q.shape[0] == poses_snap.shape[0]:
            wp.copy(poses_diff, state_in.body_q)
        dof_mass(sv.model, mass)
        mass0 = _detached(mass)
        q_diff = self._pack_diff_q(state_out, poses_diff, src.bodies.body_com, pa, dev) if pa is not None else state_out.dismech.q
        if state_in is not None:
            q0 = self._pack_diff_q(state_in, poses_diff, src.bodies.body_com, pa, dev, include_arm=False) if pa is not None else state_in.dismech.q
            q0_snap = _detached(q0)
            has_q0 = 1
        else:
            q0, q0_snap, has_q0 = q_diff, q_diff, 0
        wp.launch(_scan_kernel, dim=n_c, inputs=[s_snap, src.bodies, poses_snap, src.shape_key0, q0_snap],
                  outputs=[t0], device=dev, record_tape=False)
        wp.launch(_weights_kernel, dim=n_c,
                  inputs=[s_snap, src.bodies, poses_diff, poses_snap, src.shape_key0, q0, q0_snap, t0],
                  outputs=[weights], device=dev)
        common_in = [q_diff, q0, q0_snap, has_q0, int(has_q0 != 0), s_snap, weights, arm_snap, c.slip,
                     src.shape_key0, src.bodies, poses_diff, sv.fixed, mass, mass0, com_diff, off,
                     src._mu_shape, shape_mu, _detached(shape_mu), float(sv.model.particle_mu),
                     c.trib, c.aniso, self.of_shape, self.sensors_per_world]
        if has_q0 != 0:
            with suspended_tape():
                d_k = wp.empty(n_c, dtype=wp.int32, device=dev)
                d_off = wp.empty(n_c, dtype=wp.int32, device=dev)
                d_x = wp.empty(n_c, dtype=wp.vec3, device=dev)
                d_n = wp.empty(n_c, dtype=wp.vec3, device=dev)
                d_f = wp.empty(n_c, dtype=wp.vec3, device=dev)
            wp.launch(_taxel_kernel, dim=n_c, inputs=common_in,
                      outputs=[d_k, d_off, d_x, d_n, d_f, dummy_tot, anchors], device=dev, record_tape=False)
            wp.launch(_lever_off_kernel, dim=n_c, inputs=[s_snap.anchor, anchors], outputs=[off],
                      device=dev, record_tape=False)
        s_k, s_off, s_x, s_n, s_f, s_ws, s_near = self._slot_bufs(n_c, dev, grad=True)
        wp.launch(_taxel_kernel, dim=n_c, inputs=common_in,
                  outputs=[s_k, s_off, s_x, s_n, s_f, self.cur_total, anchors_out], device=dev)
        wp.launch(_wsum_kernel, dim=n_c, inputs=[s_k, s_x, s_n, self.first, self.radius, self.pos, self.frame],
                  outputs=[s_ws, s_near], device=dev)
        wp.launch(_scatter_kernel, dim=n_c,
                  inputs=[s_k, s_off, self.count_per_world, s_x, s_n, s_f, s_ws, s_near,
                          self.first, self.radius, self.pos, self.frame],
                  outputs=[self.cur], device=dev)
        if pa is not None and state_out.body_q is not None:
            pa._tape_last = state_out.body_q
            pa._tape_id = tape

    def hook(self) -> None:
        """Take every step of the solver (its last pass in a coupler's step stays)."""
        step = self.solver.step

        def stepped(state_in, state_out, *args, **kwargs):
            step(state_in, state_out, *args, **kwargs)
            self.step(state_out, state_in)

        self.solver.step = stepped

    def commit(self, dt: float) -> None:
        """Add the coupled step's forces (its last pass's) over ``dt``."""
        self._acc_dt += float(dt)
        wp.launch(_commit_kernel, dim=self.count, inputs=[self.cur, self.cur_total, float(dt)],
                  outputs=[self.acc, self.total_acc, self.time], device=self.acc.device)

    def sample(self) -> None:
        """The readings since the last sample into ``force``, ``raw`` and ``total``; restart the accumulation."""
        dev = self.acc.device
        tape = wp._src.context.runtime.tape
        if tape is None:
            wp.launch(_sample_kernel, dim=self.count,
                      inputs=[self.acc, self.total_acc, self.time, 0.0, self.frame, self.count_per_world],
                      outputs=[self.force, self.total], device=dev)
            wp.launch(_quantize_kernel, dim=self.count, inputs=[self.force], outputs=[self.raw], device=dev)
            self.acc.zero_()
            self.total_acc.zero_()
            self.time.zero_()
            self._acc_dt = 0.0
            return
        span = max(self._acc_dt, 1.0e-30)
        self.force = wp.zeros(self.count, dtype=wp.vec3, device=dev, requires_grad=True)
        self.total = wp.zeros_like(self.total, requires_grad=True)
        wp.launch(_sample_kernel, dim=self.count,
                  inputs=[self.acc, self.total_acc, self.time, 1.0 / span, self.frame, self.count_per_world],
                  outputs=[self.force, self.total], device=dev)
        with suspended_tape():
            wp.launch(_quantize_kernel, dim=self.count, inputs=[self.force], outputs=[self.raw], device=dev)
            self.acc = wp.zeros(self.count, dtype=wp.vec3, device=dev, requires_grad=True)
            self.total_acc = wp.zeros_like(self.total_acc, requires_grad=True)
            self.time.zero_()
            self._acc_dt = 0.0

    def readings(self, link: str, world: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """``link``'s last sample (host) in ``world``: taxel forces [N] (shear x, y, normal) and counts."""
        k = self.links.index(link)
        off = world * self.count_per_world
        a, b = off + self.start[k], off + self.start[k + 1]
        return self.force.numpy()[a:b], self.raw.numpy()[a:b]


@wp.kernel
def _tactile_node_q_kernel(pq_in: wp.array[wp.vec3], q_out: wp.array[float]):
    i = wp.tid()
    p = pq_in[i]
    for k in range(3):
        q_out[3 * i + k] = p[k]


@wp.kernel
def _tactile_scalar_q_kernel(x_in: wp.array[float], off: int, q_out: wp.array[float]):
    i = wp.tid()
    q_out[off + i] = x_in[i]


@wp.kernel
def _tactile_arm_q_kernel(body: wp.array[wp.int32], body_global: wp.array[wp.int32], has_global: int,
                          base: wp.array[wp.int32], chained: wp.array[wp.int32], has_chain: int,
                          prev_q: wp.array[wp.transform], has_prev: int, body_q: wp.array[wp.transform],
                          poses: wp.array[wp.transform], body_com: wp.array[wp.vec3], q_out: wp.array[float]):
    b = body[wp.tid()]
    i = base[b]
    if i >= 0:
        if has_chain != 0 and chained[b] != 0:
            if has_prev == 0:
                return
            q = prev_q[b]
        else:
            g = body_global[b] if has_global != 0 else b
            q = body_q[g]
        dc = wp.transform_point(q, body_com[b]) - wp.transform_point(poses[b], body_com[b])
        dr = _rotvec(wp.transform_get_rotation(q) * wp.quat_inverse(wp.transform_get_rotation(poses[b])))
        for a in range(3):
            q_out[i + a] = dc[a]
            q_out[i + 3 + a] = dr[a]


@wp.kernel
def _tactile_poses_kernel(body: wp.array[wp.int32], chained: wp.array[wp.int32], prev_q: wp.array[wp.transform],
                          has_prev: int, body_q: wp.array[wp.transform], poses_out: wp.array[wp.transform]):
    b = body[wp.tid()]
    if chained[b] != 0:
        if has_prev != 0:
            poses_out[b] = prev_q[b]
    else:
        poses_out[b] = body_q[b]


@wp.func
def _reach(x: wp.vec3, n: wp.vec3, p: wp.vec3, frame: wp.mat33, r: float) -> float:
    """A taxel's weight for a contact at ``x`` (normal ``n``): ``(1 - d^2/r^2)^2`` within ``r`` and the cone."""
    d2 = wp.length_sq(x - p)
    if d2 >= r * r or wp.dot(wp.vec3(frame[2, 0], frame[2, 1], frame[2, 2]), n) <= CONE:
        return 0.0
    u = 1.0 - d2 / (r * r)
    return u * u


@wp.kernel(module="unique")
def _taxel_kernel(q: wp.array[float], q0: wp.array[float], q0_snap: wp.array[float], has_q0: int, diff: int,
                  s: Slots, weights: wp.array[wp.vec4], arm: Arm, eps: float, key0: int, b: Bodies,
                  body_q: wp.array[wp.transform], fixed: wp.array[wp.int32], mass: wp.array[float],
                  mass0: wp.array[float], com: wp.array[wp.vec3], lever_off: wp.array[wp.vec3],
                  mu_shape: wp.array[wp.int32], shape_mu: wp.array[float], shape_mu0: wp.array[float],
                  particle_mu: float, trib: wp.array[float], aniso: wp.array[wp.vec3], of_shape: wp.array[wp.int32],
                  sensors_per_world: int, k_slot: wp.array[wp.int32], off_slot: wp.array[wp.int32],
                  x_slot: wp.array[wp.vec3], n_slot: wp.array[wp.vec3], f_slot: wp.array[wp.vec3],
                  total: wp.array[wp.vec3], anchors: wp.array[wp.vec3]):
    i = wp.tid()
    k_slot[i] = -1
    if s.active[i] == 0 or s.key[i] < key0:
        return
    shape = shape_of(s.key[i], key0)
    gid = of_shape[shape]
    if gid < 0:
        return
    w_tax = gid // sensors_per_world
    k = gid - w_tax * sensors_per_world
    ms = mu_shape[s.pair[i][0]]
    mu0 = wp.where(ms >= 0, shape_mu[wp.max(ms, 0)], particle_mu)
    mu0_d = wp.where(ms >= 0, shape_mu0[wp.max(ms, 0)], particle_mu)
    mu_pair = s.mu[i] + (0.5 * (mu0 + shape_mu[shape]) - 0.5 * (mu0_d + shape_mu0[shape]))
    X = _shape_pose(b, shape, body_q)
    inv = wp.transform_inverse(X)  # the sensor's frame at the step's start
    x = wp.transform_point(inv, s.anchor[i])
    n = wp.transform_vector(inv, s.normal[i])
    f = wp.vec3(0.0)
    if diff != 0 and has_q0 != 0:
        pair = s.pair[i]
        w = weights[i]
        th = s.thickness[i]
        moved = s.c0[i] - contact_point(q0_snap, pair, s.w[i], s.anchor[i]) - slot_roll(q0_snap, s, i)
        xq = contact_point(q0, pair, w, wp.vec3(0.0))
        dist, n_w = _shape_gap(b, shape, X, xq, 1.0e6)
        anchor = xq - dist * n_w + moved
        c0 = dist * n_w
        o = wp.dot(n_w, xq - anchor) - LIFT * th
        if o < 0.0:
            anchor = anchor + o * n_w
            c0 = c0 - o * n_w
        anchors[i] = anchor
        anchor_use = anchor + lever_off[i]
        body = arm_body(arm, s, i)
        lever = wp.vec3(0.0)
        aflex = float(0.0)
        if body >= 0:
            lever = anchor_use - com[body]
            aflex = body_flexibility(arm, arm.Minv, body, n_w, lever)
        k_eff = s.stiffness[i] * wp.max(flexibility(pair, s.w[i], mass0, fixed) + arm_flexibility(arm, s, i), 1.0e-30) / wp.max(
            flexibility(pair, w, mass, fixed) + aflex, 1.0e-30)
        d0 = wp.dot(n_w, c0)
        pf = -pad_energy(d0, th, s.pad[i])[1]
        rho = -barrier(d0, th, k_eff, s.lam[i])[1] + pf
        pf0 = -pad_energy(wp.dot(s.normal[i], s.c0[i]), th, s.pad[i])[1]
        damp = s.damp[i] * (pf / wp.max(pf0, 1.0e-30))
        roll, ta, tb = s.roll[i], s.ta[i], s.tb[i]
        c0 = c0 + roll_point(q0, pair, w, roll, n_w, ta, tb)
        cq = contact_point(q, pair, w, anchor) + roll_point(q, pair, w, roll, n_w, ta, tb)
        if body >= 0:
            cq -= arm_move(q, arm, body, lever)
        g, h, fr = contact_force(cq, n_w, c0, th, k_eff, s.lam[i], mu_pair, rho, eps, s.pad[i], damp,
                                 trib[0], trib[1], trib[2], aniso[0])
        if wp.length(g) == 0.0:
            return
        x = wp.transform_point(inv, anchor_use)
        f = wp.transform_vector(inv, g)
        n = wp.transform_vector(inv, n_w)
    else:
        c0 = s.c0[i]
        rho = s.rho[i]
        g, h, fr = contact_force(pair_point(q, s, i, arm, s.anchor[i]), s.normal[i], c0, s.thickness[i],
                                 s.stiffness[i], s.lam[i], mu_pair, rho, eps, s.pad[i], s.damp[i],
                                 trib[0], trib[1], trib[2], aniso[0])
        if wp.length(g) == 0.0:
            return
        f = wp.transform_vector(inv, g)  # the force on the sensor
    wp.atomic_add(total, gid, f)
    x_slot[i] = x
    n_slot[i] = n
    f_slot[i] = f
    off_slot[i] = w_tax
    k_slot[i] = k


@wp.kernel(module="unique")
def _wsum_kernel(k_slot: wp.array[wp.int32], x_slot: wp.array[wp.vec3], n_slot: wp.array[wp.vec3],
                 first: wp.array[wp.int32], radius: wp.array[float], pos: wp.array[wp.vec3],
                 frame: wp.array[wp.mat33], wsum: wp.array[float], near: wp.array[wp.int32]):
    i = wp.tid()
    k = k_slot[i]
    if k < 0:
        return
    x, n = x_slot[i], n_slot[i]
    best, dmin = int(first[k]), float(1.0e30)
    for t in range(first[k], first[k + 1]):
        u = _reach(x, n, pos[t], frame[t], radius[k])
        if u > 0.0:
            wp.atomic_add(wsum, i, u)
        d2 = wp.length_sq(x - pos[t])
        if d2 < dmin:
            best, dmin = t, d2
    near[i] = best


@wp.kernel(module="unique")
def _scatter_kernel(k_slot: wp.array[wp.int32], off_slot: wp.array[wp.int32], count_per_world: int,
                    x_slot: wp.array[wp.vec3], n_slot: wp.array[wp.vec3], f_slot: wp.array[wp.vec3],
                    wsum: wp.array[float], near: wp.array[wp.int32], first: wp.array[wp.int32],
                    radius: wp.array[float], pos: wp.array[wp.vec3], frame: wp.array[wp.mat33],
                    acc: wp.array[wp.vec3]):
    i = wp.tid()
    k = k_slot[i]
    if k < 0:
        return
    tax_off = off_slot[i] * count_per_world
    x, n, f = x_slot[i], n_slot[i], f_slot[i]
    ws = wsum[i]
    if ws == 0.0:
        wp.atomic_add(acc, tax_off + near[i], f)
    else:
        for t in range(first[k], first[k + 1]):
            u = _reach(x, n, pos[t], frame[t], radius[k])
            if u > 0.0:
                wp.atomic_add(acc, tax_off + t, (u / ws) * f)


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
def _sample_kernel(acc: wp.array[wp.vec3], total_acc: wp.array[wp.vec3], time: wp.array[float], inv_span: float,
                   frame: wp.array[wp.mat33], count_per_world: int, force: wp.array[wp.vec3], total: wp.array[wp.vec3]):
    t = wp.tid()
    s = inv_span if inv_span > 0.0 else 1.0 / wp.max(time[0], 1.0e-30)
    f = frame[t % count_per_world] * (acc[t] * s)
    force[t] = wp.vec3(f[0], f[1], -f[2])  # pressing in reads positive
    if t < total.shape[0]:
        total[t] = total_acc[t] * s


@wp.kernel(enable_backward=False)
def _quantize_kernel(force: wp.array[wp.vec3], raw: wp.array[wp.vec3i]):
    t = wp.tid()
    f = force[t]
    raw[t] = wp.vec3i(wp.clamp(int(wp.round(f[0] / LSB)), -128, 127), wp.clamp(int(wp.round(f[1] / LSB)), -128, 127),
                      wp.clamp(int(wp.round(f[2] / LSB)), 0, 255))
