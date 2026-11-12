"""Rods and shells as the destination of Newton's ``SolverCoupledProxy``: the source's proxy bodies yield to the
contacts within the soft step (:class:`ProxyArm`), so a pinch holds at the source's servo force."""

from types import SimpleNamespace

import numpy as np
import warp as wp
from newton import BodyFlags, JointType, Model, State, StateFlags
from newton.solvers.experimental.coupled import SolverCoupledProxy

from .contact import STIFFNESS, Arm, arm_assemble, arm_move, arm_scatter
from .linear import row_slot
from .state import node, padded, scatter_node, world_sum

_lagged_step_coupled = SolverCoupledProxy._step_coupled


def _rsr_step_coupled(self: SolverCoupledProxy, state_in: State, state_out: State, control, contacts, dt: float) -> None:
    """Follow the 1-pass proxy solve with a rigid corrector step on the harvested ``F^n`` (``R-S-R``)."""
    _lagged_step_coupled(self, state_in, state_out, control, contacts, dt)
    if int(self._coupling.iterations) != 1:
        return
    for (src_name, dst_name), group in self._proxy_groups.items():
        src, dst = self._entries[src_name], self._entries[dst_name]
        if getattr(dst.solver, "_proxy_arm", None) is None:
            continue
        body_proxies = group["bodies"]
        if not body_proxies or any(int(p.mode) != 0 for p in body_proxies):
            continue
        if src.has_body_force_input:
            self._clear_body_force_input(src)
            self._add_body_force_input(src, src.body_local_to_global, state_in.body_f)
            for proxy in body_proxies:
                self._add_body_force_input(src, proxy.source_local_to_proxy_global, proxy.coupling_forces)
            self._notify_input_state_update(src, StateFlags.BODY_F, dt=dt)
        self._step_entry(src, control, contacts, dt)


SolverCoupledProxy._step_coupled = _rsr_step_coupled


class ProxyArm:
    """Newton ``SolverCoupledProxy``'s proxy bodies (``BodyFlags.PROXY`` in the solver's view) as a soft step's
    :class:`Arm`: 6 DOFs each, its centre of mass's move and a small rotation (world frame), with the inertia of its
    view mass and inertia (the source's effective ones), ``diag(m, R I R^T)``.

    A proxy starts each step where the last left it (feasible: the source's pose, a step behind on the soft side's
    reactions, may overlap the nodes) and its inertia is about the source's own predicted end pose: its synced pose
    moved by ``h qd + h^2 (f / m + g, (R I R^T)^-1 tau)`` (the coupler's synced velocity and rewound ``body_f``). Its
    velocity out is its move from the synced pose over ``h``: the coupler's momentum harvest is then its contact force
    in the step, ``M (z - pred) / h^2``. Proxies without shapes stay out: they touch nothing, and move by their own
    step. The solver builds it itself."""

    def __init__(self, model: Model):
        every = np.flatnonzero(model.body_flags.numpy() & int(BodyFlags.PROXY))
        proxy = np.intersect1d(every, model.shape_body.numpy())  # those with shapes (a frame's has no inertia)
        self._free = wp.array(np.setdiff1d(every, proxy), dtype=wp.int32, device=model.device)
        world = np.maximum(model.body_world.numpy()[proxy], 0)
        slot = np.zeros(len(proxy), dtype=np.int32)  # its place among its world's proxies
        for w in np.unique(world):
            slot[world == w] = np.arange((world == w).sum())
        self.k = 6 * (int(slot.max()) + 1 if len(proxy) else 0)
        # each body's reach from its centre of mass (its shapes' bounding spheres): its farthest point moves at most
        # |dx| + |dth| radius, capped like a node's move (a body is detected like the nodes: at its start pose)
        radius = np.zeros(max(max(model.world_count, 1) * self.k // 6, 1), dtype=np.float32)
        sb, st, sr, com = (a.numpy() for a in (model.shape_body, model.shape_transform, model.shape_collision_radius,
                                               model.body_com))
        for b, w, s in zip(proxy, world, slot, strict=True):
            for j in np.flatnonzero(sb == b):
                i = w * self.k // 6 + s
                radius[i] = max(radius[i], np.linalg.norm(st[j, :3] - com[b]) + sr[j])
        self.radius = wp.array(radius, dtype=float, device=model.device)
        self.model, self.device = model, model.device
        self._world, self._slot = world, slot
        nb = max(model.body_count, 1)
        self._body = wp.array(proxy, dtype=wp.int32, device=self.device)
        self._q = wp.zeros(nb, dtype=wp.transform, device=self.device)  # the step's body poses: the proxies' own
        self._qd = wp.zeros(nb, dtype=wp.spatial_vector, device=self.device)
        self._last = wp.zeros(nb, dtype=wp.transform, device=self.device)  # the proxies' poses the last step left
        self._offset = wp.zeros(nb, dtype=wp.spatial_vector, device=self.device)  # synced pose - start
        self._start = wp.zeros(nb, dtype=wp.transform, device=self.device)  # the proxies' poses the step began at
        self._valid = wp.zeros(nb, dtype=wp.int32, device=self.device)  # per body: it has a last pose
        self._was_valid = wp.zeros_like(self._valid)  # the step's: its start pose is the last step's end
        self._last_grad = None  # the last recorded step's adjoint of its end poses (snapshot)
        self.restart = False  # the next step repeats the last (a coupler's iteration restart): from its start
        self._arm: Arm | None = None

    def bind(self, arm: Arm) -> None:
        self._arm = arm
        base = np.full(arm.base.shape[0], -1, dtype=np.int32)
        base[self._body.numpy()] = arm.start + self._world * self.k + 6 * self._slot  # its 6 in its world's
        arm.base.assign(base)
        eye = np.broadcast_to(np.eye(self.k, dtype=np.float32), arm.M.shape)  # the worlds' unused DOFs: still
        arm.M.assign(eye)
        arm.Minv.assign(eye)

    def reset(self, mask: wp.array | None = None) -> None:
        """Start the proxies from their synced poses next step (those of the worlds of ``mask``, else all)."""
        if mask is None:
            self._valid.zero_()
        elif self._body.shape[0]:
            wp.launch(_proxy_reset_kernel, dim=self._body.shape[0], inputs=[self._body, self.model.body_world, mask],
                      outputs=[self._valid], device=self.device)

    def update(self, state: State, h: float) -> tuple[wp.array, wp.array]:
        """The arm at the step from ``state`` (step ``h``); returns the body poses and velocities for the step: the
        proxies' at their own start, still."""
        m, a = self.model, self._arm
        wp.copy(self._was_valid, self._valid)
        wp.copy(self._q, state.body_q)
        wp.copy(self._qd, state.body_qd)
        wp.launch(_proxy_update_kernel, dim=self._body.shape[0],
                  inputs=[self._body, a.base, a.start, self.k, m.body_world, m.body_mass, m.body_inertia, m.body_com,
                          m.gravity, state.body_q, state.body_qd, state.body_f, self._last, self._valid,
                          int(self.restart), h],
                  outputs=[a.M, a.Minv, a.pred, a.com, self._q, self._qd, self._offset, self._start],
                  device=self.device)
        self.restart = False
        return self._q, self._qd

    def snapshot(self, z: wp.array, state_in: State, pins) -> SimpleNamespace:
        """The step just taken, for its adjoint: the moves ``z``, ``Minv``, the centres of mass, the step's body poses
        (``poses``: a proxy's its start), ``body_f``, the pins' stiffnesses, whether a proxy started from the last
        step's end (``chained``) and the adjoint of that end (``prev``: the last snapshot's ``end_grad``)."""
        a = self._arm
        f = state_in.body_f if state_in.body_f is not None else wp.zeros_like(self._qd)
        out = SimpleNamespace(z=z, Minv=a.Minv, com=a.com, poses=self._q, body_f=f, chained=self._was_valid)
        if pins is not None:
            out.pin_k = pins.p.k
        out = SimpleNamespace(**{k: wp.clone(v) for k, v in vars(out).items()})
        out.poses.requires_grad = True
        out.end_grad = wp.zeros_like(self._q)
        out.prev = self._last_grad
        self._last_grad = out.end_grad
        return out

    def finish(self, z: wp.array, state_in: State, state_out: State, dt: float, h: float) -> None:
        """Pose the proxies in ``state_out`` from their moves ``z`` over ``h``, extrapolated to ``dt``."""
        a, m = self._arm, self.model
        if self._free.shape[0]:
            wp.launch(_proxy_free_kernel, dim=self._free.shape[0],
                      inputs=[self._free, m.body_world, m.body_mass, m.gravity, state_in.body_q, state_in.body_qd,
                              state_in.body_f, dt], outputs=[state_out.body_q, state_out.body_qd], device=self.device)
        wp.launch(_proxy_finish_kernel, dim=self._body.shape[0],
                  inputs=[self._body, a.base, z, a.com, m.body_com, self._q, self._offset, dt, h],
                  outputs=[state_out.body_q, state_out.body_qd, self._last], device=self.device)
        self._valid.fill_(1)


@wp.func
def _rotvec(q: wp.quat) -> wp.vec3:
    """``q``'s rotation vector (the short way)."""
    if q[3] < 0.0:
        q = -q
    v = wp.vec3(q[0], q[1], q[2])
    s = wp.length(v)
    if s < 1.0e-12:
        return 2.0 * v
    return (2.0 * wp.atan2(s, q[3]) / s) * v


@wp.func
def _turn(v: wp.vec3, q: wp.quat) -> wp.quat:
    """``q`` turned by the rotation vector ``v``."""
    angle = wp.length(v)
    if angle > 0.0:
        return wp.normalize(wp.quat_from_axis_angle(v / angle, angle) * q)
    return q


@wp.func
def proxy_step(q0: wp.transform, synced: wp.transform, qd: wp.spatial_vector, f: wp.spatial_vector, mass: float,
               inertia: wp.mat33, com: wp.vec3, g: wp.vec3, h: float, w: int, i: int, i0: int, pred: wp.array[float],
               M: wp.array3d[float], Minv: wp.array3d[float]):
    """A proxy's own move over the step from its start ``q0`` (its synced pose ``synced`` moved by
    ``h qd + h^2 (f / m + g, I^-1 tau)``) into ``pred[i:i+6]`` and its inertia ``diag(m, R I R^T)`` about ``q0`` into
    world ``w``'s ``M``, ``Minv`` at ``i0``; returns the synced pose's offset from ``q0`` (``dc``, ``dr``)."""
    R = wp.quat_to_matrix(wp.transform_get_rotation(q0))
    m = wp.max(mass, 1.0e-12)
    I = R * inertia * wp.transpose(R)
    Iinv = wp.mat33(0.0)
    if wp.determinant(inertia) > 0.0:  # (a shapeless proxy may have none)
        Iinv = R * wp.inverse(inertia) * wp.transpose(R)
    dc = wp.transform_point(synced, com) - wp.transform_point(q0, com)
    dr = _rotvec(wp.transform_get_rotation(synced) * wp.quat_inverse(wp.transform_get_rotation(q0)))
    dx = dc + h * wp.spatial_top(qd) + h * h * (wp.spatial_top(f) / m + g)
    dth = dr + h * wp.spatial_bottom(qd) + h * h * (Iinv * wp.spatial_bottom(f))
    for a in range(3):
        pred[i + a] = dx[a]
        pred[i + 3 + a] = dth[a]
        for c in range(3):
            M[w, i0 + a, i0 + c] = wp.where(a == c, m, 0.0)
            M[w, i0 + 3 + a, i0 + 3 + c] = I[a, c]
            Minv[w, i0 + a, i0 + c] = wp.where(a == c, 1.0 / m, 0.0)
            Minv[w, i0 + 3 + a, i0 + 3 + c] = Iinv[a, c]
    return dc, dr


@wp.kernel(enable_backward=False)
def _proxy_update_kernel(body: wp.array[wp.int32], base: wp.array[wp.int32], start: int, k: int,
                         body_world: wp.array[wp.int32], mass: wp.array[float], inertia: wp.array[wp.mat33],
                         body_com: wp.array[wp.vec3], gravity: wp.array[wp.vec3], body_q: wp.array[wp.transform],
                         body_qd: wp.array[wp.spatial_vector], body_f: wp.array[wp.spatial_vector],
                         last: wp.array[wp.transform], valid: wp.array[wp.int32], restart: int, h: float,
                         M: wp.array3d[float], Minv: wp.array3d[float], pred: wp.array[float], com: wp.array[wp.vec3],
                         q_step: wp.array[wp.transform], qd_step: wp.array[wp.spatial_vector],
                         offset: wp.array[wp.spatial_vector], start_q: wp.array[wp.transform]):
    p = wp.tid()
    b = body[p]
    i = base[b] - start
    w = i // k
    i0 = i - w * k
    synced = body_q[b]
    q0 = synced
    if restart != 0:
        q0 = start_q[b]
    elif valid[b] != 0:
        q0 = last[b]
    start_q[b] = q0
    dc, dr = proxy_step(q0, synced, body_qd[b], body_f[b], mass[b], inertia[b], body_com[b],
                        gravity[wp.max(body_world[b], 0)], h, w, i, i0, pred, M, Minv)
    com[b] = wp.transform_point(q0, body_com[b])
    q_step[b] = q0
    qd_step[b] = wp.spatial_vector()
    offset[b] = wp.spatial_vector(dc, dr)


@wp.kernel(enable_backward=False)
def _proxy_reset_kernel(body: wp.array[wp.int32], body_world: wp.array[wp.int32], mask: wp.array[wp.bool],
                        valid: wp.array[wp.int32]):
    b = body[wp.tid()]
    w = body_world[b]
    if mask[wp.where(w >= 0, w, mask.shape[0] - 1)]:
        valid[b] = 0


@wp.kernel
def _proxy_finish_kernel(body: wp.array[wp.int32], base: wp.array[wp.int32], z: wp.array[float],
                         com: wp.array[wp.vec3], body_com: wp.array[wp.vec3], q_step: wp.array[wp.transform],
                         offset: wp.array[wp.spatial_vector], dt: float, h: float, q_out: wp.array[wp.transform],
                         qd_out: wp.array[wp.spatial_vector], last: wp.array[wp.transform]):
    p = wp.tid()
    b = body[p]
    i = base[b]
    dx = wp.vec3(z[i], z[i + 1], z[i + 2])
    dth = wp.vec3(z[i + 3], z[i + 4], z[i + 5])
    s = dt / h
    rot = _turn(s * dth, wp.transform_get_rotation(q_step[b]))
    q = wp.transform(com[b] + s * dx - wp.quat_rotate(rot, body_com[b]), rot)
    q_out[b] = q
    last[b] = q
    qd_out[b] = wp.spatial_vector((dx - wp.spatial_top(offset[b])) / h, (dth - wp.spatial_bottom(offset[b])) / h)


@wp.kernel(enable_backward=False)
def _proxy_free_kernel(body: wp.array[wp.int32], body_world: wp.array[wp.int32], mass: wp.array[float],
                       gravity: wp.array[wp.vec3], body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                       body_f: wp.array[wp.spatial_vector], dt: float, q_out: wp.array[wp.transform],
                       qd_out: wp.array[wp.spatial_vector]):
    """A proxy without shapes: its velocity by its force and gravity (its spin kept), symplectic Euler."""
    b = body[wp.tid()]
    v = wp.spatial_top(body_qd[b]) + dt * (gravity[wp.max(body_world[b], 0)])
    if mass[b] > 0.0:
        v += dt * wp.spatial_top(body_f[b]) / mass[b]
    om = wp.spatial_bottom(body_qd[b])
    rot = _turn(dt * om, wp.transform_get_rotation(body_q[b]))
    q_out[b] = wp.transform(wp.transform_get_translation(body_q[b]) + dt * v, rot)
    qd_out[b] = wp.spatial_vector(v, om)


# -- pins: Newton's ball joints between a proxy body and a rod ---------------------------------


@wp.struct
class Pins:
    """Per pin its node, body, point in the body's frame, that point at the step's start and stiffness."""

    node: wp.array[wp.int32]
    body: wp.array[wp.int32]
    local: wp.array[wp.vec3]
    point: wp.array[wp.vec3]
    k: wp.array[float]


class ProxyPins:
    """Newton's ``BALL`` joints in the entry between a :class:`ProxyArm` body and a rod's capsule (either as the
    parent): the capsule's nearer end node held to the joint's point on the body by ``k/2 |x - p - P z|^2`` in the
    step, ``p`` the point at the step's start and ``P z`` its move by the body's DOFs (as a contact's anchor), ``k``
    :data:`~.contact.STIFFNESS` times the node's inertia ``m / h^2`` (a gap ``f h^2 / (STIFFNESS m)`` under a force
    ``f``). The body takes the reaction in the step, and the coupler its momentum: a cable fixed to a rigid body."""

    def __init__(self, model: Model, arm: Arm, mass: wp.array, fixed: wp.array):
        self.arm, self.mass, self.fixed, self.device = arm, mass, fixed, model.device
        edge = {int(b): e for e, b in enumerate(model.dismech.edge_body.numpy()) if b >= 0}
        base = arm.base.numpy()
        rows = []
        if model.joint_count:
            kind, on = model.joint_type.numpy(), model.joint_enabled.numpy()
            parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
            X_p, X_c = model.joint_X_p.numpy(), model.joint_X_c.numpy()
            node0, node1 = model.dismech.edge_node0.numpy(), model.dismech.edge_node1.numpy()
            for j in np.flatnonzero((kind == int(JointType.BALL)) & on):
                (b, xb), (c, xc) = (parent[j], X_p[j]), (child[j], X_c[j])
                if b in edge:
                    (b, xb), (c, xc) = (c, xc), (b, xb)
                if c not in edge:
                    continue
                if b < 0 or base[b] < 0:
                    raise ValueError(f"joint {j} pins a rod to body {b}, not a proxy with shapes in this entry")
                e = edge[c]
                rows.append((node0[e] if xc[2] < 0.0 else node1[e], b, xb[:3]))  # the capsule's axis is its +Z
        self.count = len(rows)
        p, n = Pins(), max(self.count, 1)
        p.node = wp.array([r[0] for r in rows] or [0], dtype=wp.int32, device=self.device)
        p.body = wp.array([r[1] for r in rows] or [0], dtype=wp.int32, device=self.device)
        p.local = wp.array([r[2] for r in rows] or [np.zeros(3)], dtype=wp.vec3, device=self.device)
        p.point = wp.zeros(n, dtype=wp.vec3, device=self.device)
        p.k = wp.zeros(n, dtype=float, device=self.device)
        self.p = p

    def _launch(self, kernel, inputs, dim=None):
        wp.launch(kernel, dim=dim or self.count, inputs=inputs, device=self.device)

    def begin_step(self, body_q: wp.array, h: float) -> None:
        """The points at the step's start (``body_q``: the proxies' own start poses) and the stiffnesses."""
        self._launch(_pin_begin_kernel, [self.p, body_q, self.mass, STIFFNESS / (h * h)])

    def grad(self, q: wp.array, b: wp.array) -> None:
        self._launch(_pin_grad_kernel, [q, self.p, self.arm, self.fixed, b])

    def commit(self, q: wp.array, pick: wp.array, b: wp.array) -> None:
        self.grad(q, b)

    def energy(self, q_cand: wp.array, energy: wp.array) -> None:
        self._launch(_pin_energy_kernel, [q_cand, self.p, self.arm, self.count, self.world, self.stride, energy],
                     dim=(q_cand.shape[0], padded(self.count)))

    def assemble(self, q: wp.array, A) -> None:
        self._launch(_pin_assemble_kernel, [self.p, self.arm, self.fixed, A.indptr, A.indices, A.vals])

    def reactions(self, q: wp.array, body_f: wp.array) -> None:
        """Add each pin's force at ``q`` on its body (and its torque about the centre of mass at the step's start)."""
        self._launch(_pin_reaction_kernel, [q, self.p, self.arm, body_f])


@wp.func
def _pin_gap(q: wp.array[float], p: Pins, i: int, arm: Arm) -> wp.vec3:
    """``x - p - P z``."""
    b = p.body[i]
    return node(q, p.node[i]) - p.point[i] - arm_move(q, arm, b, p.point[i] - arm.com[b])


@wp.kernel(enable_backward=False)
def _pin_begin_kernel(p: Pins, body_q: wp.array[wp.transform], mass: wp.array[float], scale: float):
    i = wp.tid()
    p.point[i] = wp.transform_point(body_q[p.body[i]], p.local[i])
    p.k[i] = scale * mass[3 * p.node[i]]


@wp.kernel(enable_backward=False)
def _pin_grad_kernel(q: wp.array[float], p: Pins, arm: Arm, fixed: wp.array[wp.int32], b: wp.array[wp.float64]):
    i = wp.tid()
    g = p.k[i] * _pin_gap(q, p, i, arm)
    scatter_node(b, fixed, p.node[i], g)
    arm_scatter(b, arm, p.body[i], p.point[i] - arm.com[p.body[i]], g)


@wp.kernel(enable_backward=False)
def _pin_energy_kernel(q_cand: wp.array2d[float], p: Pins, arm: Arm, count: int, world: wp.array[wp.int32],
                       stride: int, energy: wp.array[wp.float64]):
    j, i = wp.tid()
    e = wp.float64(0.0)
    w = int(-1)
    if i < count:
        w = world[3 * p.node[i]]
        c = _pin_gap(q_cand[j], p, i, arm)
        e = wp.float64(0.5 * p.k[i] * wp.dot(c, c))
    world_sum(energy, w, j, stride, e)


@wp.kernel(enable_backward=False)
def _pin_assemble_kernel(p: Pins, arm: Arm, fixed: wp.array[wp.int32], indptr: wp.array[wp.int32],
                         indices: wp.array[wp.int32], vals: wp.array[wp.float64]):
    """``k [I, -P; -P^T, P^T P]`` into the upper CSR (all in the pattern: a node and its world's arm)."""
    i = wp.tid()
    k, n, body = p.k[i], p.node[i], p.body[i]
    r = p.point[i] - arm.com[body]
    for c in range(3):
        if fixed[3 * n + c] == 0:
            wp.atomic_add(vals, row_slot(indptr, indices, 3 * n + c, 3 * n + c), wp.float64(k))
    eye = wp.identity(3, float)
    arm_assemble(vals, indptr, indices, fixed, arm, body, r, k * eye, n, eye)
    arm_assemble(vals, indptr, indices, fixed, arm, body, r, k * eye, -1, eye)


@wp.kernel(enable_backward=False)
def _pin_reaction_kernel(q: wp.array[float], p: Pins, arm: Arm, body_f: wp.array[wp.spatial_vector]):
    i = wp.tid()
    f = p.k[i] * _pin_gap(q, p, i, arm)  # pulls the body toward the node
    body = p.body[i]
    wp.atomic_add(body_f, body, wp.spatial_vector(f, wp.cross(p.point[i] - arm.com[body], f)))
