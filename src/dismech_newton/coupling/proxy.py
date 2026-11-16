"""6-DOF rigid-soft proxy coupling for Newton's ``SolverCoupledProxy``."""

from types import SimpleNamespace

import numpy as np
import warp as wp
from newton import BodyFlags, Model, State, StateFlags
from newton.solvers.experimental.coupled import SolverCoupledProxy

from ..contact import Arm

COUPLING_MODULE = wp.get_module("dismech_newton.coupling")

_lagged_step_coupled = SolverCoupledProxy._step_coupled


def _rsr_step_coupled(self: SolverCoupledProxy, state_in: State, state_out: State, control, contacts, dt: float) -> None:
    """Follow the 1-pass proxy solve with a rigid corrector step on the harvested ``F^n`` (``R-S-R``)."""
    for entry in self._entries.values():
        jres = getattr(entry.solver, "joint_residual", None)
        if jres is not None:
            control = jres.apply(state_in, control)
            break
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
        self.last_wrench = wp.zeros(nb, dtype=wp.spatial_vector, device=self.device)
        self._last_grad = None  # the last recorded step's adjoint of its end poses (snapshot)
        self.restart = False  # the next step repeats the last (a coupler's iteration restart): from its start
        self._arm: "Arm | None" = None

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
            self.last_wrench.zero_()
        elif self._body.shape[0]:
            wp.launch(_proxy_reset_kernel, dim=self._body.shape[0], inputs=[self._body, self.model.body_world, mask],
                      outputs=[self._valid, self.last_wrench], device=self.device)

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
        wp.launch(_proxy_wrench_kernel, dim=self._body.shape[0],
                  inputs=[self._body, a.base, z, a.M, a.pred, a.start, self.k, 1.0 / (h * h)],
                  outputs=[self.last_wrench], device=self.device)
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


@wp.kernel(module=COUPLING_MODULE, enable_backward=False)
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


@wp.kernel(module=COUPLING_MODULE, enable_backward=False)
def _proxy_reset_kernel(body: wp.array[wp.int32], body_world: wp.array[wp.int32], mask: wp.array[wp.bool],
                        valid: wp.array[wp.int32], last_wrench: wp.array[wp.spatial_vector]):
    b = body[wp.tid()]
    w = body_world[b]
    if mask[wp.where(w >= 0, w, mask.shape[0] - 1)]:
        valid[b] = 0
        last_wrench[b] = wp.spatial_vector()


@wp.kernel(module=COUPLING_MODULE, enable_backward=False)
def _proxy_wrench_kernel(
    body: wp.array[wp.int32],
    base: wp.array[wp.int32],
    z: wp.array[float],
    arm_M: wp.array3d[float],
    arm_pred: wp.array[float],
    start: int,
    k: int,
    inv_h2: float,
    last_wrench: wp.array[wp.spatial_vector],
):
    """World-frame contact/pin wrench ``M (z - pred) / h^2`` per proxy body."""
    p = wp.tid()
    b = body[p]
    z0 = base[b]
    i = z0 - start
    w = i // k
    i0 = i - w * k
    f = wp.vec3(0.0)
    tau = wp.vec3(0.0)
    for row in range(3):
        fx = float(0.0)
        tx = float(0.0)
        for col in range(6):
            dz = z[z0 + col] - arm_pred[i + col]
            fx += arm_M[w, i0 + row, i0 + col] * dz
            tx += arm_M[w, i0 + 3 + row, i0 + col] * dz
        f[row] = fx * inv_h2
        tau[row] = tx * inv_h2
    last_wrench[b] = wp.spatial_vector(f, tau)


@wp.kernel(module=COUPLING_MODULE)
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


@wp.kernel(module=COUPLING_MODULE, enable_backward=False)
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
