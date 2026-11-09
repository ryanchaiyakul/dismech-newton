"""Step gradients by the implicit function theorem at the step's solution ``q``.

The step's residual ``r(q) = M (q - q_pred) / dt^2 - f_ext + sum_e J_e^T sigma_e`` has the exact Hessian ``H``: each
element's block by Warp's reverse mode through its ``jacobian`` (one pass per local DOF), assembled into the solver's
pattern and factorised in float64. ``H lam = -adj_q``, then ``lam^T dr/d input`` on the residual's tape.

Contacts touching at the step's end (:meth:`~.contact.BarrierBlock.snapshot`) add their barrier and friction
gradients to ``r``, and their exact Hessian to ``H``, which couples a pair's two sides outside ``A``'s pattern:
``H lam = -adj_q`` by conjugate gradients, preconditioned by ``A`` with the contacts majorized as the step assembles
them. What the step froze from ``q_in`` (a pair's closest points, normal, start and lagged normal force; a shape's
surface point and normal) and its stiffness's dependence on the masses are recomputed on the residual's tape. Held:
the thickness, friction coefficient, load scale and the shapes' poses; exact when the step found its pairs at its
start (not again at an iterate).

The solution's fixed DOFs are ``q_in``'s on the step's tape, so their adjoint (and their coupling, which the residual's
tape leaves there) reaches ``q_in`` by Warp.

Proxy bodies (:class:`~.coupling.ProxyArm`): their moves ``z`` are unknowns of the solve, with their inertia
``M (z - pred) / h^2``, their contacts' and pins' terms. Their own step ``pred`` (from the synced ``body_q``,
``body_qd``), ``M``, the pins' points and the shapes' poses are rebuilt on the tape from the proxies' start poses,
whose adjoint goes to the last step's end poses (``z``'s, its output) or, on a proxy's first step, the synced pose.
The lever arms of their contacts are the anchors' from the start poses' centres of mass: in a pinch, whose squeeze
force is large, a body's move turns it. Held: ``body_f`` and ``M^-1`` in the pairs' stiffnesses.
"""

from types import SimpleNamespace
from typing import Any

import warp as wp
from newton import State

from .collide import FLAT, OPEN, Bodies, _closest_st, _edge_scan, _shape_gap, _shape_pose
from .contact import (
    LIFT,
    SIDES,
    Arm,
    Slots,
    _arm_assemble_kernel,
    _barrier_assemble_kernel,
    arm_body,
    arm_flexibility,
    arm_move,
    barrier,
    body_flexibility,
    contact_force,
    contact_hessian,
    contact_point,
    flexibility,
    node_map,
    roll_point,
    shape_of,
    slot_map,
    slot_roll,
)
from .coupling import Pins, _pin_assemble_kernel, _pin_gap, _proxy_finish_kernel, proxy_step
from .linear import _convert, sparse_solver
from .state import (
    Dofs,
    bind_dofs,
    diagonal_kernel,
    dof_mass,
    end_kernel,
    inertia_grad,
    node,
    predict_kernel,
    scatter_node,
    suspended_tape,
)

CG_TOLERANCE = 1.0e-8  # the contact solve's relative residual, in the preconditioner's norm (exact but for rod-rod pairs)
CG_ITERATIONS = 100


class StepAdjoint:
    """The adjoint of :meth:`SolverDiSMech.step`, built on its first :meth:`~SolverDiSMech.vjp`."""

    def __init__(self, solver):
        s, dev = solver, solver.device
        n = s.num_dofs
        self.solver = s
        self.linear = sparse_solver(s.A)  # float64 on the solver's pattern

        def grad(count, dtype=float):
            return wp.zeros(count, dtype=dtype, device=dev, requires_grad=True)

        # The step recomputed out of place, every intermediate with its adjoint; q_star the solution.
        self.q_star, self.q_pred, self.q, self.qd, self.residual, self.mass = (grad(n) for _ in range(6))
        self.edge_d1 = grad(s.der.edge_length.shape[0], wp.vec3)
        self.ref_twist = grad(s.triplets.count)
        self.dahl = grad(s.triplets.count, wp.vec3)
        self.elements = [
            SimpleNamespace(blk=b, eps_prev=grad(b.count, b.k.vec), strain=grad(b.count, b.k.vec),
                            xs=grad(b.count, b.k.xvec), g=grad(b.count, b.k.xvec), params=grad(b.count, b.params.dtype),
                            xs_rows=grad((b.count, b.k.D), b.k.xvec), g_rows=grad((b.count, b.k.D), b.k.xvec),
                            rest=grad(b.count, b.k.vec))
            for b in s.elements if b.count
        ]
        self._intermediates = [self.q_star, self.q_pred, self.q, self.qd, self.edge_d1, self.ref_twist, self.dahl,
                               *(a for e in self.elements for a in (e.eps_prev, e.strain, e.params, e.rest))]
        nd = s.num_node_dofs
        self._q_nodes = self.q[:nd].reshape((nd // 3, 3)).view(wp.vec3)
        self._rhs, self._lam = (wp.zeros(n, dtype=float, device=dev) for _ in range(2))
        self._no_q = wp.zeros(0, dtype=float, device=dev)
        self._no_com = wp.zeros(1, dtype=wp.vec3, device=dev)
        self.dofs = Dofs()  # with the masses computed on the residual's tape, from the model's
        self._cg = [wp.zeros(n, dtype=wp.float64, device=dev) for _ in range(5)]  # x, r, z, p, H p
        self._dots = wp.zeros(3, dtype=wp.float64, device=dev)
        self._h = wp.zeros_like(s.A.vals)  # H but the contacts
        self.n0 = s.num_state_dofs
        self.arm = self.pins = self._snap = None  # the recorded step's (vjp)
        if s._proxy_arm is not None:  # the proxies' step from their start poses: own moves, inertia, ends, pins
            live, nb = s.contact.arm, s.contact.arm.base.shape[0]
            self.pred, self.M = grad(live.pred.shape[0]), wp.clone(live.M, requires_grad=True)
            self.Minv = wp.clone(live.Minv, requires_grad=True)
            self.end, self._com = grad(nb, wp.transform), grad(nb, wp.vec3)
            self._end_qd = wp.zeros(nb, dtype=wp.spatial_vector, device=dev)
            self._end_last = wp.zeros(nb, dtype=wp.transform, device=dev)
            self._offset = wp.zeros(nb, dtype=wp.spatial_vector, device=dev)
            self._com_end = wp.zeros(nb, dtype=wp.transform, device=dev)  # (a sink)
            self.pin_point = grad(max(s.pins.p.point.shape[0], 1), wp.vec3)
            self._intermediates += [self.pred, self.M, self.Minv, self.end, self._com, self.pin_point]

    def vjp(self, state_in: State, state_out: State, dt: float, contacts: Slots | None = None, arm=None) -> None:
        """Accumulate the adjoints from ``state_out``'s ``.grad``; both states must still hold the step (``arm``: its
        proxies', :meth:`~.coupling.ProxyArm.snapshot`)."""
        s = self.solver
        s_in, s_out = state_in.dismech, state_out.dismech
        seeds = {self.q: s_out.q.grad, self.qd: s_out.qd.grad, self.edge_d1: s_out.edge_d1_q.grad,
                 self.ref_twist: s_out.triplet_ref_twist_q.grad, self.dahl: s_out.triplet_dahl_q.grad,
                 **{e.strain: getattr(s_out, e.blk.stored).grad for e in self.elements}}
        seeds = {a: g for a, g in seeds.items() if g is not None}
        if not seeds:
            return
        dev, n = s.device, self.n0
        h = s.theta * dt  # the implicit Euler step solved
        self._bind_arm(arm, h)
        strains = {b.stored: getattr(s_out, b.stored) for b in s.elements} | {e.blk.stored: e.strain
                                                                             for e in self.elements}  # (empty blocks)
        end = SimpleNamespace(q=self.q, edge_d1_q=self.edge_d1, triplet_ref_twist_q=self.ref_twist,
                              triplet_dahl_q=self.dahl, **strains)
        with suspended_tape():
            for a in self._intermediates:
                a.grad.zero_()
            if arm is not None:
                wp.copy(self.q_star, arm.z, dest_offset=n)  # the arms' moves: the solve accounts for their inputs
                seeds[self.end] = arm.end_grad
            pa, m = s._proxy_arm, s.model
            tape = wp.Tape()
            with tape:
                if arm is not None:
                    wp.launch(_arm_pred_kernel, dim=pa._body.shape[0],
                              inputs=[pa._body, self.arm.base, self.arm.start, self.arm.k, m.body_world, m.body_mass,
                                      m.body_inertia, m.body_com, m.gravity, state_in.body_q, state_in.body_qd,
                                      arm.body_f, arm.poses, h], outputs=[self.pred, self.M, self.Minv], device=dev)
                    wp.launch(_arm_com_kernel, dim=pa._body.shape[0], inputs=[pa._body, m.body_com, arm.poses],
                              outputs=[self._com], device=dev)  # (before the solve: its lever arms use them)
                    if self.pins is not None:
                        wp.launch(_pin_point_kernel, dim=self.pins.node.shape[0],
                                  inputs=[s.pins.p.body, s.pins.p.local, arm.poses], outputs=[self.pin_point],
                                  device=dev)
                for e in self.elements:
                    wp.launch(e.blk.k.previous, dim=e.blk.count,
                              inputs=[getattr(s_in, e.blk.stored), s_in.q, e.blk.bind(s_in)], outputs=[e.eps_prev],
                              device=dev)
                wp.launch(predict_kernel, dim=n, inputs=[s.fixed, s_in.q, s_in.qd, h],
                          outputs=[self.q_pred, self._no_q], device=dev)
                # the solution: the step's free DOFs (the output's: a drive may have moved its fixed ones since)
                wp.launch(_solution_kernel, dim=n,
                          inputs=[s.fixed, s_in.q, _detached(s_in.q), _detached(s_out.q), s.theta],
                          outputs=[self.q_star], device=dev)
                tape.record_func(lambda: self._solve(state_in, h, contacts), [self.q_star])
                wp.launch(end_kernel, dim=n, inputs=[s.fixed, s_in.q, s_in.qd, self.q_star, s.theta, dt],
                          outputs=[self.q, self.qd], device=dev)
                if arm is not None:  # the proxies' end poses (the next step's start)
                    wp.launch(_proxy_finish_kernel, dim=pa._body.shape[0],
                              inputs=[pa._body, self.arm.base, self.q_star, self._com, m.body_com, arm.poses,
                                      self._offset, dt, h], outputs=[self.end, self._end_qd, self._end_last],
                              device=dev)
                s._carry(state_in.particle_q, s_in, self._q_nodes, end)
            tape.backward(grads=seeds)
            if arm is not None:  # the start poses' adjoint: to the last step's ends, or the synced poses
                arm.end_grad.zero_()
                prev = arm.prev if arm.prev is not None else arm.end_grad
                wp.launch(_route_kernel, dim=pa._body.shape[0],
                          inputs=[pa._body, arm.chained, int(arm.prev is not None), arm.poses.grad],
                          outputs=[prev, state_in.body_q.grad if state_in.body_q.requires_grad else self._com_end],
                          device=dev)
                arm.poses.grad.zero_()

    def _bind_arm(self, snap, h: float) -> None:
        """The recorded step's arm and pins (``snap``: :meth:`~.coupling.ProxyArm.snapshot`), or none."""
        s = self.solver
        self.arm = self.pins = None
        if snap is None:
            return
        a, live = Arm(), s.contact.arm
        for name in Arm.vars:
            setattr(a, name, getattr(live, name))
        a.M, a.Minv, a.com, a.pred = _detached(self.M), snap.Minv, snap.com, _detached(self.pred)
        a.inv_h2 = 1.0 / (h * h)
        self.arm, self._snap = a, snap
        if s.pins is not None and s.pins.count:
            p, live = Pins(), s.pins.p
            p.node, p.body, p.local, p.point, p.k = live.node, live.body, live.local, self.pin_point, snap.pin_k
            self.pins = p

    def _solve(self, state_in: State, dt: float, contacts: Slots | None) -> None:
        """``H lam = -adj_q``, then ``lam^T dr`` into the inputs."""
        s = self.solver
        bind_dofs(self.dofs, s.model, self.mass, s.fixed, s._no_force, state_in, dt)
        wp.launch(_rhs_kernel, dim=s.num_dofs, inputs=[s.fixed], outputs=[self.q_star.grad, self._rhs], device=s.device)
        self._hessian(state_in, dt)
        self.linear.invalidate()
        if contacts is None:
            self.linear.solve(self._rhs, self._lam)
        else:
            if self.arm is not None:  # the shapes where the step had them: the proxies at their start
                s.contact.source.bodies.body_q = self._snap.poses
            elif state_in.body_q is not None:
                s.contact.source.bodies.body_q = state_in.body_q
            self._contact_solve(contacts, s.contact.slip_speed * dt)
        tape = self._residual(state_in, dt, contacts)
        tape.backward(grads={self.residual: self._lam})

    def _contact_solve(self, c: Slots, eps: float) -> None:
        """``(H + sum C^T H_c C) lam = rhs`` by conjugate gradients, preconditioned by ``A`` with the contacts'
        majorizer (exact for a node or rod edge on a shape, but for friction along the slip)."""
        s, A, dev, n = self.solver, self.solver.A, self.solver.device, self.solver.num_dofs
        x, r, z, p, hp = self._cg
        wp.copy(self._h, A.vals)
        arm = self.arm if self.arm is not None else s.contact.arm
        wp.launch(_barrier_assemble_kernel, dim=c.active.shape[0],
                  inputs=[self.q_star, c, arm, eps, s.fixed, 1, A.indptr, A.indices, A.vals], device=dev)
        wp.launch(_convert, dim=n, inputs=[self._rhs], outputs=[r], device=dev)
        x.zero_()
        self.linear.solve(r, z)
        wp.copy(p, z)
        dots = self._dots  # r.z, p.Hp, the next r.z: on the device, one read per iteration
        dots.zero_()
        wp.launch(_dot_kernel, dim=n, inputs=[r, z, 0], outputs=[dots], device=dev)
        rz = rz0 = float(dots.numpy()[0])
        for _ in range(CG_ITERATIONS):
            if rz <= CG_TOLERANCE**2 * rz0:
                break
            hp.zero_()
            wp.launch(_spmv_kernel, dim=n, inputs=[A.indptr, A.indices, self._h, p], outputs=[hp], device=dev)
            wp.launch(_contact_matvec_kernel, dim=c.active.shape[0], inputs=[self.q_star, c, arm, eps, s.fixed, p],
                      outputs=[hp], device=dev)
            wp.launch(_dot_kernel, dim=n, inputs=[p, hp, 1], outputs=[dots], device=dev)
            wp.launch(_cg_step_kernel, dim=n, inputs=[dots, p, hp], outputs=[x, r], device=dev)
            self.linear.solve(r, z)
            wp.launch(_dot_kernel, dim=n, inputs=[r, z, 2], outputs=[dots], device=dev)
            wp.launch(_cg_direction_kernel, dim=n, inputs=[dots, z], outputs=[p], device=dev)
            wp.launch(_cg_roll_kernel, dim=1, inputs=[dots], device=dev)
            rz = float(dots.numpy()[0])
        wp.launch(_convert, dim=n, inputs=[x], outputs=[self._lam], device=dev)

    def _hessian(self, state_in: State, dt: float) -> None:
        """``H`` at ``q_star`` into the solver's ``A``: per element and local DOF, a reverse pass of its forces."""
        s = self.solver
        dev, A = s.device, s.A
        A.vals.zero_()
        for e in self.elements:
            blk, k = e.blk, e.blk.k
            v = _detached(blk.bind(state_in.dismech))
            wp.launch(k.gather, dim=blk.count, inputs=[_detached(self.q_star), v], outputs=[e.xs], device=dev)
            params, rest = self._params(e, state_in, dt)
            wp.launch(_rows_kernel, dim=(blk.count, k.D), inputs=[e.xs], outputs=[e.xs_rows], device=dev)
            tape = wp.Tape()
            with tape:
                wp.launch(k.forces_rows, dim=(blk.count, k.D),
                          inputs=[e.xs_rows, v, _detached(params), _detached(rest), _detached(e.eps_prev), dt],
                          outputs=[e.g_rows], device=dev)
            e.xs_rows.grad.zero_()
            wp.launch(_unit_kernel, dim=(blk.count * k.D, k.D), inputs=[k.D], outputs=[_flat(e.g_rows.grad)], device=dev)
            tape.backward()
            wp.launch(k.hessian_rows, dim=(blk.count, k.D), inputs=[e.xs_rows.grad, v, s.fixed, blk.slots],
                      outputs=[A.vals], device=dev)
        dof_mass(s.model, self.mass[: self.n0])
        wp.launch(diagonal_kernel, dim=s.num_dofs, inputs=[self.dofs], outputs=[A.indptr, A.vals], device=dev)
        if self.arm is not None:  # the arms' inertia, the pins (quadratic: exact)
            wp.launch(_arm_assemble_kernel, dim=(self.arm.M.shape[0], self.arm.k),
                      inputs=[self.arm, A.indptr, A.indices, A.vals], device=dev)
            if self.pins is not None:
                wp.launch(_pin_assemble_kernel, dim=self.pins.node.shape[0],
                          inputs=[self.pins, self.arm, s.fixed, A.indptr, A.indices, A.vals], device=dev)

    @staticmethod
    def _params(e, state_in: State, dt: float):
        """The element's ``(params, rest)`` in the step (with Dahl: from ``state_in``'s stresses, on any tape)."""
        if e.blk.effective is None:
            return e.blk.params, e.blk.rest
        return e.blk.effective(state_in.dismech, dt, e.params, e.rest)

    def _residual(self, state_in: State, dt: float, contacts: Slots | None) -> wp.Tape:
        """``r(q_star)`` on a tape, every input with its adjoint."""
        s, dev = self.solver, self.solver.device
        self.residual.zero_()
        self.mass.grad.zero_()
        tape = wp.Tape()
        with tape:
            dof_mass(s.model, self.mass[: self.n0])
            wp.launch(_residual_kernel, dim=self.n0, inputs=[self.q_star, self.q_pred, self.dofs],
                      outputs=[self.residual], device=dev)
            arm = self.arm if self.arm is not None else s.contact.arm
            if self.arm is not None:
                wp.launch(_arm_residual_kernel, dim=(arm.M.shape[0], arm.k),
                          inputs=[self.q_star, self.pred, self.M, arm], outputs=[self.residual], device=dev)
                if self.pins is not None:
                    wp.launch(_pin_residual_kernel, dim=self.pins.node.shape[0],
                              inputs=[self.q_star, self.pins, self.pin_point, arm, s.fixed], outputs=[self.residual],
                              device=dev)
            for e in self.elements:
                blk, k, v = e.blk, e.blk.k, e.blk.bind(state_in.dismech)
                e.xs.grad.zero_()
                e.g.grad.zero_()
                wp.launch(k.gather, dim=blk.count, inputs=[self.q_star, v], outputs=[e.xs], device=dev)
                params, rest = self._params(e, state_in, dt)
                wp.launch(k.forces, dim=blk.count, inputs=[e.xs, v, params, rest, e.eps_prev, dt],
                          outputs=[e.g], device=dev)
                wp.launch(k.scatter, dim=blk.count, inputs=[e.g, v, s.fixed], outputs=[self.residual], device=dev)
            if contacts is not None:
                src, q_in = s.contact.source, state_in.dismech.q
                poses = src.bodies.body_q if self.arm is None else self._snap.poses
                n_c = contacts.active.shape[0]
                t0 = wp.empty(n_c, dtype=float, device=dev)
                wp.launch(_scan_kernel, dim=n_c, inputs=[contacts, src.bodies, _detached(poses), src.shape_key0,
                                                         _detached(q_in)], outputs=[t0], device=dev, record_tape=False)
                weights = wp.zeros(n_c, dtype=wp.vec4, device=dev, requires_grad=True)
                wp.launch(_weights_kernel, dim=n_c, inputs=[contacts, src.bodies, poses, _detached(poses),
                                                           src.shape_key0, q_in, _detached(q_in), t0],
                          outputs=[weights], device=dev)
                # the lever arms from the step's anchors (a body's point, which its surface point at q_in can miss
                # along the surface): their values, the surface point's derivative
                inputs = [contacts, src.bodies, poses, weights, arm, src.shape_key0, s.fixed,
                          s.contact.slip_speed * dt, self.q_star, q_in, _detached(q_in), self.mass,
                          _detached(self.mass), self._com if self.arm is not None else self._no_com,
                          self.Minv if self.arm is not None else arm.Minv]
                off, anchors = (wp.zeros(n_c, dtype=wp.vec3, device=dev) for _ in range(2))
                if self.arm is not None:
                    wp.launch(_contact_residual_kernel, dim=n_c, inputs=[*inputs, off],
                              outputs=[wp.zeros_like(self.residual), anchors], device=dev, record_tape=False)
                    wp.launch(_lever_off_kernel, dim=n_c, inputs=[contacts.anchor, anchors], outputs=[off], device=dev,
                              record_tape=False)
                wp.launch(_contact_residual_kernel, dim=n_c, inputs=[*inputs, off], outputs=[self.residual, anchors],
                          device=dev)
        return tape


def _detached(a):
    """An alias of an array, or of a ``wp.struct``'s arrays, without adjoints (``a`` must outlive it)."""
    if hasattr(a, "_cls"):
        out = type(a)()
        for name in a._cls.vars:
            setattr(out, name, _detached(getattr(a, name)))
        return out
    if not isinstance(a, wp.array) or a.grad is None:
        return a
    return wp.array(ptr=a.ptr, dtype=a.dtype, shape=a.shape, strides=a.strides, device=a.device)


def _flat(a: wp.array) -> wp.array:
    """A ``(count, D)`` float view of a vector array (2d: its rows, flattened)."""
    return wp.array(ptr=a.ptr, dtype=float, shape=(a.size, a.dtype._length_), device=a.device)


# -- kernels ------------------------------------------------------------------------------


@wp.kernel
def _solution_kernel(fixed: wp.array[wp.int32], q0: wp.array[float], q0_held: wp.array[float], q1: wp.array[float],
                     theta: float, q_star: wp.array[float]):
    """The solved step from its extrapolated end ``q1``; a free DOF's from ``q0_held``, ``q0`` detached (the solve
    accounts for its dependence on the inputs)."""
    i = wp.tid()
    q_star[i] = wp.where(fixed[i] != 0, q0[i], q0_held[i] + theta * (q1[i] - q0_held[i]))


@wp.kernel
def _residual_kernel(q: wp.array[float], q_pred: wp.array[float], f: Dofs, r: wp.array[float]):
    """``r += M (q - q_pred) / dt^2 - f_ext`` on free DOFs."""
    i = wp.tid()
    if f.fixed[i] == 0:
        wp.atomic_add(r, i, inertia_grad(f, i, q[i] - q_pred[i]))


@wp.kernel
def _rhs_kernel(fixed: wp.array[wp.int32], adj: wp.array[float], rhs: wp.array[float]):
    """``rhs = -adj`` on free DOFs, whose ``adj`` the solve takes over (a fixed DOF's goes on to ``q_in``)."""
    i = wp.tid()
    if fixed[i] != 0:
        rhs[i] = 0.0
    else:
        rhs[i] = -adj[i]
        adj[i] = 0.0


@wp.kernel
def _unit_kernel(D: int, g: wp.array2d[float]):
    """``g[i D + row]``: the unit vector ``row``."""
    j, k = wp.tid()
    g[j, k] = wp.where(k == j % D, 1.0, 0.0)


@wp.kernel(module="unique")  # generic
def _rows_kernel(xs: wp.array(dtype=Any), rows: wp.array2d(dtype=Any)):
    i, row = wp.tid()
    rows[i, row] = xs[i]


# -- contacts ------------------------------------------------------------------------------


@wp.kernel
def _spmv_kernel(indptr: wp.array[wp.int32], indices: wp.array[wp.int32], vals: wp.array[wp.float64],
                 x: wp.array[wp.float64], y: wp.array[wp.float64]):
    """``y += A x``, ``A`` symmetric by its upper CSR."""
    i = wp.tid()
    acc = wp.float64(0.0)
    for k in range(indptr[i], indptr[i + 1]):
        j = indices[k]
        acc += vals[k] * x[j]
        if j != i:
            wp.atomic_add(y, j, vals[k] * x[i])
    wp.atomic_add(y, i, acc)


@wp.kernel
def _contact_matvec_kernel(q: wp.array[float], c: Slots, arm: Arm, eps: float, fixed: wp.array[wp.int32],
                           x: wp.array[wp.float64], y: wp.array[wp.float64]):
    """``y += J^T H_c J x`` on the free DOFs (``J``: :func:`~.contact.node_map` per node, ``-P`` on a yielding
    body's DOFs), :func:`~.contact.contact_hessian` at ``q``."""
    i = wp.tid()
    pair = c.pair[i]
    w = c.w[i]
    v = wp.vec3d()
    for j in range(4):
        if pair[j] >= 0:
            xj = wp.vec3d()
            for a in range(3):
                if fixed[3 * pair[j] + a] == 0:
                    xj[a] = x[3 * pair[j] + a]
            v += wp.mat33d(slot_map(c, i, j)) * xj
    body = arm_body(arm, c, i)
    r = wp.vec3d()
    zi = int(0)
    cq = contact_point(q, pair, w, c.anchor[i]) + slot_roll(q, c, i)
    if body >= 0:
        zi = arm.base[body]
        r = wp.vec3d(c.anchor[i] - arm.com[body])
        v -= wp.vec3d(x[zi], x[zi + 1], x[zi + 2]) + wp.cross(wp.vec3d(x[zi + 3], x[zi + 4], x[zi + 5]), r)
        cq -= arm_move(q, arm, body, c.anchor[i] - arm.com[body])
    hv = wp.mat33d(contact_hessian(cq, c, i, eps)) * v
    if body >= 0:
        t = wp.cross(r, hv)
        for a in range(3):
            wp.atomic_add(y, zi + a, -hv[a])
            wp.atomic_add(y, zi + 3 + a, -t[a])
    for j in range(4):
        if pair[j] >= 0:
            yj = wp.transpose(wp.mat33d(slot_map(c, i, j))) * hv
            for a in range(3):
                if fixed[3 * pair[j] + a] == 0:
                    wp.atomic_add(y, 3 * pair[j] + a, yj[a])


@wp.kernel
def _cg_step_kernel(dots: wp.array[wp.float64], p: wp.array[wp.float64], hp: wp.array[wp.float64],
                    x: wp.array[wp.float64], r: wp.array[wp.float64]):
    i = wp.tid()
    alpha = dots[0] / dots[1]
    x[i] = x[i] + alpha * p[i]
    r[i] = r[i] - alpha * hp[i]


@wp.kernel
def _cg_direction_kernel(dots: wp.array[wp.float64], z: wp.array[wp.float64], p: wp.array[wp.float64]):
    i = wp.tid()
    p[i] = z[i] + dots[2] / dots[0] * p[i]


@wp.kernel
def _dot_kernel(a: wp.array[wp.float64], b: wp.array[wp.float64], slot: int, dots: wp.array[wp.float64]):
    i = wp.tid()
    wp.atomic_add(dots, slot, a[i] * b[i])


@wp.kernel
def _cg_roll_kernel(dots: wp.array[wp.float64]):
    dots[0] = dots[2]
    dots[1] = wp.float64(0.0)
    dots[2] = wp.float64(0.0)


@wp.kernel
def _contact_residual_kernel(
    c: Slots,
    b: Bodies,
    poses: wp.array[wp.transform],
    weights: wp.array[wp.vec4],
    arm: Arm,
    key0: int,
    fixed: wp.array[wp.int32],
    eps: float,
    q: wp.array[float],
    q_in: wp.array[float],
    q_in0: wp.array[float],
    mass: wp.array[float],
    mass0: wp.array[float],
    com: wp.array[wp.vec3],
    Minv: wp.array3d[float],
    lever_off: wp.array[wp.vec3],
    r: wp.array[float],
    anchors: wp.array[wp.vec3],
):
    """``r += C^T`` (the pair's gradient at ``q``) as a function of ``q_in`` and the masses through what the step
    froze from them (``q_in0``, ``mass0``: without their adjoints): a shape's surface point and normal at the start,
    moved as the step moved it and lifted as :func:`~.contact._lift_kernel` lifts it; else the closest points, and the
    normal along them. The stiffness scales with ``1 / (C M^-1 C^T)``; friction's normal force is the barrier's at
    the start. A yielding body's lever arms are from its centre of mass ``com`` at its start pose."""
    i = wp.tid()
    pair = c.pair[i]
    w = c.w[i]
    th = c.thickness[i]
    anchor = wp.vec3(0.0)
    n = wp.vec3(0.0)
    c0 = wp.vec3(0.0)
    if c.key[i] >= key0:
        moved = c.c0[i] - contact_point(q_in0, pair, w, c.anchor[i]) - slot_roll(q_in0, c, i)  # the anchor's (the
        # start's surface turn is ``c0``'s, below)
        sk = shape_of(c.key[i], key0)
        X = _shape_pose(b, sk, poses)  # (``poses``: ``b.body_q``, with its adjoint)
        w = weights[i]
        x = contact_point(q_in, pair, w, wp.vec3(0.0))
        dist, n = _shape_gap(b, sk, X, x, 1.0e6)
        anchor = x - dist * n + moved
        c0 = dist * n
        o = wp.dot(n, x - anchor) - LIFT * th
        if o < 0.0:
            anchor = anchor + o * n
            c0 = c0 - o * n
    else:
        w = _closest_weights(q_in, pair, w)
        c0 = contact_point(q_in, pair, w, anchor)
        n = wp.normalize(c0)
    body = arm_body(arm, c, i)
    lever = wp.vec3(0.0)
    aflex = float(0.0)
    if body >= 0:  # the anchor moved by its body's DOFs, about its centre of mass (``com``: the start pose's)
        lever = wp.where(c.key[i] >= key0, anchor + lever_off[i], c.anchor[i]) - com[body]
        aflex = body_flexibility(arm, Minv, body, n, lever)
    k = c.stiffness[i] * wp.max(flexibility(pair, c.w[i], mass0, fixed) + arm_flexibility(arm, c, i), 1.0e-30) / wp.max(
        flexibility(pair, w, mass, fixed) + aflex, 1.0e-30)
    rho = -barrier(wp.dot(n, c0), th, k, c.lam[i])[1]
    roll, ta, tb = c.roll[i], c.ta[i], c.tb[i]  # the rods' surface turn, as the step froze it
    c0 = c0 + roll_point(q_in, pair, w, roll, n, ta, tb)
    cq = contact_point(q, pair, w, anchor) + roll_point(q, pair, w, roll, n, ta, tb)
    if body >= 0:
        cq -= arm_move(q, arm, body, lever)
    anchors[i] = anchor
    g, h, fr = contact_force(cq, n, c0, th, k, c.lam[i], c.mu[i] * rho, eps, c.pad[i], c.damp[i])
    for j in range(4):
        if pair[j] >= 0:
            scatter_node(r, fixed, pair[j], wp.transpose(node_map(w, roll, n, ta, tb, j)) * g)
    if body >= 0:
        _arm_scatter(r, arm, body, lever, g)


@wp.kernel(enable_backward=False)
def _lever_off_kernel(slot: wp.array[wp.vec3], anchor: wp.array[wp.vec3], off: wp.array[wp.vec3]):
    i = wp.tid()
    off[i] = slot[i] - anchor[i]


@wp.kernel(module="unique", enable_backward=False)
def _scan_kernel(c: Slots, b: Bodies, poses: wp.array[wp.transform], key0: int, q: wp.array[float],
                 t0: wp.array[float]):
    """A line contact's other end's closest point, as the step scanned it (:func:`~.collide._edge_scan`)."""
    i = wp.tid()
    t0[i] = 0.0
    if c.key[i] >= key0 and c.pair[i][1] >= 0:
        sk = shape_of(c.key[i], key0)
        if c.key[i] != key0 + SIDES * sk:
            t0[i] = _edge_scan(b, sk, _shape_pose(b, sk, poses), node(q, c.pair[i][0]), node(q, c.pair[i][1]), 1.0e6)[0]


@wp.kernel(module="unique")
def _weights_kernel(c: Slots, b: Bodies, poses: wp.array[wp.transform], poses0: wp.array[wp.transform], key0: int,
                    q: wp.array[float], q0: wp.array[float], t0: wp.array[float], out: wp.array[wp.vec4]):
    """A shape pair's weights: the step's, moving as a rod edge's closest point or a line contact's other end does
    (:func:`_edge_shape_weights`; ``poses0``, ``q0``: ``poses``, ``q`` without adjoints)."""
    i = wp.tid()
    w = c.w[i]
    if c.key[i] >= key0 and c.pair[i][1] >= 0:
        sk = shape_of(c.key[i], key0)
        w = _edge_shape_weights(q, q0, c.pair[i], w, b, sk, _shape_pose(b, sk, poses), _shape_pose(b, sk, poses0),
                                c.key[i] != key0 + SIDES * sk, t0[i])
    out[i] = w


@wp.func
def _closest_weights(q: wp.array[float], pair: wp.vec4i, w: wp.vec4) -> wp.vec4:
    """The pair's weights at ``q``, its features as the step found them: two edges' closest points, or node ``0``'s
    closest point on the line or plane through the other nodes (a node's alone: unchanged)."""
    if pair[3] >= 0 and w[1] > 0.0:
        st = _closest_st(q, pair[0], pair[1], pair[2], pair[3])
        return wp.vec4(1.0 - st[0], st[0], st[1] - 1.0, -st[1])
    j0 = int(-1)
    j1 = int(-1)
    j2 = int(-1)
    for k in range(1, 4):
        if pair[k] >= 0:
            if j0 < 0:
                j0 = k
            elif j1 < 0:
                j1 = k
            else:
                j2 = k
    if j1 < 0:
        return w
    x = node(q, pair[0]) - node(q, pair[j0])
    e0 = node(q, pair[j1]) - node(q, pair[j0])
    b1 = wp.dot(x, e0) / wp.dot(e0, e0)
    b2 = float(0.0)
    if j2 >= 0:  # barycentric coordinates of the projection onto the triangle's plane
        e1 = node(q, pair[j2]) - node(q, pair[j0])
        d00 = wp.dot(e0, e0)
        d01 = wp.dot(e0, e1)
        d11 = wp.dot(e1, e1)
        den = d00 * d11 - d01 * d01
        b1 = (d11 * wp.dot(x, e0) - d01 * wp.dot(x, e1)) / den
        b2 = (d00 * wp.dot(x, e1) - d01 * wp.dot(x, e0)) / den
    out = wp.vec4(1.0, 0.0, 0.0, 0.0)
    for k in range(1, 4):
        if k == j0:
            out[k] = b1 + b2 - 1.0
        elif k == j1:
            out[k] = -b1
        elif k == j2:
            out[k] = -b2
    return out


@wp.func
def _gap_slope(b: Bodies, k: int, X: wp.transform, xa: wp.vec3, xb: wp.vec3, t: float) -> float:
    d, n = _shape_gap(b, k, X, wp.lerp(xa, xb, t), 1.0e6)
    return wp.dot(n, xb - xa)


@wp.func
def _edge_closest(xa: wp.vec3, xb: wp.vec3, xa0: wp.vec3, xb0: wp.vec3, t: float, b: Bodies, k: int, X: wp.transform,
                  X0: wp.transform) -> float:
    """A rod edge's closest point to shape ``k``, the root of the gap's slope ``s`` along the edge, which the step
    found at ``t``: ``t - (s - s0) / s'`` (``s0``, ``s'`` without adjoints: ``t``'s value, the root's derivative),
    ``s`` evaluated on the slope's curved side: at a face's end (an edge along the face) it is flat on the other, so
    the point moves with the face's end. Held on a flat face."""
    h = 1.0e-2
    s0 = _gap_slope(b, k, X0, xa0, xb0, t)
    below = s0 - _gap_slope(b, k, X0, xa0, xb0, t - h)
    above = _gap_slope(b, k, X0, xa0, xb0, t + h) - s0
    if wp.max(below, above) > 0.0:
        side = wp.where(above > below, 1.0, -1.0)
        e = 1.0e-3  # just past the root, and the curvature there
        u = t + side * e
        su = _gap_slope(b, k, X0, xa0, xb0, u)
        curvature = side * (_gap_slope(b, k, X0, xa0, xb0, u + side * e) - su) / e
        if curvature > 0.0:
            t = t - (_gap_slope(b, k, X, xa, xb, u) - su) / curvature
    return t


@wp.func
def _face_turn(b: Bodies, k: int, X: wp.transform, xa: wp.vec3, xb: wp.vec3, u: float, nt: wp.vec3) -> float:
    d, n = _shape_gap(b, k, X, wp.lerp(xa, xb, u), 1.0e6)
    return wp.dot(n, nt) - (1.0 - FLAT)


@wp.func
def _edge_shape_weights(q: wp.array[float], q0: wp.array[float], pair: wp.vec4i, w: wp.vec4, b: Bodies, k: int,
                        X: wp.transform, X0: wp.transform, other: bool, t0: float) -> wp.vec4:
    """A rod edge's point against shape ``k`` at ``q``, ``X`` (``q0``, ``X0``: the same without adjoints), the step's
    ``w`` moving as its closest point does (:func:`_edge_closest`), or a line contact's other end (``other``; ``t0``
    the closest point the step scanned from, :func:`_scan_kernel`): where the shape's normal leaves the closest
    point's by :data:`~.collide.FLAT` (a face's end, which moves with the shape), ``t - (f - f0) / f'`` likewise."""
    xa = node(q, pair[0])
    xb = node(q, pair[1])
    xa0 = node(q0, pair[0])
    xb0 = node(q0, pair[1])
    t = w[1]
    if other:
        d, nt0 = _shape_gap(b, k, X0, wp.lerp(xa0, xb0, t0), 1.0e6)
        h = 1.0e-3
        slope = (_face_turn(b, k, X0, xa0, xb0, t + h, nt0) - _face_turn(b, k, X0, xa0, xb0, t - h, nt0)) / (2.0 * h)
        if wp.abs(slope) > 1.0e-3:  # (else held)
            tc = t0
            if t0 > OPEN and t0 < 1.0 - OPEN:  # (an end: the node's)
                tc = _edge_closest(xa, xb, xa0, xb0, t0, b, k, X, X0)
            d, nt = _shape_gap(b, k, X, wp.lerp(xa, xb, tc), 1.0e6)
            f = _face_turn(b, k, X, xa, xb, t, nt)
            t = t - (f - _face_turn(b, k, X0, xa0, xb0, t, nt0)) / slope
    else:
        t = _edge_closest(xa, xb, xa0, xb0, t, b, k, X, X0)
    return wp.vec4(1.0 - t, t, 0.0, 0.0)


# -- proxies -------------------------------------------------------------------------------


@wp.func
def _arm_scatter(r: wp.array[float], arm: Arm, body: int, x: wp.vec3, g: wp.vec3):
    """``r -= P^T g`` (:func:`~.contact.arm_scatter` into the residual)."""
    i = arm.base[body]
    t = wp.cross(x, g)
    for a in range(3):
        wp.atomic_add(r, i + a, -g[a])
        wp.atomic_add(r, i + 3 + a, -t[a])


@wp.kernel
def _arm_residual_kernel(q: wp.array[float], pred: wp.array[float], M: wp.array3d[float], arm: Arm,
                         r: wp.array[float]):
    """``r += M (z - pred) / h^2``, world ``w``'s DOF ``m``."""
    w, m = wp.tid()
    v = float(0.0)
    for m2 in range(arm.k):
        j = w * arm.k + m2
        v += M[w, m, m2] * (q[arm.start + j] - pred[j])
    wp.atomic_add(r, arm.start + w * arm.k + m, v * arm.inv_h2)


@wp.kernel
def _pin_residual_kernel(q: wp.array[float], p: Pins, point: wp.array[wp.vec3], arm: Arm, fixed: wp.array[wp.int32],
                         r: wp.array[float]):
    """``r += k (x - p - P z)`` on the node, ``-P^T`` of it on the body (``point``: ``p.point``, with its adjoint)."""
    i = wp.tid()
    g = p.k[i] * (_pin_gap(q, p, i, arm) + p.point[i] - point[i])
    scatter_node(r, fixed, p.node[i], g)
    _arm_scatter(r, arm, p.body[i], p.point[i] - arm.com[p.body[i]], g)


@wp.kernel
def _arm_pred_kernel(body: wp.array[wp.int32], base: wp.array[wp.int32], start: int, k: int,
                     body_world: wp.array[wp.int32], mass: wp.array[float], inertia: wp.array[wp.mat33],
                     body_com: wp.array[wp.vec3], gravity: wp.array[wp.vec3], body_q: wp.array[wp.transform],
                     body_qd: wp.array[wp.spatial_vector], body_f: wp.array[wp.spatial_vector],
                     q0: wp.array[wp.transform], h: float, pred: wp.array[float], M: wp.array3d[float],
                     Minv: wp.array3d[float]):
    """The proxies' own moves over the step and their rotational inertia, as
    :func:`~.coupling._proxy_update_kernel` has them, from the synced ``body_q``, ``body_qd`` and their start poses
    ``q0``."""
    p = wp.tid()
    b = body[p]
    i = base[b] - start
    w = i // k
    i0 = i - w * k
    proxy_step(q0[b], body_q[b], body_qd[b], body_f[b], mass[b], inertia[b], body_com[b],
               gravity[wp.max(body_world[b], 0)], h, w, i, i0, pred, M, Minv)


@wp.kernel
def _arm_com_kernel(body: wp.array[wp.int32], body_com: wp.array[wp.vec3], q0: wp.array[wp.transform],
                    com: wp.array[wp.vec3]):
    b = body[wp.tid()]
    com[b] = wp.transform_point(q0[b], body_com[b])


@wp.kernel
def _pin_point_kernel(body: wp.array[wp.int32], local: wp.array[wp.vec3], q0: wp.array[wp.transform],
                      point: wp.array[wp.vec3]):
    i = wp.tid()
    point[i] = wp.transform_point(q0[body[i]], local[i])


@wp.kernel
def _route_kernel(body: wp.array[wp.int32], chained: wp.array[wp.int32], has_prev: int,
                  g: wp.array[wp.transform], prev: wp.array[wp.transform], synced: wp.array[wp.transform]):
    """A proxy's start pose's adjoint ``g``: to the last step's end (``prev``) if it started there, else to its synced
    pose (a coupler's iteration restart: unsupported)."""
    b = body[wp.tid()]
    if chained[b] != 0:
        if has_prev != 0:
            prev[b] = prev[b] + g[b]
    else:
        synced[b] = synced[b] + g[b]
