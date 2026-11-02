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
"""

from types import SimpleNamespace

import warp as wp
from newton import State

from .collide import Bodies, _closest_st, _shape_gap, _shape_pose
from .contact import (
    LIFT,
    Slots,
    _barrier_assemble_kernel,
    barrier,
    contact_force,
    contact_hessian,
    contact_point,
    flexibility,
    shape_of,
)
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

CG_TOLERANCE = 1.0e-10  # the contact solve's relative residual, in the preconditioner's norm
CG_ITERATIONS = 100


class StepAdjoint:
    """The adjoint of :meth:`DiSMechSolver.step`, built on its first :meth:`~DiSMechSolver.vjp`."""

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
                            rest=grad(b.count, b.k.vec))
            for b in s.elements if b.count
        ]
        self._intermediates = [self.q_star, self.q_pred, self.q, self.qd, self.edge_d1, self.ref_twist, self.dahl,
                               *(a for e in self.elements for a in (e.eps_prev, e.strain, e.params, e.rest))]
        nd = s.num_node_dofs
        self._q_nodes = self.q[:nd].reshape((nd // 3, 3)).view(wp.vec3)
        self._rhs, self._lam = (wp.zeros(n, dtype=float, device=dev) for _ in range(2))
        self._no_q = wp.zeros(0, dtype=float, device=dev)
        self.dofs = Dofs()  # with the masses computed on the residual's tape, from the model's
        self._cg = [wp.zeros(n, dtype=wp.float64, device=dev) for _ in range(5)]  # x, r, z, p, H p
        self._h = wp.zeros_like(s.A.vals)  # H but the contacts

    def vjp(self, state_in: State, state_out: State, dt: float, contacts: Slots | None = None) -> None:
        """Accumulate the adjoints from ``state_out``'s ``.grad``; both states must still hold the step."""
        s = self.solver
        s_in, s_out = state_in.dismech, state_out.dismech
        seeds = {self.q: s_out.q.grad, self.qd: s_out.qd.grad, self.edge_d1: s_out.edge_d1_q.grad,
                 self.ref_twist: s_out.triplet_ref_twist_q.grad, self.dahl: s_out.triplet_dahl_q.grad,
                 **{e.strain: getattr(s_out, e.blk.stored).grad for e in self.elements}}
        seeds = {a: g for a, g in seeds.items() if g is not None}
        if not seeds:
            return
        dev, n = s.device, s.num_dofs
        h = s.theta * dt  # the implicit Euler step solved
        strains = {b.stored: getattr(s_out, b.stored) for b in s.elements} | {e.blk.stored: e.strain
                                                                             for e in self.elements}  # (empty blocks)
        end = SimpleNamespace(q=self.q, edge_d1_q=self.edge_d1, triplet_ref_twist_q=self.ref_twist,
                              triplet_dahl_q=self.dahl, **strains)
        with suspended_tape():
            for a in self._intermediates:
                a.grad.zero_()
            tape = wp.Tape()
            with tape:
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
                s._carry(state_in.particle_q, s_in, self._q_nodes, end)
            tape.backward(grads=seeds)

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
            if state_in.body_q is not None:
                s.contact.source.bodies.body_q = state_in.body_q
            self._contact_solve(contacts, s.contact.slip_speed * dt)
        self._residual(state_in, dt, contacts).backward(grads={self.residual: self._lam})

    def _contact_solve(self, c: Slots, eps: float) -> None:
        """``(H + sum C^T H_c C) lam = rhs`` by conjugate gradients, preconditioned by ``A`` with the contacts'
        majorizer (exact for a node or rod edge on a shape, but for friction along the slip)."""
        s, A, dev, n = self.solver, self.solver.A, self.solver.device, self.solver.num_dofs
        x, r, z, p, hp = self._cg
        wp.copy(self._h, A.vals)
        wp.launch(_barrier_assemble_kernel, dim=c.active.shape[0],
                  inputs=[self.q_star, c, s.contact.arm, eps, s.fixed, A.indptr, A.indices, A.vals], device=dev)
        wp.launch(_convert, dim=n, inputs=[self._rhs], outputs=[r], device=dev)
        x.zero_()
        self.linear.solve(r, z)
        wp.copy(p, z)
        rz = rz0 = wp.utils.array_inner(r, z)
        for _ in range(CG_ITERATIONS):
            if rz <= CG_TOLERANCE**2 * rz0:
                break
            hp.zero_()
            wp.launch(_spmv_kernel, dim=n, inputs=[A.indptr, A.indices, self._h, p], outputs=[hp], device=dev)
            wp.launch(_contact_matvec_kernel, dim=c.active.shape[0], inputs=[self.q_star, c, eps, s.fixed, p],
                      outputs=[hp], device=dev)
            alpha = rz / wp.utils.array_inner(p, hp)
            wp.launch(_cg_step_kernel, dim=n, inputs=[alpha, p, hp], outputs=[x, r], device=dev)
            self.linear.solve(r, z)
            rz, rz_prev = wp.utils.array_inner(r, z), rz
            wp.launch(_cg_direction_kernel, dim=n, inputs=[rz / rz_prev, z], outputs=[p], device=dev)
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
            tape = wp.Tape()
            with tape:
                wp.launch(k.forces, dim=blk.count,
                          inputs=[e.xs, v, _detached(params), _detached(rest), _detached(e.eps_prev), dt],
                          outputs=[e.g], device=dev)
            for row in range(k.D):
                e.xs.grad.zero_()
                wp.launch(_unit_kernel, dim=(blk.count, k.D), inputs=[row], outputs=[_flat(e.g.grad)], device=dev)
                tape.backward()
                wp.launch(k.hessian_row, dim=blk.count, inputs=[row, e.xs.grad, v, s.fixed, blk.slots],
                          outputs=[A.vals], device=dev)
        dof_mass(s.model, self.mass)
        wp.launch(diagonal_kernel, dim=s.num_dofs, inputs=[self.dofs], outputs=[A.indptr, A.vals], device=dev)

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
            dof_mass(s.model, self.mass)
            wp.launch(_residual_kernel, dim=s.num_dofs, inputs=[self.q_star, self.q_pred, self.dofs],
                      outputs=[self.residual], device=dev)
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
                wp.launch(_contact_residual_kernel, dim=contacts.active.shape[0],
                          inputs=[contacts, src.bodies, src.shape_key0, s.fixed, s.contact.slip_speed * dt,
                                  self.q_star, q_in, _detached(q_in), self.mass, _detached(self.mass)],
                          outputs=[self.residual], device=dev)
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
    """A ``(count, D)`` float view of a vector array."""
    return wp.array(ptr=a.ptr, dtype=float, shape=(a.shape[0], a.dtype._length_), device=a.device)


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
def _unit_kernel(row: int, g: wp.array2d[float]):
    i, k = wp.tid()
    g[i, k] = wp.where(k == row, 1.0, 0.0)


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
def _contact_matvec_kernel(q: wp.array[float], c: Slots, eps: float, fixed: wp.array[wp.int32],
                           x: wp.array[wp.float64], y: wp.array[wp.float64]):
    """``y += C^T H_c C x`` on the free DOFs, :func:`~.contact.contact_hessian` at ``q``."""
    i = wp.tid()
    pair = c.pair[i]
    w = c.w[i]
    v = wp.vec3d()
    for j in range(4):
        if pair[j] >= 0:
            for a in range(3):
                if fixed[3 * pair[j] + a] == 0:
                    v[a] += wp.float64(w[j]) * x[3 * pair[j] + a]
    hv = wp.mat33d(contact_hessian(contact_point(q, pair, w, c.anchor[i]), c, i, eps)) * v
    for j in range(4):
        if pair[j] >= 0:
            for a in range(3):
                if fixed[3 * pair[j] + a] == 0:
                    wp.atomic_add(y, 3 * pair[j] + a, wp.float64(w[j]) * hv[a])


@wp.kernel
def _cg_step_kernel(alpha: wp.float64, p: wp.array[wp.float64], hp: wp.array[wp.float64], x: wp.array[wp.float64],
                    r: wp.array[wp.float64]):
    i = wp.tid()
    x[i] = x[i] + alpha * p[i]
    r[i] = r[i] - alpha * hp[i]


@wp.kernel
def _cg_direction_kernel(beta: wp.float64, z: wp.array[wp.float64], p: wp.array[wp.float64]):
    i = wp.tid()
    p[i] = z[i] + beta * p[i]


@wp.kernel
def _contact_residual_kernel(
    c: Slots,
    b: Bodies,
    key0: int,
    fixed: wp.array[wp.int32],
    eps: float,
    q: wp.array[float],
    q_in: wp.array[float],
    q_in0: wp.array[float],
    mass: wp.array[float],
    mass0: wp.array[float],
    r: wp.array[float],
):
    """``r += C^T`` (the pair's gradient at ``q``) as a function of ``q_in`` and the masses through what the step
    froze from them (``q_in0``, ``mass0``: without their adjoints): a shape's surface point and normal at the start,
    moved as the step moved it and lifted as :func:`~.contact._lift_kernel` lifts it; else the closest points, and the
    normal along them. The stiffness scales with ``1 / (C M^-1 C^T)``; friction's normal force is the barrier's at
    the start."""
    i = wp.tid()
    pair = c.pair[i]
    w = c.w[i]
    th = c.thickness[i]
    anchor = wp.vec3(0.0)
    n = wp.vec3(0.0)
    c0 = wp.vec3(0.0)
    if c.key[i] >= key0:
        moved = c.c0[i] - contact_point(q_in0, pair, w, c.anchor[i])
        sk = shape_of(c.key[i], key0)
        X = _shape_pose(b, sk)
        if pair[1] >= 0 and c.key[i] == key0 + 2 * sk:  # (a line contact's other end: held)
            w = _edge_shape_weights(q_in, pair, w, b, sk, X)
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
    k = c.stiffness[i] * wp.max(flexibility(pair, c.w[i], mass0, fixed), 1.0e-30) / wp.max(
        flexibility(pair, w, mass, fixed), 1.0e-30)
    rho = -barrier(wp.dot(n, c0), th, k, c.lam[i])[1]
    g, h, fr = contact_force(contact_point(q, pair, w, anchor), n, c0, th, k, c.lam[i], c.mu[i] * rho, eps)
    for j in range(4):
        if pair[j] >= 0:
            scatter_node(r, fixed, pair[j], w[j] * g)



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
def _edge_shape_weights(q: wp.array[float], pair: wp.vec4i, w: wp.vec4, b: Bodies, k: int, X: wp.transform) -> wp.vec4:
    """A rod edge's closest point to shape ``k`` at ``q``: one Newton step on the gap's slope along the edge from
    where the step found it (its root there, so the step carries the exact derivative); held on a flat face."""
    xa = node(q, pair[0])
    xb = node(q, pair[1])
    t = w[1]
    h = 1.0e-2
    curvature = (_gap_slope(b, k, X, xa, xb, t + h) - _gap_slope(b, k, X, xa, xb, t - h)) / (2.0 * h)
    if curvature > 0.0:
        t = t - _gap_slope(b, k, X, xa, xb, t) / curvature
    return wp.vec4(1.0 - t, t, 0.0, 0.0)
