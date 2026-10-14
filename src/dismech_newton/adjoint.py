"""Step gradients by the implicit function theorem at the step's solution ``q``.

The step's residual ``r(q) = M (q - q_pred) / dt^2 - f_ext + sum_e J_e^T sigma_e`` has the exact Hessian ``H``: each
element's block by Warp's reverse mode through its ``jacobian`` (one pass per local DOF), assembled into the solver's
pattern and factorised in float64. ``H lam = -adj_q``, then ``lam^T dr/d input`` on the residual's tape.

Contacts (the step's active ones, :meth:`~.contact.ContactBlock.snapshot`) add ``rho C^T (p - Pi(p))`` to ``r`` and
the equation ``d(q) - Pi(p) = 0``, ``p = d(q) + u`` the projection's argument: ``M^T w = [-adj_q, 0]``,
``M = [[H, rho C^T (I - D)], [C, -D]]``, ``D = dPi/dp``, an LU analysed once per pattern of contact pairs. What the
step froze from ``q_in`` (a pair's closest points, normal and reference point; a shape's surface point and normal) is
recomputed from it on a tape. Held: the thickness, friction coefficient and penalty (the converged step does not
depend on it), and the shapes' poses.

The solution's fixed DOFs are ``q_in``'s on the step's tape, so their adjoint (and their coupling, which the residual's
tape leaves there) reaches ``q_in`` by Warp.
"""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import warp as wp
from newton import State

from .collide import Bodies, _closest_st, _shape_gap, _shape_pose
from .contact import Slots, contact_d, contact_point, friction_projection, gap_slip
from .linear import GeneralCSR, sparse_solver
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

# A stuck contact's constraint, softened to a penalty COMPLIANCE^-1 times its own: redundant contacts (overlapping
# features stuck on one sheet) make the exact system singular, while the motion's adjoint is unique.
COMPLIANCE = 1.0e-6


class StepAdjoint:
    """The adjoint of :meth:`DiSMechSolver.step`, built on its first :meth:`~DiSMechSolver.vjp`."""

    max_patterns = 8  # contact patterns whose analysis is kept

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
        self.elements = [
            SimpleNamespace(blk=b, eps_prev=grad(b.count, b.k.vec), strain=grad(b.count, b.k.vec),
                            xs=grad(b.count, b.k.xvec), g=grad(b.count, b.k.xvec))
            for b in s.elements if b.count
        ]
        self._intermediates = [self.q_star, self.q_pred, self.q, self.qd, self.edge_d1, self.ref_twist,
                               *(a for e in self.elements for a in (e.eps_prev, e.strain))]
        nd = s.num_node_dofs
        self._q_nodes = self.q[:nd].reshape((nd // 3, 3)).view(wp.vec3)
        self._rhs, self._lam = (wp.zeros(n, dtype=float, device=dev) for _ in range(2))
        self._no_q = wp.zeros(0, dtype=float, device=dev)
        self.dofs = Dofs()  # with the masses computed on the residual's tape, from the model's
        self._patterns = OrderedDict()  # contact pattern key -> _ContactPattern, least recently used first

    def vjp(self, state_in: State, state_out: State, dt: float, contacts: Slots | None = None) -> None:
        """Accumulate the adjoints from ``state_out``'s ``.grad``; both states must still hold the step."""
        s = self.solver
        s_in, s_out = state_in.dismech, state_out.dismech
        seeds = {self.q: s_out.q.grad, self.qd: s_out.qd.grad, self.edge_d1: s_out.edge_d1_q.grad,
                 self.ref_twist: s_out.triplet_ref_twist_q.grad,
                 **{e.strain: getattr(s_out, e.blk.stored).grad for e in self.elements}}
        seeds = {a: g for a, g in seeds.items() if g is not None}
        if not seeds:
            return
        dev, n = s.device, s.num_dofs
        h = s.theta * dt  # the implicit Euler step solved
        strains = {b.stored: getattr(s_out, b.stored) for b in s.elements} | {e.blk.stored: e.strain
                                                                             for e in self.elements}  # (empty blocks)
        end = SimpleNamespace(q=self.q, edge_d1_q=self.edge_d1, triplet_ref_twist_q=self.ref_twist, **strains)
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
        """``H lam = -adj_q`` (or the contact system), then ``lam^T dr`` into the inputs."""
        s = self.solver
        bind_dofs(self.dofs, s.model, self.mass, s.fixed, s._no_force, state_in, dt)
        wp.launch(_rhs_kernel, dim=s.num_dofs, inputs=[s.fixed], outputs=[self.q_star.grad, self._rhs], device=s.device)
        self._hessian(state_in, dt)
        system = None
        if contacts is not None:
            pairs = contacts.pair.numpy()
            idx = np.lexsort(pairs.T[::-1]).astype(np.int32)  # the pairs' order: the same set, the same pattern
            system = self._pattern(pairs[idx])
            system.set_values(self, contacts, idx)
            system.solve(self._rhs, self._lam)
        else:
            self.linear.invalidate()
            self.linear.solve(self._rhs, self._lam)
        self._residual(state_in, dt).backward(grads={self.residual: self._lam})
        if system is not None and state_in.dismech.q.grad is not None:
            system.input_adjoint(self, state_in, contacts)

    def _hessian(self, state_in: State, dt: float) -> None:
        """``H`` at ``q_star`` into the solver's ``A``: per element and local DOF, a reverse pass of its forces."""
        s = self.solver
        dev, A = s.device, s.A
        A.vals.zero_()
        for e in self.elements:
            blk, k = e.blk, e.blk.k
            v = _detached(blk.bind(state_in.dismech))
            wp.launch(k.gather, dim=blk.count, inputs=[_detached(self.q_star), v], outputs=[e.xs], device=dev)
            tape = wp.Tape()
            with tape:
                wp.launch(k.forces, dim=blk.count,
                          inputs=[e.xs, v, _detached(blk.params), _detached(blk.rest), _detached(e.eps_prev), dt],
                          outputs=[e.g], device=dev)
            for row in range(k.D):
                e.xs.grad.zero_()
                wp.launch(_unit_kernel, dim=(blk.count, k.D), inputs=[row], outputs=[_flat(e.g.grad)], device=dev)
                tape.backward()
                wp.launch(k.hessian_row, dim=blk.count, inputs=[row, e.xs.grad, v, s.fixed, blk.slots],
                          outputs=[A.vals], device=dev)
        dof_mass(s.model, self.mass)
        wp.launch(diagonal_kernel, dim=s.num_dofs, inputs=[self.dofs], outputs=[A.indptr, A.vals], device=dev)

    def _residual(self, state_in: State, dt: float) -> wp.Tape:
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
                wp.launch(k.forces, dim=blk.count, inputs=[e.xs, v, blk.params, blk.rest, e.eps_prev, dt],
                          outputs=[e.g], device=dev)
                wp.launch(k.scatter, dim=blk.count, inputs=[e.g, v, s.fixed], outputs=[self.residual], device=dev)
        return tape

    def _pattern(self, pairs: np.ndarray) -> "_ContactPattern":
        """The cached system for these contact pairs (in this order), least recently used evicted."""
        key = pairs.tobytes()
        p = self._patterns.pop(key, None)
        if p is None:
            p = _ContactPattern(self.solver, pairs)
            while len(self._patterns) >= self.max_patterns:
                self._patterns.popitem(last=False)
        self._patterns[key] = p
        return p


class _ContactPattern:
    """``M^T`` for one list of contact pairs: a pattern independent of the values (cuDSS analyses it once), where
    each entry's value comes from, and the step's work arrays. Rows: the DOFs, then 3 per contact."""

    def __init__(self, solver, pairs: np.ndarray):
        dev, n, m = solver.device, solver.num_dofs, len(pairs)
        self.n, self.m = n, m
        h_rows, h_cols, h_slot = solver.A.entries()
        # C's entries over the free DOFs: per entry its contact, node slot in the pair (0-3), component and DOF.
        dof = 3 * pairs.astype(np.int64)[:, :, None] + np.arange(3)
        contact, slot, comp = np.nonzero((dof >= 0) & (solver.fixed.numpy()[np.maximum(dof, 0)] == 0))
        dof = dof[contact, slot, comp]
        size = n + 3 * m
        lower_rows = n + 3 * contact[:, None] + np.arange(3)  # rho (I - D)^T C: per C entry, its contact's 3 rows
        block = n + 3 * np.arange(m)[:, None, None] + np.zeros((1, 3, 3), dtype=np.int64)
        # The entries: H (both triangles), C^T, rho (I - D)^T C, -D^T.
        rows = np.concatenate([h_rows, dof, lower_rows.ravel(), (block + np.arange(3)[:, None]).ravel()])
        cols = np.concatenate([h_cols, n + 3 * contact + comp, np.repeat(dof, 3), (block + np.arange(3)).ravel()])
        order = np.lexsort((cols, rows))
        indptr = np.zeros(size + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows, minlength=size), out=indptr[1:])
        self.csr = GeneralCSR(sp.csr_matrix((np.zeros(len(rows)), cols[order], indptr), shape=(size, size)), dev)
        self.linear = sparse_solver(self.csr)
        position = np.empty(len(order), dtype=np.int32)  # each entry's index in the CSR values
        position[order] = np.arange(len(order), dtype=np.int32)
        nh, nc = len(h_slot), len(dof)

        def ints(a):
            return wp.array(np.asarray(a).astype(np.int32), dtype=wp.int32, device=dev)

        self._h_slot, self._h_pos = ints(h_slot), ints(position[:nh])
        self._entry = ints(contact), ints(slot), ints(comp)
        self._c_pos, self._lower_pos = ints(position[nh : nh + nc]), ints(position[nh + nc : nh + 4 * nc])
        self._d_pos = ints(position[nh + 4 * nc :])
        self.idx = wp.zeros(m, dtype=wp.int32, device=dev)
        self.jac = wp.zeros(m, dtype=wp.mat33, device=dev)
        self.point = wp.zeros(m, dtype=wp.vec3, device=dev)
        self.b, self.w = (wp.zeros(size, dtype=float, device=dev) for _ in range(2))
        self.w_p = self.w[n:].reshape((m, 3)).view(wp.vec3)
        # The contacts' equations as functions of q_in, differentiated in input_adjoint.
        self.e_q = wp.zeros(n, dtype=float, device=dev, requires_grad=True)
        self.e_p = wp.zeros(m, dtype=wp.vec3, device=dev, requires_grad=True)

    def set_values(self, adjoint: StepAdjoint, c: Slots, idx: np.ndarray) -> None:
        """``M^T``'s values from ``H`` and the contacts' ``D`` at the solution; refactorise on the next solve."""
        vals, dev = self.csr.vals, self.csr.vals.device
        self.idx.assign(idx)
        wp.launch(_linearize_kernel, dim=self.m, inputs=[self.idx, adjoint.q_star, c], outputs=[self.jac, self.point],
                  device=dev)
        wp.launch(_gather_kernel, dim=self._h_slot.shape[0], inputs=[self._h_slot, self._h_pos, adjoint.solver.A.vals],
                  outputs=[vals], device=dev)
        if self._c_pos.shape[0]:
            wp.launch(_contact_values_kernel, dim=self._c_pos.shape[0],
                      inputs=[*self._entry, self._c_pos, self._lower_pos, self.idx, c, self.jac], outputs=[vals],
                      device=dev)
        wp.launch(_minus_d_transpose_kernel, dim=self.m, inputs=[self._d_pos, self.jac], outputs=[vals], device=dev)
        self.linear.invalidate()

    def solve(self, rhs: wp.array, lam: wp.array) -> None:
        """``M^T w = [rhs, 0]``; ``w_q`` into ``lam``."""
        self.b.zero_()
        wp.copy(self.b, rhs, count=self.n)
        self.linear.solve(self.b, self.w)
        wp.copy(lam, self.w, count=self.n)

    def input_adjoint(self, adjoint: StepAdjoint, state_in: State, c: Slots) -> None:
        """``w^T dF/dq_in`` through what the step froze from ``q_in`` into its adjoint."""
        s, q_in = adjoint.solver, state_in.dismech.q
        src = s.contact.source
        if state_in.body_q is not None:
            src.bodies.body_q = state_in.body_q
        self.e_q.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(_frozen_kernel, dim=self.m,
                      inputs=[self.idx, c, src.bodies, src.shape_key0, s.fixed,
                              adjoint.q_star, q_in, _detached(q_in), self.point],
                      outputs=[self.e_q, self.e_p], device=s.device)
        tape.backward(grads={self.e_q: self.w[: self.n], self.e_p: self.w_p})


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


@wp.kernel
def _gather_kernel(
    src_index: wp.array[wp.int32], dst_index: wp.array[wp.int32], src: wp.array[wp.float64], dst: wp.array[wp.float64]
):
    e = wp.tid()
    dst[dst_index[e]] = src[src_index[e]]


# -- contacts -----------------------------------------------------------------------------


@wp.func
def _step(x: float) -> float:
    """``d max(x, 0) / dx``, ``1/2`` at ``0``."""
    return wp.where(x > 0.0, 1.0, wp.where(x < 0.0, 0.0, 0.5))


@wp.kernel
def _linearize_kernel(
    idx: wp.array[wp.int32], q: wp.array[float], c: Slots, jac: wp.array[wp.mat33], point: wp.array[wp.vec3]
):
    """``D = dPi/dp`` at ``p = d(q) + u`` (into ``point``), :func:`~.contact.friction_projection` differentiated."""
    k = wp.tid()
    i = idx[k]
    n = c.normal[i]
    p = contact_d(q, c, i) + c.u[i]
    point[k] = p
    pn = wp.dot(p, n)
    pt = p - pn * n
    r = wp.length(pt)
    P = wp.identity(3, dtype=float) - wp.outer(n, n)
    a = c.mu[i] * wp.max(-pn, 0.0)  # the slip threshold
    D = _step(pn) * wp.outer(n, n)
    if r > 0.0:
        # z_t = g(r) t, g = max(r - a, 0) - max(-r - a, 0)
        t = pt / r
        tt = wp.outer(t, t)
        g = wp.max(r - a, 0.0) - wp.max(-r - a, 0.0)
        g_a = _step(-r - a) - _step(r - a)
        D = D + (_step(r - a) + _step(-r - a)) * tt + (g / r) * (P - tt) - (g_a * c.mu[i] * _step(-pn)) * wp.outer(t, n)
    else:
        D = D + (2.0 * _step(-a)) * P
    jac[k] = D


@wp.kernel
def _contact_values_kernel(
    entry_contact: wp.array[wp.int32],
    entry_slot: wp.array[wp.int32],
    entry_comp: wp.array[wp.int32],
    pos_ct: wp.array[wp.int32],
    pos_lower: wp.array[wp.int32],
    idx: wp.array[wp.int32],
    c: Slots,
    jac: wp.array[wp.mat33],
    vals: wp.array[wp.float64],
):
    """Per entry of ``C``: its weight at ``pos_ct`` (``C^T``), and ``rho ((I - D)^T C)`` in its contact's three rows at
    ``pos_lower``."""
    e = wp.tid()
    k = entry_contact[e]
    i = idx[k]
    comp = entry_comp[e]
    w = c.w[i][entry_slot[e]]
    vals[pos_ct[e]] = wp.float64(w)
    for a in range(3):
        vals[pos_lower[3 * e + a]] = wp.float64(c.rho[i] * (wp.where(a == comp, 1.0, 0.0) - jac[k][comp, a]) * w)


@wp.kernel
def _minus_d_transpose_kernel(pos: wp.array[wp.int32], jac: wp.array[wp.mat33], vals: wp.array[wp.float64]):
    """``-D^T - eps I``, a contact's 3x3 block, at ``pos`` (row-major): :data:`COMPLIANCE`."""
    k = wp.tid()
    for a in range(3):
        for b in range(3):
            vals[pos[9 * k + 3 * a + b]] = wp.float64(-jac[k][b, a] - wp.where(a == b, COMPLIANCE, 0.0))


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


@wp.kernel
def _frozen_kernel(
    idx: wp.array[wp.int32],
    c: Slots,
    b: Bodies,
    key0: int,
    fixed: wp.array[wp.int32],
    q: wp.array[float],
    q_in: wp.array[float],
    q_in0: wp.array[float],
    point: wp.array[wp.vec3],
    e_q: wp.array[float],
    e_p: wp.array[wp.vec3],
):
    """The contacts' equations at the solution ``q`` as functions of ``q_in`` through what the step froze from it,
    differentiated by Warp: ``e_q = rho C^T (p - Pi(p))`` and ``e_p = d(q) - Pi(p)``; ``p`` held, ``q_in0``: ``q_in``
    without its adjoint. (``q``'s adjoint reaches ``q_in`` on its fixed DOFs only.)"""
    k = wp.tid()
    i = idx[k]
    pair = c.pair[i]
    w = c.w[i]
    anchor = wp.vec3(0.0)
    n = wp.vec3(0.0)
    c0 = wp.vec3(0.0)
    if c.key[i] >= key0:  # a shape's: its surface point at the start, moved as the step moved it
        moved = c.c0[i] - contact_point(q_in0, pair, w, c.anchor[i])
        X = _shape_pose(b, c.key[i] - key0)
        if pair[1] >= 0:
            w = _edge_shape_weights(q_in, pair, w, b, c.key[i] - key0, X)
        x = contact_point(q_in, pair, w, wp.vec3(0.0))
        dist, n = _shape_gap(b, c.key[i] - key0, X, x, 1.0e6)
        anchor = x - dist * n + moved
        c0 = dist * n
    else:
        w = _closest_weights(q_in, pair, w)
        c0 = contact_point(q_in, pair, w, anchor)
        n = wp.normalize(c0)
    p = point[k]
    z = friction_projection(p, n, c.mu[i])
    e_p[k] = gap_slip(contact_point(q, pair, w, anchor), n, c0, c.thickness[i]) - z
    for j in range(4):
        if pair[j] >= 0:
            scatter_node(e_q, fixed, pair[j], w[j] * c.rho[i] * (p - z))
