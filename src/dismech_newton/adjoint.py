"""Gradients of a step by the implicit function theorem, not by backpropagating through the iterations.

A step solves ``r(q; inputs) = M alpha (q - q_pred) - f_ext + dE/dq = 0`` on its free DOFs. Newton and
ADMM reach the same root, so the adjoint is one linear solve with the Jacobian at the solution,

    J^T lam = adj_q,      and every input's adjoint is ``-lam^T dr/d input`` (Warp autodiff).

``J`` is not quite the symmetric Hessian ``A`` (DER's twist gradient in the node positions is exact
only while the reference frame is current): ``A`` is factorised and the solve refined against the
exact ``J^T lam``, ``lam += A^-1 (adj_q - J^T lam)``. The rest of the step is ordinary kernels.

Contacts (ADMM). With contact forces ``lam_c``, the root is ``r(q) - C^T lam_c = 0`` with the Coulomb
law, which the ADMM fixed point states for any ``rho > 0`` as

    d(q) = Pi(p),    lam_c = rho (Pi(p) - p),    p = d - lam_c / rho,

``d`` the gap and slip and ``Pi`` the prox of :class:`~dismech_newton.contact.ContactTerm`. Linearised,

    M = [[J, rho C^T (I - D)], [C, -D]],     D = dPi/dp,

and ``M^T w = [adj_q, 0]`` replaces ``J^T lam = adj_q`` (cuDSS LU, refined the same way). ``D`` uses
``Pi`` with its ramps smoothed into the log-barrier prox ``(x + sqrt(x^2 + 4 eps^2)) / 2``, ``eps`` =
``contact_smoothing * h``: exact for ``eps -> 0`` where contacts do not switch, blended within ``eps``
of a switch otherwise. Normals and barycentrics are frozen; the friction coefficient takes a gradient.

:meth:`StepAdjoint.vjp` reads ``state_out``'s ``.grad`` and accumulates into ``state_in``'s
(``q``, ``qd``, ``edge_d1_q``, ``triplet_ref_twist_q``, ``particle_f``) and into these parameters when
they ``requires_grad``: ``model.dismech.triplet_params``, ``solver.triplets.rest``,
``model.dismech.edge_length``, ``model.particle_mass``, ``model.dismech.edge_inertia`` and
``solver.contact.friction`` (call :meth:`DiSMechSolver.refresh_mass` after changing masses).
"""

from contextlib import contextmanager

import numpy as np
import scipy.sparse as sp
import warp as wp
from newton import State

from .contact import (
    ContactSnapshot,
    contact_c_matrix,
    contact_input_adjoint_kernel,
    contact_linearize_kernel,
    contact_transpose_residual_kernel,
)
from .frames import advance_frames_kernel, external_force
from .linear import CudssSolver, GeneralCSR, SymmetricCSR
from .strains import vec5f
from .triplet import advance_ref_twist_kernel, make_residual_kernel


class StepAdjoint:
    """The adjoint of :meth:`DiSMechSolver.step`; built by the solver on first use.

    Attributes:
        refine: Refinement iterations against the exact ``J^T``.
    """

    refine = 2

    def __init__(self, solver):
        self.solver = solver
        tr, dev = solver.triplets, solver.device
        n, nd = solver.num_dofs, solver.num_node_dofs
        self.hessian = SymmetricCSR(n, tr.dofs(), dev)
        self._linear = CudssSolver(self.hessian, refactorize=False)  # once per step, see invalidate
        self._residual_kernel = make_residual_kernel(tr.energy)
        # The per-DOF masses, with an adjoint for the residual to write (mapped to the model's masses).
        self._mass = wp.array(ptr=solver.mass.ptr, dtype=float, shape=solver.mass.shape, device=dev)
        self._mass.requires_grad = True

        def grad_array(count, dtype=float):
            return wp.zeros(count, dtype=dtype, device=dev, requires_grad=True)

        # The step recomputed out of place, every intermediate with its adjoint.
        self.q_theta, self.q_pred, self.q, self.qd, self.residual = (grad_array(n) for _ in range(5))
        self.strain_prev = grad_array(tr.count, vec5f)
        self.edge_d1 = grad_array(solver.der.edge_length.shape[0], wp.vec3)
        self.ref_twist = grad_array(tr.count)
        self._intermediates = (self.q_theta, self.q_pred, self.q, self.qd, self.strain_prev, self.edge_d1,
                               self.ref_twist, self.residual)
        self._q_theta_nodes = self.q_theta[:nd].reshape((nd // 3, 3)).view(wp.vec3)
        self._q_theta_edges = self.q_theta[nd:]
        self._q_nodes = self.q[:nd].reshape((nd // 3, 3)).view(wp.vec3)
        (self._rhs, self._lam, self._neg_lam, self._res, self._step, self._adj_theta,
         self._gradient) = (wp.zeros(n, dtype=float, device=dev) for _ in range(7))  # _gradient: unused

    def vjp(self, state_in: State, state_out: State, dt: float, contacts: ContactSnapshot | None = None) -> None:
        """Accumulate the adjoints of the step ``state_in -> state_out`` from ``state_out``'s ``.grad``.
        ``contacts`` are the step's (:meth:`ContactTerm.snapshot`); both states must still hold the step."""
        s = self.solver
        s_in, s_out = state_in.dismech, state_out.dismech
        seeds = {self.q: s_out.q.grad, self.qd: s_out.qd.grad, self.edge_d1: s_out.edge_d1_q.grad,
                 self.ref_twist: s_out.triplet_ref_twist_q.grad}
        seeds = {a: g for a, g in seeds.items() if g is not None}
        if not seeds:
            return
        dev, n, d = s.device, s.num_dofs, s.der
        h = s.theta * dt
        with suspended_tape():
            for a in self._intermediates:
                a.grad.zero_()
            wp.launch(_q_theta_kernel, dim=n, inputs=[s.fixed, s_in.q, s_out.q, s.theta], outputs=[self.q_theta],
                      device=dev)
            tape = wp.Tape()
            with tape:
                s.triplets.measure(state_in, self.strain_prev)
                wp.launch(_predict_kernel, dim=n, inputs=[s_in.q, s_in.qd, h], outputs=[self.q_pred], device=dev)
                tape.record_func(lambda: self._solve_adjoint(state_in, h, contacts), [self.q_theta])
                wp.launch(_theta_kernel, dim=n, inputs=[s.fixed, s_in.q, s_in.qd, self.q_theta, s.theta, dt],
                          outputs=[self.q, self.qd], device=dev)
                wp.launch(advance_frames_kernel, dim=d.edge_length.shape[0],
                          inputs=[state_in.particle_q, self._q_nodes, d.edge_node0, d.edge_node1, s_in.edge_d1_q],
                          outputs=[self.edge_d1], device=dev)
                wp.launch(advance_ref_twist_kernel, dim=s.triplets.count,
                          inputs=[self._q_nodes, self.edge_d1, s.triplets.conn, s_in.triplet_ref_twist_q],
                          outputs=[self.ref_twist], device=dev)
            tape.backward(grads=seeds)

    def _solve_adjoint(self, state_in: State, h: float, contacts: ContactSnapshot | None) -> None:
        """``J^T lam = adj_q_theta`` (or the contact system), then ``-lam^T dr/d inputs`` into the inputs."""
        s, tr = self.solver, self.solver.triplets
        s_in = state_in.dismech
        dev, n, nd = s.device, s.num_dofs, s.num_node_dofs

        self.hessian.vals.zero_()
        tr.assemble_at(self._q_theta_nodes, self._q_theta_edges, state_in.particle_q, s_in.edge_d1_q,
                       s_in.triplet_ref_twist_q, self.strain_prev, self._gradient, self.hessian, h)
        wp.launch(_inertia_hessian_kernel, dim=n, inputs=[s.mass, 1.0 / (h * h), s.fixed],
                  outputs=[self.hessian.indptr, self.hessian.vals], device=dev)
        wp.launch(_adjoint_rhs_kernel, dim=n, inputs=[s.fixed], outputs=[self.q_theta.grad, self._rhs,
                  self._adj_theta], device=dev)
        contact = _ContactSystem.build(self, state_in, h, contacts)
        if contact is None:
            self._linear.invalidate()
            self._linear.solve(self._rhs, self._lam)
            for _ in range(self.refine):
                self._jt_lam(state_in, h)
                wp.launch(_refine_residual_kernel, dim=n, inputs=[s.fixed, self._rhs, self.q_theta.grad],
                          outputs=[self._res], device=dev)
                self._linear.solve(self._res, self._step)
                wp.launch(_add_kernel, dim=n, inputs=[self._step], outputs=[self._lam], device=dev)
        else:
            contact.solve(state_in, h)
        wp.launch(_negate_kernel, dim=n, inputs=[self._lam], outputs=[self._neg_lam], device=dev)

        # -lam^T dr into the inputs; what it leaves in q_theta is the fixed DOFs' coupling.
        self.q_theta.grad.zero_()
        self._mass.grad.zero_()
        self._residual(state_in, h, inputs_grad=True).backward(grads={self.residual: self._neg_lam})
        if s_in.q.grad is not None:
            wp.launch(_fixed_adjoint_kernel, dim=n, inputs=[s.fixed, self._adj_theta, self.q_theta.grad],
                      outputs=[s_in.q.grad], device=dev)
        if s.model.particle_mass.grad is not None:
            wp.launch(_node_mass_adjoint_kernel, dim=nd // 3, inputs=[self._mass.grad],
                      outputs=[s.model.particle_mass.grad], device=dev)
        if s.der.edge_inertia.grad is not None:
            wp.launch(_add_kernel, dim=n - nd, inputs=[self._mass.grad[nd:]], outputs=[s.der.edge_inertia.grad],
                      device=dev)
        if contact is not None:
            contact.input_adjoint(s_in.q.grad)

    def _jt_lam(self, state_in: State, h: float) -> None:
        """Exact ``J^T lam`` into ``q_theta.grad``."""
        self.q_theta.grad.zero_()
        self._residual(state_in, h, inputs_grad=False).backward(grads={self.residual: self._lam})

    def _residual(self, state_in: State, h: float, inputs_grad: bool) -> wp.Tape:
        """``r(q_theta)`` into :attr:`residual` on a tape; without ``inputs_grad`` only ``q_theta`` takes adjoints."""
        s, tr = self.solver, self.solver.triplets
        s_in = state_in.dismech
        g = (lambda a: a) if inputs_grad else _without_grad
        self.residual.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(_inertia_residual_kernel, dim=s.num_dofs,
                      inputs=[self.q_theta, g(self.q_pred), g(self._mass), 1.0 / (h * h), s.fixed, s.model.gravity,
                              g(s._particle_f(state_in)), s.num_node_dofs],
                      outputs=[self.residual], device=s.device)
            wp.launch(self._residual_kernel, dim=tr.count,
                      inputs=[self.q_theta, g(s_in.q), g(s_in.edge_d1_q), g(s_in.triplet_ref_twist_q), tr.conn,
                              g(tr.der.edge_length), g(tr.params), g(tr.rest), g(self.strain_prev), h,
                              s.num_node_dofs, s.fixed],
                      outputs=[self.residual], device=s.device)
        return tape


class _ContactSystem:
    """The augmented adjoint system ``M^T`` of a step with active contacts, built per step."""

    @classmethod
    def build(cls, adjoint: StepAdjoint, state_in: State, h: float, contacts: ContactSnapshot | None):
        if contacts is None:
            return None
        idx = np.flatnonzero(contacts.active.numpy() != 0).astype(np.int32)
        return cls(adjoint, state_in, h, contacts, idx) if len(idx) else None

    def __init__(self, adjoint: StepAdjoint, state_in: State, h: float, contacts: ContactSnapshot, idx: np.ndarray):
        self.adjoint, self.contacts = adjoint, contacts
        c, s = contacts, adjoint.solver
        dev = s.device
        n, m = self.n, self.m = s.num_dofs, len(idx)
        self.idx = wp.array(idx, dtype=wp.int32, device=dev)
        self.jac = wp.zeros(m, dtype=wp.mat33, device=dev)
        self.jac_mu = wp.zeros(m, dtype=wp.vec3, device=dev)
        wp.launch(contact_linearize_kernel, dim=m,
                  inputs=[self.idx, adjoint.q_theta, state_in.dismech.q, c.pairs, c.bary, c.normal, c.anchor, c.shift,
                          c.thickness, c.force, c.rho, c.friction, c.smoothing * h],
                  outputs=[self.jac, self.jac_mu], device=dev)

        # M^T = [[J^T, C^T], [rho (I - D)^T C, -D^T]], J as A.
        C = contact_c_matrix(c.pairs.numpy()[idx], c.bary.numpy()[idx], s.fixed.numpy() != 0, n).tocsr()
        Dt = self.jac.numpy().astype(np.float64).transpose(0, 2, 1)
        rho = c.rho.numpy()[idx].astype(np.float64)

        def block_diagonal(blocks):
            return sp.bsr_matrix((blocks, np.arange(m), np.arange(m + 1)), shape=(3 * m, 3 * m))

        lower = block_diagonal(rho[:, None, None] * (np.eye(3) - Dt)) @ C
        MT = sp.bmat([[adjoint.hessian.to_scipy(), C.T], [lower, -block_diagonal(Dt)]], format="csr")
        self._linear = CudssSolver(GeneralCSR(MT, dev), refactorize=False)
        self.rhs, self.w, self.res, self.step = (wp.zeros(n + 3 * m, dtype=float, device=dev) for _ in range(4))

    def solve(self, state_in: State, h: float) -> None:
        """``M^T w = [adj_q_theta, 0]``, refined; ``w_q`` into the adjoint's ``lam``."""
        a, s, c = self.adjoint, self.adjoint.solver, self.contacts
        dev, n = s.device, self.n
        w_q, w_p = self.w[:n], self.w[n:]
        wp.copy(self.rhs, a._rhs, count=n)
        self._linear.solve(self.rhs, self.w)
        for _ in range(a.refine):
            wp.copy(a._lam, w_q)
            a._jt_lam(state_in, h)
            wp.launch(_refine_residual_kernel, dim=n, inputs=[s.fixed, a._rhs, a.q_theta.grad],
                      outputs=[self.res[:n]], device=dev)
            wp.launch(contact_transpose_residual_kernel, dim=self.m,
                      inputs=[self.idx, c.pairs, c.bary, s.fixed, self.jac, c.rho, w_q, w_p],
                      outputs=[self.res[:n], self.res[n:]], device=dev)
            self._linear.solve(self.res, self.step)
            wp.launch(_add_kernel, dim=n + 3 * self.m, inputs=[self.step], outputs=[self.w], device=dev)
        wp.copy(a._lam, w_q)

    def input_adjoint(self, q_in_grad: wp.array | None) -> None:
        """The contacts' ``-w^T dF / d input`` into ``q_in_grad`` and the friction coefficient's adjoint."""
        a, s, c = self.adjoint, self.adjoint.solver, self.contacts
        empty = wp.zeros(0, dtype=float, device=s.device)
        wp.launch(contact_input_adjoint_kernel, dim=self.m,
                  inputs=[self.idx, c.pairs, c.bary, c.normal, s.fixed, c.rho, self.jac_mu, a._lam, self.w[self.n:]],
                  outputs=[q_in_grad if q_in_grad is not None else empty,
                           c.friction.grad if c.friction.grad is not None else empty],
                  device=s.device)


@contextmanager
def suspended_tape():
    """Stop recording on the active ``wp.Tape`` (if any) inside the block; yields that tape."""
    runtime = wp._src.context.runtime  # as warp.fem does: kernels that must not be recorded
    tape, runtime.tape = runtime.tape, None
    try:
        yield tape
    finally:
        runtime.tape = tape


def _without_grad(a: wp.array | None) -> wp.array | None:
    """An alias of ``a`` that autodiff does not write adjoints into (``a`` must outlive it)."""
    if a is None or a.grad is None:
        return a
    return wp.array(ptr=a.ptr, dtype=a.dtype, shape=a.shape, strides=a.strides, device=a.device)


# -- kernels ------------------------------------------------------------------------------


@wp.kernel
def _q_theta_kernel(fixed: wp.array[wp.int32], q0: wp.array[float], q: wp.array[float], theta: float,
                    q_theta: wp.array[float]):
    """``q_theta = q0 + theta (q - q0)``, the solved point of a step that ended at ``q``."""
    i = wp.tid()
    if fixed[i] != 0:
        q_theta[i] = q0[i]
    else:
        q_theta[i] = q0[i] + theta * (q[i] - q0[i])


@wp.kernel
def _predict_kernel(q0: wp.array[float], v0: wp.array[float], h: float, q_pred: wp.array[float]):
    i = wp.tid()
    q_pred[i] = q0[i] + h * v0[i]


@wp.kernel
def _theta_kernel(fixed: wp.array[wp.int32], q0: wp.array[float], v0: wp.array[float], q_theta: wp.array[float],
                  theta: float, dt: float, q: wp.array[float], v: wp.array[float]):
    """The solver's extrapolation from ``q_theta`` to the end of the step, out of place."""
    i = wp.tid()
    if fixed[i] != 0:
        q[i] = q_theta[i]
        v[i] = 0.0
        return
    d = q_theta[i] - q0[i]
    q[i] = q0[i] + d / theta
    v[i] = v0[i] + (d - theta * dt * v0[i]) / (theta * theta * dt)


@wp.kernel
def _inertia_residual_kernel(q: wp.array[float], q_pred: wp.array[float], mass: wp.array[float], alpha: float,
                             fixed: wp.array[wp.int32], gravity: wp.array[wp.vec3], particle_f: wp.array[wp.vec3],
                             num_node_dofs: int, residual: wp.array[float]):
    """``residual += M alpha (q - q_pred) - f_ext`` on free DOFs."""
    i = wp.tid()
    if fixed[i] != 0:
        return
    r = mass[i] * alpha * (q[i] - q_pred[i])
    if i < num_node_dofs:
        r = r - external_force(i, mass, gravity, particle_f)
    wp.atomic_add(residual, i, r)


@wp.kernel
def _inertia_hessian_kernel(mass: wp.array[float], alpha: float, fixed: wp.array[wp.int32],
                            hess_indptr: wp.array[wp.int32], hess_vals: wp.array[wp.float64]):
    """``M alpha`` on the diagonal (which leads each row); unit rows for fixed DOFs."""
    i = wp.tid()
    slot = hess_indptr[i]
    if fixed[i] != 0:
        hess_vals[slot] = wp.float64(1.0)
    else:
        hess_vals[slot] = hess_vals[slot] + wp.float64(mass[i] * alpha)


@wp.kernel
def _adjoint_rhs_kernel(fixed: wp.array[wp.int32], adj: wp.array[float], rhs: wp.array[float],
                        adj_copy: wp.array[float]):
    """``rhs = adj`` on free DOFs (``0`` on fixed ones), ``adj_copy = adj``, ``adj = 0``."""
    i = wp.tid()
    a = adj[i]
    adj_copy[i] = a
    adj[i] = 0.0
    if fixed[i] != 0:
        rhs[i] = 0.0
    else:
        rhs[i] = a


@wp.kernel
def _negate_kernel(x: wp.array[float], y: wp.array[float]):
    i = wp.tid()
    y[i] = -x[i]


@wp.kernel
def _refine_residual_kernel(fixed: wp.array[wp.int32], rhs: wp.array[float], jt_lam: wp.array[float],
                            out: wp.array[float]):
    """``out = rhs - J^T lam`` on free DOFs."""
    i = wp.tid()
    if fixed[i] != 0:
        out[i] = 0.0
    else:
        out[i] = rhs[i] - jt_lam[i]


@wp.kernel
def _add_kernel(x: wp.array[float], y: wp.array[float]):
    i = wp.tid()
    y[i] = y[i] + x[i]


@wp.kernel
def _node_mass_adjoint_kernel(adj_mass: wp.array[float], particle_mass_grad: wp.array[float]):
    """A node's mass is that of its three DOFs."""
    n = wp.tid()
    particle_mass_grad[n] = particle_mass_grad[n] + adj_mass[3 * n] + adj_mass[3 * n + 1] + adj_mass[3 * n + 2]


@wp.kernel
def _fixed_adjoint_kernel(fixed: wp.array[wp.int32], adj_theta: wp.array[float], adj_coupling: wp.array[float],
                          adj_q0: wp.array[float]):
    """A fixed DOF of ``q_theta`` is the prescribed ``q0``: its adjoint and coupling go to ``q0``."""
    i = wp.tid()
    if fixed[i] != 0:
        adj_q0[i] = adj_q0[i] + adj_theta[i] + adj_coupling[i]
