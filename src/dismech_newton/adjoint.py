"""Step gradients by the implicit function theorem: ``J^T lam = adj_q`` at the solution, then ``-lam^T dr/d input``.

``lam`` is refined against the exact ``J^T`` with the symmetric Hessian ``A``: ``lam += A^-1 (adj_q - J^T lam)``.
Contacts: ``M^T w = [adj_q, 0]``, ``M = [[J, rho C^T (I - D)], [C, -D]]``, ``D = dPi/dp`` (smoothed).
"""

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
from .linear import sparse_solver
from .solver import advance_frames_kernel, inertia_kernel, predict_kernel, suspended_tape, theta_kernel
from .sparse import GeneralCSR, SymmetricCSR
from .strains import vec5f
from .triplet import advance_ref_twist_kernel


class StepAdjoint:
    """The adjoint of :meth:`DiSMechSolver.step`, built on first use."""

    refine = 2

    def __init__(self, solver):
        self.solver = solver
        tr, dev = solver.triplets, solver.device
        n, nd = solver.num_dofs, solver.num_node_dofs
        self.hessian = SymmetricCSR(n, tr.dofs(), dev)
        self._linear = sparse_solver(self.hessian, refactorize=False)  # once per step, see invalidate
        self._residual_kernel = tr.kernels.residual
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
        self.strain = grad_array(tr.count, vec5f)
        self._intermediates = (self.q_theta, self.q_pred, self.q, self.qd, self.strain_prev, self.edge_d1,
                               self.ref_twist, self.strain, self.residual)
        self._q_nodes = self.q[:nd].reshape((nd // 3, 3)).view(wp.vec3)
        self._rhs, self._lam, self._neg_lam, self._res, self._step, self._adj_theta = (
            wp.zeros(n, dtype=float, device=dev) for _ in range(6))
        self._no_q = wp.zeros(0, dtype=float, device=dev)
        self._no_indptr = wp.zeros(0, dtype=wp.int32, device=dev)
        self._no_vals = wp.zeros(0, dtype=wp.float64, device=dev)

    def vjp(self, state_in: State, state_out: State, dt: float, contacts: ContactSnapshot | None = None) -> None:
        """Accumulate the adjoints from ``state_out``'s ``.grad``; both states must still hold the step."""
        s = self.solver
        s_in, s_out = state_in.dismech, state_out.dismech
        seeds = {self.q: s_out.q.grad, self.qd: s_out.qd.grad, self.edge_d1: s_out.edge_d1_q.grad,
                 self.ref_twist: s_out.triplet_ref_twist_q.grad, self.strain: s_out.triplet_strain_q.grad}
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
                s.triplets.previous(state_in, self.strain_prev)
                wp.launch(predict_kernel, dim=n, inputs=[s.fixed, s_in.q, s_in.qd, h],
                          outputs=[self.q_pred, self._no_q], device=dev)
                tape.record_func(lambda: self._solve_adjoint(state_in, h, contacts), [self.q_theta])
                wp.launch(theta_kernel, dim=n, inputs=[s.fixed, s_in.q, s_in.qd, self.q_theta, s.theta, dt],
                          outputs=[self.q, self.qd], device=dev)
                wp.launch(advance_frames_kernel, dim=d.edge_length.shape[0],
                          inputs=[state_in.particle_q, self._q_nodes, d.edge_node0, d.edge_node1, s_in.edge_d1_q],
                          outputs=[self.edge_d1], device=dev)
                wp.launch(advance_ref_twist_kernel, dim=s.triplets.count,
                          inputs=[self.q, self.edge_d1, s.triplets.conn, s_in.triplet_ref_twist_q],
                          outputs=[self.ref_twist], device=dev)
                s.triplets.measure_arrays(self.q, self.edge_d1, self.ref_twist, self.strain)
            tape.backward(grads=seeds)

    def _solve_adjoint(self, state_in: State, h: float, contacts: ContactSnapshot | None) -> None:
        """``J^T lam = adj_q_theta`` (or the contact system), then ``-lam^T dr`` into the inputs."""
        s, tr = self.solver, self.solver.triplets
        s_in = state_in.dismech
        dev, n, nd = s.device, s.num_dofs, s.num_node_dofs

        # The Hessian at q_theta; the gradient goes to residual, scratch here.
        self.hessian.vals.zero_()
        tr.assemble(self.q_theta, state_in, self.strain_prev, self.residual, self.hessian, h)
        wp.launch(inertia_kernel, dim=n,
                  inputs=[self.q_theta, self.q_pred, s.mass, 1.0 / (h * h), s.fixed, s.model.gravity,
                          s._particle_f(state_in), nd],
                  outputs=[self.residual, self.hessian.indptr, self.hessian.vals], device=dev)
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
        """``J^T lam`` into ``q_theta.grad``."""
        self.q_theta.grad.zero_()
        self._residual(state_in, h, inputs_grad=False).backward(grads={self.residual: self._lam})

    def _residual(self, state_in: State, h: float, inputs_grad: bool) -> wp.Tape:
        """``r(q_theta)`` on a tape; without ``inputs_grad`` only ``q_theta`` takes adjoints."""
        s, tr = self.solver, self.solver.triplets
        s_in = state_in.dismech
        g = (lambda a: a) if inputs_grad else _without_grad
        self.residual.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(inertia_kernel, dim=s.num_dofs,
                      inputs=[self.q_theta, g(self.q_pred), g(self._mass), 1.0 / (h * h), s.fixed, s.model.gravity,
                              g(s._particle_f(state_in)), s.num_node_dofs],
                      outputs=[self.residual, self._no_indptr, self._no_vals], device=s.device)
            wp.launch(self._residual_kernel, dim=tr.count,
                      inputs=[self.q_theta, g(s_in.q), g(s_in.edge_d1_q), g(s_in.triplet_ref_twist_q), tr.conn,
                              g(tr.der.edge_length), g(tr.params), g(tr.rest), g(self.strain_prev), h,
                              s.num_node_dofs, s.fixed],
                      outputs=[self.residual], device=s.device)
        return tape


class _ContactSystem:
    """The augmented adjoint system ``M^T`` of a step with active contacts."""

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
        self._linear = sparse_solver(GeneralCSR(MT, dev), refactorize=False)
        self.rhs, self.w, self.res, self.step = (wp.zeros(n + 3 * m, dtype=float, device=dev) for _ in range(4))

    def solve(self, state_in: State, h: float) -> None:
        """``M^T w = [adj_q_theta, 0]``, refined; ``w_q`` into ``lam``."""
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
        """``-w^T dF/d input`` into ``q_in_grad`` and the friction adjoint."""
        a, s, c = self.adjoint, self.adjoint.solver, self.contacts
        empty = wp.zeros(0, dtype=float, device=s.device)
        wp.launch(contact_input_adjoint_kernel, dim=self.m,
                  inputs=[self.idx, c.pairs, c.bary, c.normal, s.fixed, c.rho, self.jac_mu, a._lam, self.w[self.n:]],
                  outputs=[q_in_grad if q_in_grad is not None else empty,
                           c.friction.grad if c.friction.grad is not None else empty],
                  device=s.device)


def _without_grad(a: wp.array | None) -> wp.array | None:
    """An alias of ``a`` without adjoints (``a`` must outlive it)."""
    if a is None or a.grad is None:
        return a
    return wp.array(ptr=a.ptr, dtype=a.dtype, shape=a.shape, strides=a.strides, device=a.device)


# -- kernels ------------------------------------------------------------------------------


@wp.kernel
def _q_theta_kernel(fixed: wp.array[wp.int32], q0: wp.array[float], q: wp.array[float], theta: float,
                    q_theta: wp.array[float]):
    """``q_theta = q0 + theta (q - q0)``."""
    i = wp.tid()
    if fixed[i] != 0:
        q_theta[i] = q0[i]
    else:
        q_theta[i] = q0[i] + theta * (q[i] - q0[i])


@wp.kernel
def _adjoint_rhs_kernel(fixed: wp.array[wp.int32], adj: wp.array[float], rhs: wp.array[float],
                        adj_copy: wp.array[float]):
    """``rhs = adj`` on free DOFs, ``adj_copy = adj``, ``adj = 0``."""
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
    n = wp.tid()
    particle_mass_grad[n] = particle_mass_grad[n] + adj_mass[3 * n] + adj_mass[3 * n + 1] + adj_mass[3 * n + 2]


@wp.kernel
def _fixed_adjoint_kernel(fixed: wp.array[wp.int32], adj_theta: wp.array[float], adj_coupling: wp.array[float],
                          adj_q0: wp.array[float]):
    """Fixed DOFs of ``q_theta`` are ``q0``: their adjoint and coupling go to ``q0``."""
    i = wp.tid()
    if fixed[i] != 0:
        adj_q0[i] = adj_q0[i] + adj_theta[i] + adj_coupling[i]
