"""Gradients of a step by the implicit function theorem, not by backpropagating through the iterations.

A step solves, on its free DOFs, for ``q = q_theta`` (the fixed ones are prescribed)

    r(q; inputs) = M alpha (q - q_pred) - f_ext + dE/dq = 0.

Newton-Raphson and ADMM converge to the same root (the ADMM penalties drop out at its fixed point),
so the adjoint of the solve is one linear solve with the residual's Jacobian at the solution,

    J^T lam = adj_q,      J = dr/dq = M alpha + d^2 E / dq^2 + (asymmetric part),

and the adjoint of every input is ``-lam^T dr/d input``, by Warp's autodiff of the residual kernels.
``J`` is not quite the symmetric Newton Hessian ``A``: DER's twist gradient in the node positions,
``kb / 2|e|``, is exact only while the reference frame is current, so the node rows of ``r`` are
not exactly the gradient of the energy its twist rows differentiate. ``A`` is factorised and the
solve is refined against the exact ``J^T lam`` (autodiff), ``lam += A^{-1} (adj_q - J^T lam)``,
which contracts by about ``1e-4`` per iteration.
The rest of the step (prediction, start-of-step strains, theta extrapolation, frame transport) is
ordinary kernels, differentiated by Warp as well. The gradient is that of the exact solution,
evaluated at the solver's iterate: it is as accurate as the solver's tolerance.

:meth:`StepAdjoint.vjp` is framework-neutral: it reads the adjoints of a step's outputs from the
``.grad`` arrays of ``state_out`` and accumulates the adjoints of its inputs into the ``.grad``
arrays of ``state_in`` and of ``model.dismech.triplet_params``. :meth:`DiSMechSolver.step` records
it on an active ``wp.Tape``; a PyTorch or JAX binding calls it from its own backward.

Inputs with gradients: ``state_in`` ``q``, ``qd``, ``edge_d1_q``, ``triplet_ref_twist_q`` and
``particle_f``, and the triplet parameters. Outputs: ``state_out`` ``q``, ``qd``, ``edge_d1_q`` and
``triplet_ref_twist_q`` (``particle_q`` and ``edge_q`` are views of ``q``). Contacts are not
differentiated; rest strains, masses and gravity are constants; the proxy poses carry no gradient.
"""

from contextlib import contextmanager

import warp as wp
from newton import State

from .frames import advance_frames_kernel, external_force
from .linear import CudssSolver, SymmetricCSR
from .strains import vec5f
from .triplet import advance_ref_twist_kernel, make_residual_kernel


@contextmanager
def suspended_tape():
    """Stop recording on the active ``wp.Tape`` (if any) inside the block; yields that tape."""
    runtime = wp._src.context.runtime  # as warp.fem does: kernels that must not be recorded
    tape, runtime.tape = runtime.tape, None
    try:
        yield tape
    finally:
        runtime.tape = tape


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
    """``rhs = adj`` on free DOFs (``0`` on fixed ones), ``adj_copy = adj``."""
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


def _without_grad(a: wp.array | None) -> wp.array | None:
    """An alias of ``a`` that autodiff does not write adjoints into (``a`` must outlive it)."""
    if a is None or a.grad is None:
        return a
    return wp.array(ptr=a.ptr, dtype=a.dtype, shape=a.shape, strides=a.strides, device=a.device)


@wp.kernel
def _fixed_adjoint_kernel(fixed: wp.array[wp.int32], adj_theta: wp.array[float], adj_coupling: wp.array[float],
                          adj_q0: wp.array[float]):
    """A fixed DOF of ``q_theta`` is the prescribed ``q0``: its adjoint, and its coupling into the
    free DOFs' residual, go to ``q0``."""
    i = wp.tid()
    if fixed[i] != 0:
        adj_q0[i] = adj_q0[i] + adj_theta[i] + adj_coupling[i]


class StepAdjoint:
    """The adjoint of :meth:`DiSMechSolver.step` (see the module docstring); built by the solver on first use.

    Attributes:
        refine: Refinement iterations of the adjoint solve against the exact ``J^T``.
    """

    refine = 2

    def __init__(self, solver):
        self.solver = solver
        tr = solver.triplets
        dev = solver.device
        n, nd = solver.num_dofs, solver.num_node_dofs
        self.hessian = SymmetricCSR(n, tr.dofs(), dev)
        self._linear = CudssSolver(self.hessian, refactorize=False)  # once per step, see invalidate
        self._residual_kernel = make_residual_kernel(tr.energy)

        def grad_array(count, dtype=float):
            return wp.zeros(count, dtype=dtype, device=dev, requires_grad=True)

        # The step recomputed out of place, every intermediate with its adjoint.
        self.q_theta = grad_array(n)
        self.q_pred = grad_array(n)
        self.q = grad_array(n)
        self.qd = grad_array(n)
        self.strain_prev = grad_array(tr.count, vec5f)
        self.edge_d1 = grad_array(solver.der.edge_length.shape[0], wp.vec3)
        self.ref_twist = grad_array(tr.count)
        self.residual = grad_array(n)
        self._intermediates = (self.q_theta, self.q_pred, self.q, self.qd, self.strain_prev, self.edge_d1,
                               self.ref_twist, self.residual)
        self._q_theta_nodes = self.q_theta[:nd].reshape((nd // 3, 3)).view(wp.vec3)
        self._q_theta_edges = self.q_theta[nd:]
        self._q_nodes = self.q[:nd].reshape((nd // 3, 3)).view(wp.vec3)
        # Linear solve.
        self._rhs = wp.zeros(n, dtype=float, device=dev)
        self._lam = wp.zeros(n, dtype=float, device=dev)
        self._neg_lam = wp.zeros(n, dtype=float, device=dev)
        self._res = wp.zeros(n, dtype=float, device=dev)
        self._step = wp.zeros(n, dtype=float, device=dev)
        self._adj_theta = wp.zeros(n, dtype=float, device=dev)
        self._gradient = wp.zeros(n, dtype=float, device=dev)  # assembly's residual, unused

    def vjp(self, state_in: State, state_out: State, dt: float) -> None:
        """Accumulate the adjoints of the step ``state_in -> state_out`` (of size ``dt``) into
        ``state_in``'s ``.grad`` arrays and the triplet parameters', from ``state_out``'s.

        Recomputes what it needs from the two states, so they must still hold that step.
        """
        s = self.solver
        s_in, s_out = state_in.dismech, state_out.dismech
        seeds = {self.q: s_out.q.grad, self.qd: s_out.qd.grad, self.edge_d1: s_out.edge_d1_q.grad,
                 self.ref_twist: s_out.triplet_ref_twist_q.grad}
        seeds = {a: g for a, g in seeds.items() if g is not None}
        if not seeds:
            return
        dev = s.device
        n = s.num_dofs
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
                tape.record_func(lambda: self._solve_adjoint(state_in, h), [self.q_theta])
                wp.launch(_theta_kernel, dim=n, inputs=[s.fixed, s_in.q, s_in.qd, self.q_theta, s.theta, dt],
                          outputs=[self.q, self.qd], device=dev)
                d = s.der
                wp.launch(advance_frames_kernel, dim=d.edge_length.shape[0],
                          inputs=[state_in.particle_q, self._q_nodes, d.edge_node0, d.edge_node1, s_in.edge_d1_q],
                          outputs=[self.edge_d1], device=dev)
                wp.launch(advance_ref_twist_kernel, dim=s.triplets.count,
                          inputs=[self._q_nodes, self.edge_d1, s.triplets.conn, s_in.triplet_ref_twist_q],
                          outputs=[self.ref_twist], device=dev)
            tape.backward(grads=seeds)

    def _solve_adjoint(self, state_in: State, h: float) -> None:
        """``J^T lam = adj_q_theta`` (``A`` factorised, refined), then ``-lam^T dr/d inputs`` into the inputs' adjoints."""
        s, tr = self.solver, self.solver.triplets
        s_in = state_in.dismech
        dev, n = s.device, s.num_dofs
        alpha = 1.0 / (h * h)

        self.hessian.vals.zero_()
        tr.assemble_at(self._q_theta_nodes, self._q_theta_edges, state_in.particle_q, s_in.edge_d1_q,
                       s_in.triplet_ref_twist_q, self.strain_prev, self._gradient, self.hessian, h)
        wp.launch(_inertia_hessian_kernel, dim=n, inputs=[s.mass, alpha, s.fixed],
                  outputs=[self.hessian.indptr, self.hessian.vals], device=dev)
        self._linear.invalidate()
        wp.launch(_adjoint_rhs_kernel, dim=n, inputs=[s.fixed], outputs=[self.q_theta.grad, self._rhs,
                  self._adj_theta], device=dev)
        self._linear.solve(self._rhs, self._lam)
        # Refine against the exact J^T lam; only q_theta takes adjoints here.
        for _ in range(self.refine):
            self.q_theta.grad.zero_()
            self._residual(state_in, h, inputs_grad=False).backward(grads={self.residual: self._lam})
            wp.launch(_refine_residual_kernel, dim=n, inputs=[s.fixed, self._rhs, self.q_theta.grad],
                      outputs=[self._res], device=dev)
            self._linear.solve(self._res, self._step)
            wp.launch(_add_kernel, dim=n, inputs=[self._step], outputs=[self._lam], device=dev)
        wp.launch(_negate_kernel, dim=n, inputs=[self._lam], outputs=[self._neg_lam], device=dev)

        # -lam^T dr into the inputs. The adjoint it leaves in q_theta is the coupling of the fixed
        # DOFs (the free DOFs' part is J^T lam, already accounted for).
        self.q_theta.grad.zero_()
        self._residual(state_in, h, inputs_grad=True).backward(grads={self.residual: self._neg_lam})
        if s_in.q.grad is not None:
            wp.launch(_fixed_adjoint_kernel, dim=n, inputs=[s.fixed, self._adj_theta, self.q_theta.grad],
                      outputs=[s_in.q.grad], device=dev)

    def _residual(self, state_in: State, h: float, inputs_grad: bool) -> wp.Tape:
        """``r(q_theta)`` into :attr:`residual`, on a tape; without ``inputs_grad`` only
        ``q_theta`` takes adjoints."""
        s, tr = self.solver, self.solver.triplets
        s_in = state_in.dismech
        g = (lambda a: a) if inputs_grad else _without_grad
        self.residual.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(_inertia_residual_kernel, dim=s.num_dofs,
                      inputs=[self.q_theta, g(self.q_pred), s.mass, 1.0 / (h * h), s.fixed, s.model.gravity,
                              g(s._particle_f(state_in)), s.num_node_dofs],
                      outputs=[self.residual], device=s.device)
            wp.launch(self._residual_kernel, dim=tr.count,
                      inputs=[self.q_theta, g(s_in.q), g(s_in.edge_d1_q), g(s_in.triplet_ref_twist_q), tr.conn,
                              tr.der.edge_length, g(tr.params), tr.rest, g(self.strain_prev), h, s.num_node_dofs,
                              s.fixed],
                      outputs=[self.residual], device=s.device)
        return tape
