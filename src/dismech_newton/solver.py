"""Implicit discrete elastic rod solver (Newton-Raphson, direct solve)."""

from collections.abc import Sequence
from typing import Any

import warp as wp
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase

from .builder import NAMESPACE, add_rod, fix_segment, register_custom_attributes
from .frames import advance_edge_frames, pose_edge_proxies
from .integrators import ImplicitEuler, IntegratorBase
from .linear_solvers import get_linear_solver
from .stencils import STENCILS, Stencil
from .system import SymmetricCSR, dof_constants, flatten_state


@wp.kernel
def _external_force_kernel(
    mass: wp.array[float], gravity: wp.array[wp.vec3], particle_f: wp.array[wp.vec3], residual: wp.array[float]
):
    """``residual -= m g + particle_f``; one thread per node DOF ``i = 3 * node + k`` (none on twist)."""
    i = wp.tid()
    node = i // 3
    k = i - 3 * node
    residual[i] = residual[i] - (mass[i] * gravity[0][k] + particle_f[node][k])


@wp.kernel
def _hold_fixed_kernel(fixed: wp.array[wp.int32], q_ref: wp.array[float], q: wp.array[float]):
    """Pin the fixed DOFs of the initial guess ``q`` to ``q_ref``."""
    i = wp.tid()
    if fixed[i] != 0:
        q[i] = q_ref[i]


@wp.kernel
def _constrain_kernel(
    fixed: wp.array[wp.int32],
    # outputs
    residual: wp.array[float],
    hess_indptr: wp.array[wp.int32],
    hess_vals: wp.array[Any],
):
    """Dirichlet rows: zero residual, unit diagonal. Stencils never write to fixed rows or
    columns, so only the diagonal needs setting."""
    i = wp.tid()
    if fixed[i] != 0:
        residual[i] = 0.0
        hess_vals[hess_indptr[i]] = type(hess_vals[0])(1.0)  # the diagonal leads each row


@wp.kernel
def _zero_fixed_kernel(fixed: wp.array[wp.int32], v: wp.array[float]):
    i = wp.tid()
    if fixed[i] != 0:
        v[i] = 0.0


@wp.kernel
def _apply_step_kernel(dx: wp.array[float], q: wp.array[float], track: int, err: wp.array[float]):
    """``q -= dx`` (``dx`` solves ``K dx = residual``); with ``track``, ``err = max |dx| / (1 + |q|)``."""
    i = wp.tid()
    qi = q[i]
    d = dx[i]
    q[i] = qi - d
    if track != 0:
        wp.atomic_max(err, 0, wp.abs(d) / (1.0 + wp.abs(qi)))


class DiSMechSolver(SolverBase):
    """Implicit discrete elastic rods, solved with Newton-Raphson.

    Each step solves ``inertia(q) + dE/dq(q) - f_ext = 0`` for node positions
    (``particle_q``) and edge twist angles (``dismech.edge_q``), iterating directly in
    ``state_out``; ``state_in`` is the start-of-step snapshot. The inertia term comes from the
    integrator (implicit Euler by default). Damping is strain-rate
    viscosity, ``sigma += kd * d eps / dt``. External force is ``m * model.gravity[0]`` plus
    ``state_in.particle_f`` on the nodes (clear it with ``state.clear_forces()``). Contacts and
    controls are ignored.

    States passed to :meth:`step` are flattened on first use (:func:`~dismech_newton.system.flatten_state`): they gain the flat DOF vectors
    ``state.dismech.q`` / ``qd`` and ``particle_q``, ``edge_q`` and the velocity arrays become
    views into them (see :mod:`dismech_newton.system`).

    Args:
        model: Model built with :meth:`add_rod`.
        integrator: Time integrator (:mod:`dismech_newton.integrators`); defaults to
            :class:`~dismech_newton.integrators.ImplicitEuler`.
        stencils: Elastic stencils (:mod:`dismech_newton.stencils`), instances of concrete
            stencil classes (which fix the energy); defaults to one of every defined class the
            model has rows of. Every class with rows in the model must be covered.
        newton_iterations: Maximum Newton-Raphson iterations per step.
        newton_tol: Stop when ``max |dq| / (1 + |q|) < newton_tol`` (absolute for small
            coordinates, relative for large ones, which keeps it reachable in float32);
            ``0`` always runs ``newton_iterations`` without host syncs (graph-capturable).
        hessian_dtype: ``wp.float64`` (default) or ``wp.float32`` for the Hessian values and
            the linear solve. Float32 halves the Hessian and the cuDSS factor and drops the
            conversion buffers; the assembly is float32 regardless.
        pose_proxies: Pose the capsule proxies (``state.body_q``) at the end of every step. Turn
            it off when nothing draws them and call :meth:`update_proxies` when needed.
    """

    register_custom_attributes = staticmethod(register_custom_attributes)
    add_rod = staticmethod(add_rod)
    fix_segment = staticmethod(fix_segment)

    def __init__(
        self,
        model: Model,
        *,
        integrator: IntegratorBase | None = None,
        stencils: Sequence[Stencil] | None = None,
        newton_iterations: int = 20,
        newton_tol: float = 1.0e-6,
        hessian_dtype=wp.float64,
        pose_proxies: bool = True,
    ):
        super().__init__(model)
        self.integrator = integrator if integrator is not None else ImplicitEuler()
        present = [cls for cls in STENCILS.values() if cls.rows(model) > 0]
        if stencils is None:
            self.stencils = [cls() for cls in present]
        else:
            self.stencils = list(stencils)
            missing = {cls.name for cls in present} - {s.name for s in self.stencils}
            if missing:
                raise ValueError(f"model has stencils {sorted(missing)} that `stencils` does not cover")
        self.newton_iterations = newton_iterations
        self.newton_tol = newton_tol
        self.pose_proxies = pose_proxies
        self.last_iterations = 0

        self.der = getattr(model, NAMESPACE)
        dev = self.device
        # Per-DOF constants of the flat vector (see :mod:`dismech_newton.system`).
        self.mass, self.fixed = dof_constants(model)
        self.num_dofs = self.mass.shape[0]
        self.num_node_dofs = 3 * model.particle_count
        self._no_force = wp.zeros(model.particle_count, dtype=wp.vec3, device=dev)  # for states without particle_f

        # Stencils measure their rest strains here: the initial configuration is unstressed.
        for stencil in self.stencils:
            stencil.bind(model, self.fixed)

        # Newton system: residual and the upper-triangle CSR Hessian (fixed pattern).
        self.hessian = SymmetricCSR(self.num_dofs, [s.dofs() for s in self.stencils], dev, hessian_dtype)
        self.residual = wp.zeros(self.num_dofs, dtype=float, device=dev)
        self.dx = wp.zeros(self.num_dofs, dtype=float, device=dev)
        self._err = wp.zeros(1, dtype=float, device=dev)
        self._has_proxies = bool((self.der.edge_body.numpy() >= 0).any())
        self._linear = get_linear_solver(self.hessian, dev)
        self.integrator.bind(self.mass)

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        dt = float(dt)
        flatten_state(state_in)
        flatten_state(state_out)
        self._begin_step(state_in, state_out, dt)
        self._newton_solve(state_in, state_out, dt)
        self._end_step(state_in, state_out, dt)

    def _begin_step(self, state_in: State, state_out: State, dt: float) -> None:
        """Start-of-step strains and the Newton initial guess (in ``state_out``)."""
        # Start-of-step strains, for rate-dependent energies.
        for stencil in self.stencils:
            stencil.begin_step(state_in)
        q_in, q = state_in.dismech.q, state_out.dismech.q
        self.integrator.begin_step(q_in, state_in.dismech.qd, dt, q)
        wp.launch(_hold_fixed_kernel, dim=self.num_dofs, inputs=[self.fixed, q_in], outputs=[q], device=self.device)

    def _newton_solve(self, state_in: State, state_out: State, dt: float) -> None:
        q = state_out.dismech.q
        track = int(self.newton_tol > 0.0)
        self.last_iterations = self.newton_iterations
        for it in range(self.newton_iterations):
            self._assemble(state_in, state_out, dt)
            self._linear.solve(self.residual, self.dx)
            self._err.zero_()
            wp.launch(
                _apply_step_kernel,
                dim=self.num_dofs,
                inputs=[self.dx, q, track, self._err],
                device=self.device,
            )
            if track and float(self._err.numpy()[0]) < self.newton_tol:  # a 4-byte read
                self.last_iterations = it + 1
                break

    def _assemble(self, state_in: State, state_out: State, dt: float) -> None:
        """Residual and CSR Hessian at ``state_out``: elastic terms, external force, inertia, Dirichlet."""
        self.residual.zero_()
        self.hessian.vals.zero_()
        for stencil in self.stencils:
            stencil.assemble(state_in, state_out, self.residual, self.hessian, dt)
        particle_f = state_in.particle_f if state_in.particle_f is not None else self._no_force
        wp.launch(
            _external_force_kernel,
            dim=self.num_node_dofs,
            inputs=[self.mass, self.model.gravity, particle_f],
            outputs=[self.residual],
            device=self.device,
        )
        self.integrator.assemble(state_out.dismech.q, self.residual, self.hessian)
        wp.launch(
            _constrain_kernel,
            dim=self.num_dofs,
            inputs=[self.fixed],
            outputs=[self.residual, self.hessian.indptr, self.hessian.vals],
            device=self.device,
        )

    def _end_step(self, state_in: State, state_out: State, dt: float) -> None:
        """Velocities, advanced reference frames, and (optionally) the kinematic proxies."""
        s_out = state_out.dismech
        self.integrator.end_step(s_out.q, s_out.qd)
        wp.launch(_zero_fixed_kernel, dim=self.num_dofs, inputs=[self.fixed], outputs=[s_out.qd], device=self.device)
        advance_edge_frames(self.der, state_in, state_out, self.device)
        for stencil in self.stencils:
            stencil.end_step(state_in, state_out)
        if self.pose_proxies:
            self.update_proxies(state_out)

    def update_proxies(self, state: State) -> None:
        """Pose the capsule proxies (``state.body_q``) from ``state``'s rod configuration."""
        if self._has_proxies and state.body_q is not None:
            pose_edge_proxies(self.der, state, self.device)
