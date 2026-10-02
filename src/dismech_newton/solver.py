"""Implicit discrete elastic rods: the theta method (implicit Euler to implicit midpoint), Newton-Raphson, cuDSS."""

import warp as wp
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase

from .adjoint import StepAdjoint, suspended_tape
from .builder import add_rod, fix_segment, register_custom_attributes
from .contact import ContactSnapshot
from .frames import (
    advance_frames_kernel,
    dof_constants,
    external_force,
    flatten_state,
    pose_proxies_kernel,
    proxy_joints,
)
from .linear import CudssSolver, SymmetricCSR
from .triplet import Triplets, linear_energy


class DiSMechSolver(SolverBase):
    """Implicit discrete elastic rods, solved with Newton-Raphson.

    Each step solves ``M (q - q_n - dt v_n) / dt^2 + dE/dq(q) - f_ext = 0`` for the node positions
    and edge twist angles, iterating in ``state_out``. More generally it solves that for a step
    ``theta dt``, which gives ``q_theta = (1 - theta) q_n + theta q_{n+1}`` of the (one-leg) theta method,
    and extrapolates to ``q_{n+1}``. ``theta = 1`` is implicit Euler (first order, damps every mode);
    ``theta = 1/2`` is implicit midpoint (second order, symplectic, no numerical damping), and a little
    above 1/2 damps mostly the stiff, high-frequency modes. External force is ``m * model.gravity`` plus
    ``state_in.particle_f``. Contacts and controls are ignored (see the ADMM subclass). Fixed DOFs
    (:func:`fix_segment`) are prescribed by writing ``state_in.dismech.q``. States are flattened on
    first use. Subclasses replace :meth:`_build_system` and :meth:`_solve`.

    Args:
        model: Model built with :meth:`add_rod`; must be on a CUDA device.
        energy: Triplet energy (see :mod:`dismech_newton.triplet`).
        newton_iterations: Maximum Newton-Raphson iterations per step.
        newton_tol: Stop when ``max |dq| / (1 + |q|) < newton_tol``; ``0`` always runs
            ``newton_iterations``. The test runs on the device (``wp.capture_while``).
        pose_proxies: Pose the capsule proxies (and set their velocities) at the end of every step;
            otherwise call :meth:`update_proxies` when needed.
        theta: Where in the step the forces are evaluated, in ``[1/2, 1]``: ``1`` implicit Euler,
            ``1/2`` implicit midpoint.
    """

    register_custom_attributes = staticmethod(register_custom_attributes)
    add_rod = staticmethod(add_rod)
    fix_segment = staticmethod(fix_segment)

    def __init__(
        self, model: Model, *, energy=linear_energy, newton_iterations: int = 20, newton_tol: float = 1.0e-6,
        pose_proxies: bool = True, theta: float = 1.0,
    ):
        super().__init__(model)
        if not 0.5 <= theta <= 1.0:
            raise ValueError(f"theta must be in [1/2, 1], got {theta}")
        self.theta = float(theta)
        if not wp.get_device(model.device).is_cuda:
            raise ValueError(f"{type(self).__name__} runs on CUDA")
        self.newton_iterations = newton_iterations
        self.newton_tol = newton_tol
        self.pose_proxies = pose_proxies
        self._count = wp.zeros(1, dtype=wp.int32, device=self.device)  # iterations of the last step
        self._go = wp.zeros(1, dtype=wp.int32, device=self.device)  # loop condition

        self.der = model.dismech
        self.mass, self.fixed = dof_constants(model)
        self.num_dofs = self.mass.shape[0]
        self.num_node_dofs = 3 * model.particle_count
        self.triplets = Triplets(model, self.fixed, energy)
        self.q_pred = wp.zeros(self.num_dofs, dtype=float, device=self.device)
        self.alpha = 0.0  # inertia weight 1 / dt^2
        self._no_force = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self._no_velocity = wp.zeros(0, dtype=wp.spatial_vector, device=self.device)
        self._has_proxies = bool((self.der.edge_body.numpy() >= 0).any())
        self._edge_joint = wp.array(proxy_joints(model), dtype=wp.vec2i, device=self.device)
        self._no_joint = wp.zeros(0, dtype=float, device=self.device)
        self._adjoint = None  # StepAdjoint, built on the first vjp
        self._build_system()

    def _build_system(self) -> None:
        """Residual, upper-triangle CSR Hessian and its solver."""
        self.hessian = SymmetricCSR(self.num_dofs, self.triplets.dofs(), self.device)
        self.residual = wp.zeros(self.num_dofs, dtype=float, device=self.device)
        self.dx = wp.zeros(self.num_dofs, dtype=float, device=self.device)
        self._err = wp.zeros(1, dtype=float, device=self.device)
        self._linear = CudssSolver(self.hessian)

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float):
        """Advance ``state_in`` by ``dt`` into ``state_out``.

        Under an active ``wp.Tape`` the iterations are not recorded: the step goes on the tape as
        its implicit-function adjoint (:meth:`vjp`), when ``state_out`` has gradients.
        """
        dt = float(dt)
        with suspended_tape() as tape:
            self._step(state_in, state_out, contacts, dt)
            if tape is not None:
                self._record(tape, state_in, state_out, dt)

    def refresh_mass(self) -> None:
        """Re-read ``model.particle_mass`` and ``model.dismech.edge_inertia`` after they changed."""
        self.mass.assign(dof_constants(self.model)[0])

    def vjp(self, state_in: State, state_out: State, dt: float, contacts: ContactSnapshot | None = None) -> None:
        """Backpropagate the step ``state_in -> state_out`` (:mod:`~dismech_newton.adjoint`) from
        ``state_out``'s ``.grad`` arrays; ``contacts`` from :meth:`contact_snapshot`, right after the step."""
        if self._adjoint is None:
            self._adjoint = StepAdjoint(self)
        self._adjoint.vjp(state_in, state_out, float(dt), contacts)

    def contact_snapshot(self) -> ContactSnapshot | None:
        """The last step's contacts, for :meth:`vjp`; this solver has none."""
        return None

    def _record(self, tape: wp.Tape, state_in: State, state_out: State, dt: float):
        s_in, s_out = state_in.dismech, state_out.dismech
        outputs = (s_out.q, s_out.qd, s_out.edge_d1_q, s_out.triplet_ref_twist_q)
        if not any(a.requires_grad for a in outputs):
            return
        contacts = self.contact_snapshot()
        inputs = (s_in.q, s_in.qd, s_in.edge_d1_q, s_in.triplet_ref_twist_q, state_in.particle_f, self.triplets.params,
                  self.triplets.rest, self.der.edge_length, self.model.particle_mass, self.der.edge_inertia,
                  contacts and contacts.friction)
        arrays = [a for a in (*inputs, *outputs) if a is not None and a.requires_grad]
        tape.record_func(lambda: self.vjp(state_in, state_out, dt, contacts), arrays)

    def _step(self, state_in: State, state_out: State, contacts: Contacts | None, dt: float):
        flatten_state(state_in)
        flatten_state(state_out)
        q_in, q = state_in.dismech.q, state_out.dismech.q
        h = self.theta * dt  # the implicit Euler step solved
        self.alpha = 1.0 / (h * h)
        self.triplets.begin_step(state_in)
        wp.launch(_predict_kernel, dim=self.num_dofs, inputs=[self.fixed, q_in, state_in.dismech.qd, h],
                  outputs=[self.q_pred, q], device=self.device)

        self._solve(state_in, state_out, contacts, h)

        wp.launch(_theta_kernel, dim=self.num_dofs, inputs=[self.fixed, q_in, state_in.dismech.qd, self.theta, dt],
                  outputs=[q, state_out.dismech.qd], device=self.device)
        wp.launch(
            advance_frames_kernel,
            dim=self.der.edge_length.shape[0],
            inputs=[state_in.particle_q, state_out.particle_q, self.der.edge_node0, self.der.edge_node1,
                    state_in.dismech.edge_d1_q],
            outputs=[state_out.dismech.edge_d1_q],
            device=self.device,
        )
        self.triplets.end_step(state_in, state_out)
        if self.pose_proxies:
            self.update_proxies(state_out)

    def _solve(self, state_in: State, state_out: State, contacts: Contacts | None, dt: float) -> None:
        """Newton-Raphson on ``state_out.dismech.q``, starting from the initial guess."""
        self._iterate(self._newton_iteration, self.newton_iterations, 1, self.newton_tol, self._err,
                      state_in=state_in, state_out=state_out, dt=dt)

    def _newton_iteration(self, state_in: State, state_out: State, dt: float) -> None:
        q = state_out.dismech.q
        self.residual.zero_()
        self.hessian.vals.zero_()
        self.triplets.assemble(state_in, state_out, self.residual, self.hessian, dt)
        wp.launch(
            _inertia_kernel,
            dim=self.num_dofs,
            inputs=[q, self.q_pred, self.mass, self.alpha, self.fixed, self.model.gravity,
                    self._particle_f(state_in), self.num_node_dofs],
            outputs=[self.residual, self.hessian.indptr, self.hessian.vals],
            device=self.device,
        )
        self._linear.solve(self.residual, self.dx)
        self._err.zero_()
        wp.launch(_apply_step_kernel, dim=self.num_dofs, inputs=[self.dx, q, int(self.newton_tol > 0.0), self._err],
                  device=self.device)

    def _iterate(self, block, max_iterations: int, block_size: int, tol: float, err: wp.array, **kwargs) -> None:
        """Run ``block(**kwargs)`` (``block_size`` iterations) until ``err < tol`` or ``max_iterations``,
        testing on the device, so the loop can be graph-captured; ``tol = 0`` runs a fixed count."""
        if tol <= 0.0:
            for _ in range(-(-max_iterations // block_size)):
                block(**kwargs)
            self._count.fill_(max_iterations)
            return

        def body(**kw):
            block(**kw)
            wp.launch(_stop_kernel, dim=1, inputs=[err, tol, block_size, max_iterations],
                      outputs=[self._count, self._go], device=self.device)

        self._count.zero_()
        self._go.fill_(1)
        wp.capture_while(self._go, body, **kwargs)

    @property
    def last_iterations(self) -> int:
        """Iterations of the last step (a device read)."""
        return int(self._count.numpy()[0])

    def _particle_f(self, state: State) -> wp.array:
        return state.particle_f if state.particle_f is not None else self._no_force

    def update_proxies(self, state: State) -> None:
        """Pose the capsule proxies and set their velocities, in ``body_q``/``body_qd`` and their free joints."""
        if not self._has_proxies or state.body_q is None:
            return
        body_qd = state.body_qd if state.body_qd is not None else self._no_velocity
        joints = state.joint_q is not None and state.joint_qd is not None and state.joint_q.shape[0] > 0
        d = self.der
        wp.launch(
            pose_proxies_kernel,
            dim=d.edge_length.shape[0],
            inputs=[state.particle_q, state.particle_qd, state.dismech.edge_q, state.dismech.edge_d1_q, d.edge_node0,
                    d.edge_node1, d.edge_body, self._edge_joint, int(state.body_qd is not None)],
            outputs=[state.body_q, body_qd, state.joint_q if joints else self._no_joint,
                     state.joint_qd if joints else self._no_joint],
            device=self.device,
        )


# -- kernels ------------------------------------------------------------------------------


@wp.kernel
def _predict_kernel(
    fixed: wp.array[wp.int32], q0: wp.array[float], v0: wp.array[float], dt: float,
    q_pred: wp.array[float], q: wp.array[float],
):
    """``q_pred = q0 + dt v0``, the initial guess; fixed DOFs keep their prescribed ``q0``."""
    i = wp.tid()
    q_pred[i] = q0[i] + dt * v0[i]
    if fixed[i] != 0:
        q[i] = q0[i]
    else:
        q[i] = q_pred[i]


@wp.kernel
def _theta_kernel(fixed: wp.array[wp.int32], q0: wp.array[float], v0: wp.array[float], theta: float, dt: float,
                  q: wp.array[float], v: wp.array[float]):
    """From ``q_theta`` (an implicit Euler step ``h = theta dt``): ``q = q0 + (q_theta - q0) / theta`` and
    ``v = v0 + (q_theta - q0 - h v0) / (theta^2 dt)``; ``theta = 1`` is implicit Euler itself."""
    i = wp.tid()
    if fixed[i] != 0:
        v[i] = 0.0
        return
    d = q[i] - q0[i]
    q[i] = q0[i] + d / theta
    v[i] = v0[i] + (d - theta * dt * v0[i]) / (theta * theta * dt)


@wp.kernel
def _inertia_kernel(
    q: wp.array[float], q_pred: wp.array[float], mass: wp.array[float], alpha: float, fixed: wp.array[wp.int32],
    gravity: wp.array[wp.vec3], particle_f: wp.array[wp.vec3], num_node_dofs: int,
    # outputs
    residual: wp.array[float], hess_indptr: wp.array[wp.int32], hess_vals: wp.array[wp.float64],
):
    """Inertia ``M alpha (q - q_pred)`` minus external force; Dirichlet rows get a zero residual and a unit diagonal."""
    i = wp.tid()
    slot = hess_indptr[i]  # the diagonal leads each row
    if fixed[i] != 0:
        residual[i] = 0.0
        hess_vals[slot] = wp.float64(1.0)
        return
    r = residual[i]
    if i < num_node_dofs:
        r = r - external_force(i, mass, gravity, particle_f)
    k = mass[i] * alpha
    residual[i] = r + k * (q[i] - q_pred[i])
    hess_vals[slot] = hess_vals[slot] + wp.float64(k)


@wp.kernel
def _apply_step_kernel(dx: wp.array[float], q: wp.array[float], track: int, err: wp.array[float]):
    """``q -= dx``; with ``track``, ``err = max |dx| / (1 + |q|)``."""
    i = wp.tid()
    qi = q[i]
    d = dx[i]
    q[i] = qi - d
    if track != 0:
        wp.atomic_max(err, 0, wp.abs(d) / (1.0 + wp.abs(qi)))


@wp.kernel
def _stop_kernel(err: wp.array[float], tol: float, block: int, max_iterations: int,
                 count: wp.array[wp.int32], go: wp.array[wp.int32]):
    """After ``block`` more iterations: stop once every ``err`` is below ``tol`` or at ``max_iterations``."""
    n = count[0] + block
    count[0] = n
    converged = int(1)
    for i in range(err.shape[0]):
        if err[i] >= tol:
            converged = 0
    if converged != 0 or n >= max_iterations:
        go[0] = 0
