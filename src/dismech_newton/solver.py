"""Implicit discrete elastic rods: the theta method, Newton-Raphson, a sparse direct solve."""

from contextlib import contextmanager

import numpy as np
import warp as wp
from newton import Contacts, Control, JointType, Model, State
from newton.solvers import SolverBase

from .builder import add_rod, fix_segment, register_custom_attributes
from .contact import ContactSnapshot
from .dofs import dof_constants, external_force, flatten_state
from .linear import sparse_solver
from .sparse import SymmetricCSR
from .strains import material_frame, parallel_transport
from .triplet import Triplets, linear_energy


class DiSMechSolver(SolverBase):
    """Implicit discrete elastic rods, solved with Newton-Raphson.

    Fixed DOFs are prescribed through ``state_in.dismech.q``; contacts and controls are ignored. The linear
    solves use cuDSS on CUDA (the 'gpu' extra) and SciPy on the CPU; :meth:`step` is graph-capturable with
    ``newton_tol=0`` (cuDSS cannot run inside the device-side loop a tolerance needs).

    Args:
        newton_iterations: Maximum Newton iterations per step.
        newton_tol: Stop at ``max |dq| / (1 + |q|) < newton_tol``; ``0`` runs ``newton_iterations`` (capturable).
        pose_proxies: Pose the capsule proxies every step (else call :meth:`update_proxies`).
        theta: In ``[1/2, 1]``: ``1`` implicit Euler, ``1/2`` implicit midpoint.
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
        """Residual, CSR Hessian and its solver."""
        self.hessian = SymmetricCSR(self.num_dofs, self.triplets.dofs(), self.device)
        self.residual = wp.zeros(self.num_dofs, dtype=float, device=self.device)
        self.dx = wp.zeros(self.num_dofs, dtype=float, device=self.device)
        self._err = wp.zeros(1, dtype=float, device=self.device)
        self._linear = sparse_solver(self.hessian)

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float):
        """Advance ``state_in`` by ``dt``; under a ``wp.Tape`` the step is recorded as its adjoint (:meth:`vjp`)."""
        dt = float(dt)
        with suspended_tape() as tape:
            self._step(state_in, state_out, contacts, dt)
            if tape is not None:
                self._record(tape, state_in, state_out, dt)

    def refresh_mass(self) -> None:
        """Re-read the masses and twist inertias after they changed."""
        self.mass.assign(dof_constants(self.model)[0])

    def vjp(self, state_in: State, state_out: State, dt: float, contacts: ContactSnapshot | None = None) -> None:
        """Backpropagate the step from ``state_out``'s ``.grad``; ``contacts`` from :meth:`contact_snapshot`.

        Gradients reach ``state_in`` and, where ``requires_grad``, ``triplet_params``, ``triplets.rest``,
        ``edge_length``, ``particle_mass``, ``edge_inertia`` and ``contact.friction``."""
        if self._adjoint is None:
            from .adjoint import StepAdjoint  # imports this module's step kernels

            self._adjoint = StepAdjoint(self)
        self._adjoint.vjp(state_in, state_out, float(dt), contacts)

    def contact_snapshot(self) -> ContactSnapshot | None:
        """The last step's contacts for :meth:`vjp` (none here)."""
        return None

    def _record(self, tape: wp.Tape, state_in: State, state_out: State, dt: float):
        s_in, s_out = state_in.dismech, state_out.dismech
        outputs = (s_out.q, s_out.qd, s_out.edge_d1_q, s_out.triplet_ref_twist_q, s_out.triplet_strain_q)
        if not any(a.requires_grad for a in outputs):
            return
        contacts = self.contact_snapshot()
        inputs = (s_in.q, s_in.qd, s_in.edge_d1_q, s_in.triplet_ref_twist_q, s_in.triplet_strain_q,
                  state_in.particle_f, self.triplets.params,
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
        wp.launch(predict_kernel, dim=self.num_dofs, inputs=[self.fixed, q_in, state_in.dismech.qd, h],
                  outputs=[self.q_pred, q], device=self.device)

        self._solve(state_in, state_out, contacts, h)

        wp.launch(theta_kernel, dim=self.num_dofs, inputs=[self.fixed, q_in, state_in.dismech.qd, q, self.theta, dt],
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
        """Newton-Raphson on ``state_out.dismech.q``."""
        self._iterate(self._newton_iteration, self.newton_iterations, 1, self.newton_tol, self._err,
                      state_in=state_in, state_out=state_out, dt=dt)

    def _newton_iteration(self, state_in: State, state_out: State, dt: float) -> None:
        q = state_out.dismech.q
        self.residual.zero_()
        self.hessian.vals.zero_()
        self.triplets.assemble(q, state_in, self.triplets.strain_prev, self.residual, self.hessian, dt)
        wp.launch(
            inertia_kernel,
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
        """Run ``block`` until ``err < tol`` or ``max_iterations``, tested on the device (graph-capturable)."""
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
    def graph_capturable(self) -> bool:
        """Whether :meth:`step` can be captured in a CUDA graph (decided by the first step for ADMM)."""
        linear = self._linear
        if not self.device.is_cuda or linear is None:
            return self.device.is_cuda
        return linear.graph_capturable and (not self._device_loop or linear.loop_capturable)

    @property
    def _device_loop(self) -> bool:
        """Whether the iterations stop on a tolerance (a device-side loop)."""
        return self.newton_tol > 0.0

    @property
    def last_iterations(self) -> int:
        """Iterations of the last step (a device read)."""
        return int(self._count.numpy()[0])

    def _particle_f(self, state: State) -> wp.array:
        return state.particle_f if state.particle_f is not None else self._no_force

    def update_proxies(self, state: State) -> None:
        """Pose the capsule proxies and set their velocities (and their free joints)."""
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


def active_tape() -> wp.Tape | None:
    """The ``wp.Tape`` recording now, if any (a Warp private, as warp.fem reads it)."""
    return wp._src.context.runtime.tape


@contextmanager
def suspended_tape():
    """Pause the active ``wp.Tape`` (if any) inside the block; yields it."""
    runtime = wp._src.context.runtime
    tape, runtime.tape = runtime.tape, None
    try:
        yield tape
    finally:
        runtime.tape = tape


def proxy_joints(model: Model) -> np.ndarray:
    """Per edge, the ``(joint_q, joint_qd)`` starts of its proxy's free root joint, ``-1`` without."""
    out = np.full((model.dismech.edge_body.shape[0], 2), -1, dtype=np.int32)
    if not model.joint_count:
        return out
    free = (model.joint_type.numpy() == int(JointType.FREE)) & (model.joint_parent.numpy() < 0)
    joint_of = np.full(max(model.body_count, 1), -1)
    joint_of[model.joint_child.numpy()[free]] = np.nonzero(free)[0]
    edge_body = model.dismech.edge_body.numpy()
    j = np.where(edge_body >= 0, joint_of[np.maximum(edge_body, 0)], -1)
    has = j >= 0
    out[has, 0] = model.joint_q_start.numpy()[j[has]]
    out[has, 1] = model.joint_qd_start.numpy()[j[has]]
    return out


# -- kernels ------------------------------------------------------------------------------


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
    """Stop once every ``err < tol`` or at ``max_iterations``."""
    n = count[0] + block
    count[0] = n
    converged = int(1)
    for i in range(err.shape[0]):
        if err[i] >= tol:
            converged = 0
    if converged != 0 or n >= max_iterations:
        go[0] = 0


# -- theta-method step kernels (one thread per DOF), shared with the adjoint ---------------


@wp.kernel
def predict_kernel(
    fixed: wp.array[wp.int32], q0: wp.array[float], v0: wp.array[float], h: float,
    # outputs
    q_pred: wp.array[float], q: wp.array[float],
):
    """``q_pred = q0 + h v0``; given ``q``, also the initial guess (fixed DOFs keep ``q0``)."""
    i = wp.tid()
    q_pred[i] = q0[i] + h * v0[i]
    if q.shape[0] > 0:
        if fixed[i] != 0:
            q[i] = q0[i]
        else:
            q[i] = q_pred[i]


@wp.kernel
def theta_kernel(
    fixed: wp.array[wp.int32], q0: wp.array[float], v0: wp.array[float], q_theta: wp.array[float], theta: float,
    dt: float,
    # outputs
    q: wp.array[float], v: wp.array[float],
):
    """Extrapolate the implicit Euler step ``h = theta dt`` to the step end; ``q`` may alias ``q_theta``."""
    i = wp.tid()
    if fixed[i] != 0:
        q[i] = q_theta[i]
        v[i] = 0.0
        return
    d = q_theta[i] - q0[i]
    q[i] = q0[i] + d / theta
    v[i] = v0[i] + (d - theta * dt * v0[i]) / (theta * theta * dt)


@wp.kernel
def inertia_kernel(
    q: wp.array[float], q_pred: wp.array[float], mass: wp.array[float], alpha: float, fixed: wp.array[wp.int32],
    gravity: wp.array[wp.vec3], particle_f: wp.array[wp.vec3], num_node_dofs: int,
    # outputs
    residual: wp.array[float], hess_indptr: wp.array[wp.int32], hess_vals: wp.array[wp.float64],
):
    """``residual += M alpha (q - q_pred) - f_ext`` on free DOFs; given a Hessian, its leading diagonals."""
    i = wp.tid()
    if hess_vals.shape[0] > 0:
        slot = hess_indptr[i]
        if fixed[i] != 0:
            hess_vals[slot] = wp.float64(1.0)
        else:
            hess_vals[slot] = hess_vals[slot] + wp.float64(mass[i] * alpha)
    if fixed[i] != 0:
        return
    r = mass[i] * alpha * (q[i] - q_pred[i])
    if i < num_node_dofs:
        r = r - external_force(i, mass, gravity, particle_f)
    wp.atomic_add(residual, i, r)


# -- end-of-step kernels (one thread per edge) --------------------------------------------


@wp.kernel
def advance_frames_kernel(
    node_q_prev: wp.array[wp.vec3], node_q: wp.array[wp.vec3], edge_node0: wp.array[wp.int32],
    edge_node1: wp.array[wp.int32], edge_d1_prev: wp.array[wp.vec3],
    # outputs
    edge_d1: wp.array[wp.vec3],
):
    """Parallel-transport every ``d1`` to the current tangent."""
    e = wp.tid()
    n0 = edge_node0[e]
    n1 = edge_node1[e]
    t_prev = wp.normalize(node_q_prev[n1] - node_q_prev[n0])
    edge_d1[e] = parallel_transport(edge_d1_prev[e], t_prev, wp.normalize(node_q[n1] - node_q[n0]))


@wp.kernel
def pose_proxies_kernel(
    node_q: wp.array[wp.vec3], node_qd: wp.array[wp.vec3], edge_q: wp.array[float], edge_d1: wp.array[wp.vec3],
    edge_node0: wp.array[wp.int32], edge_node1: wp.array[wp.int32], edge_body: wp.array[wp.int32],
    edge_joint: wp.array[wp.vec2i], set_velocity: int,
    # outputs
    body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    """Pose each proxy (midpoint, +Z tangent, +X ``m1``) and optionally its velocity and free joint."""
    e = wp.tid()
    body = edge_body[e]
    if body < 0:
        return
    n0 = edge_node0[e]
    n1 = edge_node1[e]
    x0 = node_q[n0]
    x1 = node_q[n1]
    t = wp.normalize(x1 - x0)
    m1, m2 = material_frame(edge_d1[e], t, edge_q[e])
    R = wp.mat33(
        m1[0], m2[0], t[0],
        m1[1], m2[1], t[1],
        m1[2], m2[2], t[2],
    )
    X = wp.transform(0.5 * (x0 + x1), wp.quat_from_matrix(R))
    body_q[body] = X
    v0 = node_qd[n0]
    v1 = node_qd[n1]
    d = x1 - x0
    twist = wp.spatial_vector(0.5 * (v0 + v1), wp.cross(d, v1 - v0) / wp.dot(d, d))
    if set_velocity != 0:
        body_qd[body] = twist
    j = edge_joint[e]
    if joint_q.shape[0] > 0 and j[0] >= 0:  # a free root joint: joint_q = body_q, joint_qd = body_qd
        for k in range(7):
            joint_q[j[0] + k] = X[k]
        for k in range(6):
            joint_qd[j[1] + k] = twist[k]
