"""ADMM quasi-static spring-mass solver following docs/admm.md.

Per time step, with S the spring incidence matrix (S x)_m = x_j - x_i:

    x-update:  (M/dt^2 + rho S^T S) x^{k+1} = M/dt^2 y + rho S^T (z^k - u^k)
    z-update:  z_m^{k+1} = prox_{W_m / rho}(S_m x^{k+1} + u_m^k)
    u-update:  u^{k+1}   = u^k + S x^{k+1} - z^{k+1}

Kinematic particles (ParticleFlags.ACTIVE cleared, typically mass 0) are pinned
via identity rows in H, and are moved by their current velocity.

Spring damping (model.spring_damping) acts on the rate of change of spring length
and enters W_m through the backward-Euler Rayleigh dissipation
c_m / (2 dt) (|z_m| - l_m^t)^2, with l_m^t the start-of-step length.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warp as wp
from newton import State, Model, Contacts, Control, ModelBuilder, ParticleFlags
from newton.solvers import SolverBase

from ..constants import EPS


@wp.func
def is_active(flags: wp.int32) -> bool:
    return (flags & ParticleFlags.ACTIVE) != 0


@wp.func
def spring_prox(
    d: wp.vec3,
    fallback: wp.vec3,
    stiffness: wp.float32,
    rest_length: wp.float32,
    damping: wp.float32,
    prev_length: wp.float32,
    rho: wp.float32,
    dt: wp.float32,
) -> wp.vec3:
    """Closed-form damped Hookean local step (docs eq. for z_m)."""
    d_len = wp.length(d)
    direction = fallback
    if d_len > EPS:
        direction = d / d_len
    c = damping / dt
    target_length = (rho * d_len + stiffness * rest_length + c * prev_length) / (
        rho + stiffness + c
    )
    return direction * target_length


@wp.kernel
def compute_inertia_b(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    mass: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    dt: wp.float32,
    b_inertia: wp.array[wp.vec3],
):
    tid = wp.tid()
    x = particle_q[tid]
    v = particle_qd[tid]
    if not is_active(particle_flags[tid]):
        # Identity row in H: x^{k+1} = x^t + dt v^t
        b_inertia[tid] = x + v * dt
        return
    world_idx = wp.max(particle_world[tid], 0)
    m = mass[tid]
    # M/dt^2 y with y = x + dt v + dt^2 (g + f_ext / m)
    b_inertia[tid] = (
        (x + v * dt) * (m / (dt * dt)) + gravity[world_idx] * m + particle_f[tid]
    )


@wp.kernel
def init_spring_z(
    particle_q: wp.array[wp.vec3],
    spring_indices: wp.array[wp.int32],
    z: wp.array[wp.vec3],
    u: wp.array[wp.vec3],
):
    tid = wp.tid()
    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    z[tid] = particle_q[j] - particle_q[i]
    u[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def compute_spring_b(
    z: wp.array[wp.vec3],
    u: wp.array[wp.vec3],
    spring_indices: wp.array[wp.int32],
    particle_flags: wp.array[wp.int32],
    rho: wp.float32,
    b: wp.array[wp.vec3],
):
    """b += rho S^T (z^k - u^k)"""
    tid = wp.tid()
    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    r = rho * (z[tid] - u[tid])
    if is_active(particle_flags[i]):
        wp.atomic_add(b, i, -r)
    if is_active(particle_flags[j]):
        wp.atomic_add(b, j, r)


@wp.kernel
def compute_spring_z(
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    u: wp.array[wp.vec3],
    spring_indices: wp.array[wp.int32],
    spring_stiffness: wp.array[wp.float32],
    spring_rest_length: wp.array[wp.float32],
    spring_damping: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    rho: wp.float32,
    dt: wp.float32,
    z: wp.array[wp.vec3],
    dual_res: wp.array[wp.vec3],
):
    """z^{k+1} = prox(S x^{k+1} + u^k); dual_res += rho S^T (z^{k+1} - z^k)"""
    tid = wp.tid()
    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    d = particle_q[j] - particle_q[i] + u[tid]

    # Start-of-step length for damping; its direction is the fallback for a degenerate target
    e_prev = particle_q_prev[j] - particle_q_prev[i]
    l_prev = wp.length(e_prev)
    fallback = wp.vec3(0.0, 0.0, 0.0)
    if l_prev > EPS:
        fallback = e_prev / l_prev

    z_old = z[tid]
    z_new = spring_prox(
        d,
        fallback,
        spring_stiffness[tid],
        spring_rest_length[tid],
        spring_damping[tid],
        l_prev,
        rho,
        dt,
    )
    z[tid] = z_new

    s = rho * (z_new - z_old)
    if is_active(particle_flags[i]):
        wp.atomic_add(dual_res, i, -s)
    if is_active(particle_flags[j]):
        wp.atomic_add(dual_res, j, s)


@wp.kernel
def compute_spring_u(
    particle_q: wp.array[wp.vec3],
    z: wp.array[wp.vec3],
    spring_indices: wp.array[wp.int32],
    particle_flags: wp.array[wp.int32],
    rho: wp.float32,
    u: wp.array[wp.vec3],
    rho_St_u: wp.array[wp.vec3],
    norms: wp.array[wp.float32],
):
    """u^{k+1} = u^k + S x^{k+1} - z^{k+1}; accumulates residual norms."""
    tid = wp.tid()
    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    Sx = particle_q[j] - particle_q[i]
    z_k = z[tid]
    r = Sx - z_k
    u_new = u[tid] + r
    u[tid] = u_new

    wp.atomic_add(norms, 0, wp.dot(r, r))
    wp.atomic_add(norms, 1, wp.dot(Sx, Sx))
    wp.atomic_add(norms, 2, wp.dot(z_k, z_k))

    ru = rho * u_new
    if is_active(particle_flags[i]):
        wp.atomic_add(rho_St_u, i, -ru)
    if is_active(particle_flags[j]):
        wp.atomic_add(rho_St_u, j, ru)


@wp.kernel
def update_velocities(
    q_new: wp.array[wp.vec3],
    q_old: wp.array[wp.vec3],
    dt: wp.float32,
    v_new: wp.array[wp.vec3],
):
    tid = wp.tid()
    v_new[tid] = (q_new[tid] - q_old[tid]) / dt


class ADMMSpringMassSolver(SolverBase):
    """ADMM spring-mass solver (docs/admm.md, "Quasi-static Spring-Mass System").

    Args:
        model: Newton model with springs and registered custom attributes.
        max_iters: Maximum ADMM iterations per time step.
        eps_abs: Absolute tolerance for the primal/dual residual stopping test.
        eps_rel: Relative tolerance for the primal/dual residual stopping test.
            Set both tolerances to 0 to always run ``max_iters`` iterations.
    """

    namespace: str = "spring_mass"

    def __init__(
        self,
        model: Model,
        max_iters: int = 50,
        eps_abs: float = 1e-7,
        eps_rel: float = 1e-4,
    ):
        super().__init__(model)
        if model.spring_indices is None or model.spring_count == 0:
            raise ValueError(
                "Cannot initialize spring mass solver on model without springs"
            )
        if not hasattr(model, self.namespace):
            raise ValueError(
                "Register custom attributes before initializing an ADMMSpringMassSolver"
            )

        self.max_iters = max_iters
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.iterations = 0  # iterations used in the last step

        rho = float(getattr(model, self.namespace).rho.numpy()[0])
        if rho <= 0.0:
            # Auto: match the average spring stiffness
            rho = float(np.mean(model.spring_stiffness.numpy()))
        self.rho = rho

        device = model.device
        N, M = model.particle_count, model.spring_count
        self.z_s = wp.zeros(M, dtype=wp.vec3, device=device)
        self.u_s = wp.zeros(M, dtype=wp.vec3, device=device)
        self.b_inertia = wp.zeros(N, dtype=wp.vec3, device=device)
        self.b = wp.zeros(N, dtype=wp.vec3, device=device)
        self.dual_res = wp.zeros(N, dtype=wp.vec3, device=device)
        self.rho_St_u = wp.zeros(N, dtype=wp.vec3, device=device)
        self.norms = wp.zeros(3, dtype=wp.float32, device=device)

        # H is factorized lazily for the dt passed to step()
        self._factor_dt = None
        self._lu = None

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="rho",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=float,
                default=0.0,  # <= 0 selects mean spring stiffness
                namespace=cls.namespace,
            )
        )

    def _factorize(self, dt: float) -> None:
        """Build and factorize H = M/dt^2 + rho S^T S, with identity rows for kinematic particles."""
        N = self.model.particle_count
        mass = self.model.particle_mass.numpy().astype(np.float64)
        active = (self.model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0
        indices = self.model.spring_indices.numpy().reshape(-1, 2)
        i, j = indices[:, 0], indices[:, 1]
        rho = self.rho

        if np.any(active & (mass <= 0.0)):
            raise ValueError("Active particles must have positive mass")

        # Diagonal: M/dt^2 for active, 1 for kinematic
        diag = np.where(active, mass / dt**2, 1.0)
        rows = [np.arange(N), i, j, i, j]
        cols = [np.arange(N), i, j, j, i]
        vals = [
            diag,
            np.full(len(i), rho),
            np.full(len(i), rho),
            np.full(len(i), -rho),
            np.full(len(i), -rho),
        ]
        rows, cols, vals = map(np.concatenate, (rows, cols, vals))

        # Kinematic rows keep only their identity diagonal
        keep = active[rows] | (np.arange(len(rows)) < N)
        H = sp.csc_matrix((vals[keep], (rows[keep], cols[keep])), shape=(N, N))

        self._lu = spla.splu(H)
        self._factor_dt = dt

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        if state_in.particle_f is None:
            raise ValueError("Cannot run spring mass solver on state without particles")

        if self._factor_dt != dt:
            self._factorize(dt)

        model = self.model
        device = model.device
        N, M = model.particle_count, model.spring_count
        rho = self.rho

        # Per-step constant: M/dt^2 y
        wp.launch(
            kernel=compute_inertia_b,
            dim=N,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                model.particle_mass,
                model.particle_flags,
                model.particle_world,
                model.gravity,
                dt,
            ],
            outputs=[self.b_inertia],
            device=device,
        )
        # z^0 = S x^t, u^0 = 0
        wp.launch(
            kernel=init_spring_z,
            dim=M,
            inputs=[state_in.particle_q, model.spring_indices],
            outputs=[self.z_s, self.u_s],
            device=device,
        )

        for k in range(self.max_iters):
            # 1. Global step
            wp.copy(self.b, self.b_inertia)
            wp.launch(
                kernel=compute_spring_b,
                dim=M,
                inputs=[
                    self.z_s,
                    self.u_s,
                    model.spring_indices,
                    model.particle_flags,
                    rho,
                ],
                outputs=[self.b],
                device=device,
            )
            x_new = self._lu.solve(self.b.numpy().astype(np.float64))
            state_out.particle_q.assign(x_new.astype(np.float32))

            # 2. Local step
            self.dual_res.zero_()
            wp.launch(
                kernel=compute_spring_z,
                dim=M,
                inputs=[
                    state_out.particle_q,
                    state_in.particle_q,
                    self.u_s,
                    model.spring_indices,
                    model.spring_stiffness,
                    model.spring_rest_length,
                    model.spring_damping,
                    model.particle_flags,
                    rho,
                    dt,
                ],
                outputs=[self.z_s, self.dual_res],
                device=device,
            )

            # 3. Dual update
            self.rho_St_u.zero_()
            self.norms.zero_()
            wp.launch(
                kernel=compute_spring_u,
                dim=M,
                inputs=[
                    state_out.particle_q,
                    self.z_s,
                    model.spring_indices,
                    model.particle_flags,
                    rho,
                ],
                outputs=[self.u_s, self.rho_St_u, self.norms],
                device=device,
            )

            self.iterations = k + 1
            if self._converged(N, M):
                break

        wp.launch(
            kernel=update_velocities,
            dim=N,
            inputs=[state_out.particle_q, state_in.particle_q, dt],
            outputs=[state_out.particle_qd],
            device=device,
        )

    def _converged(self, N: int, M: int) -> bool:
        """Boyd et al. (2011) Sec. 3.3.1 stopping criterion."""
        if self.eps_abs <= 0.0 and self.eps_rel <= 0.0:
            return False
        r2, Sx2, z2 = self.norms.numpy()
        s = np.linalg.norm(self.dual_res.numpy())
        y = np.linalg.norm(self.rho_St_u.numpy())
        eps_pri = np.sqrt(3 * M) * self.eps_abs + self.eps_rel * np.sqrt(max(Sx2, z2))
        eps_dual = np.sqrt(3 * N) * self.eps_abs + self.eps_rel * y
        return np.sqrt(r2) <= eps_pri and s <= eps_dual
