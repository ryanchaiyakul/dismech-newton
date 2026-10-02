"""The triplet stencil: stretch, bend and twist of two consecutive edges.

One thread per triplet evaluates the strains, applies the energy and adds ``g = J^T sigma`` and the
upper triangle of ``K = J^T C J + sum_i sigma_i H_i`` straight into the CSR Hessian.

An energy is a ``wp.func (eps, eps_prev, rest, params, dt) -> (sigma, C)`` with
``sigma = dE/d eps`` (``vec5f``) and ``C = d^2E/d eps^2`` (``mat55f``). ``params`` is the triplet's
``[k, c]`` (per-strain stiffness and damping, ``vec10f``) and ``eps_prev`` its start-of-step strains.
Kernels are compiled per energy, so a solver's ``energy`` can be replaced (e.g. by a learned one).

ADMM (:class:`TripletTerm`): a triplet's local variable is ``z = [x1 - x0, theta_e, x2 - x1,
theta_f - theta_e]``; translation invariance makes ``W(z)`` the triplet energy at ``x0 = 0``. Splitting
the twist difference puts the twist Laplacian into the ADMM matrix ``H``, and the penalty
``diag(rho_x I3, rho_abs, rho_x I3, rho_twist)`` keeps ``H`` from coupling the x, y and z components.
"""

from functools import cache

import numpy as np
import warp as wp
from newton import Model, State

from .frames import fixed_node, flatten_state, node, reference_twist, scatter_dof, scatter_node
from .linear import SymmetricCSR, csr_slot
from .strains import (
    TripletGeometry,
    edge_direction,
    mat55f,
    rest_strain,
    strain_derivatives,
    strain_gradient,
    triplet_geometry,
    unpack_conn,
    vec5f,
    vec5i,
    vec10f,
)

vec8f = wp.types.vector(8, float)
mat88f = wp.types.matrix((8, 8), float)
mat11_8f = wp.types.matrix((11, 8), float)


@wp.func
def linear_energy(eps: vec5f, eps_prev: vec5f, rest: vec5f, p: vec10f, dt: float):
    """``E = 1/2 sum k_i (eps_i - rest_i)^2`` plus strain-rate viscosity ``c_i d eps_i / dt``."""
    k = vec5f(p[0], p[1], p[2], p[3], p[4])
    c = vec5f(p[5], p[6], p[7], p[8], p[9])
    sigma = wp.cw_mul(k, eps - rest) + wp.cw_mul(c, eps - eps_prev) / dt
    return sigma, wp.diag(k + c / dt)


class Triplets:
    """The model's triplets, bound for a solver.

    Rest curvatures and twist are measured on the initial configuration, so the rod starts
    unstressed. Never writes to rows or columns of ``fixed`` DOFs.
    """

    def __init__(self, model: Model, fixed: wp.array, energy=linear_energy):
        d = model.dismech
        self.model = model
        self.der = d
        self.device = wp.get_device(model.device)
        self.energy = energy
        self.count = model.custom_frequency_counts.get("dismech:triplet", 0)
        self.num_node_dofs = 3 * model.particle_count
        self.dof_fixed = fixed
        self.params = d.triplet_params
        self._kernel = make_assemble_kernel(energy)

        # One connectivity record per triplet, (e, f, n0, n1, n2): no dependent loads in the kernels.
        e, f = d.triplet_edge0.numpy(), d.triplet_edge1.numpy()
        node0, node1 = d.edge_node0.numpy(), d.edge_node1.numpy()
        conn = np.column_stack((e, f, node0[e], node1[e], node1[f])).astype(np.int32)
        self.conn = wp.array(conn, dtype=vec5i, device=self.device)

        self.strain_prev = wp.zeros(self.count, dtype=vec5f, device=self.device)
        rest = wp.zeros(self.count, dtype=vec5f, device=self.device)
        self.measure(model.state(), rest)
        self.rest = wp.array(rest.numpy()[:, 2:], dtype=wp.vec3, device=self.device)  # [kappa1, kappa2, tau]

    def dofs(self) -> np.ndarray:
        """``(T, 11)`` DOFs ``[x0, theta_e, x1, theta_f, x2]`` of every triplet."""
        e, f, n0, n1, n2 = self.conn.numpy().astype(np.int64).T
        th = self.num_node_dofs
        return np.column_stack([3 * n0, 3 * n0 + 1, 3 * n0 + 2, th + e, 3 * n1, 3 * n1 + 1, 3 * n1 + 2,
                                th + f, 3 * n2, 3 * n2 + 1, 3 * n2 + 2])

    def measure(self, state: State, out: wp.array) -> None:
        s = state.dismech
        wp.launch(
            strain_kernel,
            dim=self.count,
            inputs=[state.particle_q, s.edge_q, s.edge_d1_q, s.triplet_ref_twist_q, self.conn, self.der.edge_length],
            outputs=[out],
            device=self.device,
        )

    def begin_step(self, state_in: State) -> None:
        """Record the start-of-step strains."""
        self.measure(state_in, self.strain_prev)

    def assemble(self, state_in: State, state_out: State, residual: wp.array, hessian: SymmetricCSR, dt: float):
        """Add the gradient and upper-triangle Hessian at ``state_out`` into ``residual`` and ``hessian``."""
        s_in = state_in.dismech
        self.assemble_at(state_out.particle_q, state_out.dismech.edge_q, state_in.particle_q, s_in.edge_d1_q,
                         s_in.triplet_ref_twist_q, self.strain_prev, residual, hessian, dt)

    def assemble_at(self, node_q, edge_q, node_q_old, edge_d1_old, ref_twist_old, strain_prev, residual: wp.array,
                    hessian: SymmetricCSR, dt: float) -> None:
        """:meth:`assemble` on explicit arrays: at ``(node_q, edge_q)``, frames from the step start."""
        wp.launch(
            self._kernel,
            dim=self.count,
            inputs=[
                node_q, node_q_old, edge_q, edge_d1_old, ref_twist_old, self.conn, self.der.edge_length,
                self.params, self.rest, strain_prev, dt, self.num_node_dofs, self.dof_fixed,
            ],
            outputs=[residual, hessian.indptr, hessian.indices, hessian.vals],
            device=self.device,
        )

    def end_step(self, state_in: State, state_out: State) -> None:
        """Advance the reference twist (the edge frames of ``state_out`` are already advanced)."""
        wp.launch(
            advance_ref_twist_kernel,
            dim=self.count,
            inputs=[state_out.particle_q, state_out.dismech.edge_d1_q, self.conn, state_in.dismech.triplet_ref_twist_q],
            outputs=[state_out.dismech.triplet_ref_twist_q],
            device=self.device,
        )


class TripletTerm:
    """Local variables, duals and prox of every triplet.

    Args:
        triplets: The bound triplets; read in place.
        rho_scale: ``rho_x`` is ``rho_scale`` times the geometric mean of the triplet's axial and
            transverse stiffness in edge-vector space at rest, ``rho_twist`` ``rho_scale`` times its
            twist stiffness.
        rho_abs_ratio: ``rho_abs / rho_twist``, the weight of the absolute twist angle.
        local_iterations: Newton iterations of every prox (warm-started).
    """

    def __init__(self, triplets: Triplets, *, rho_scale: float, rho_abs_ratio: float, local_iterations: int):
        self.triplets = triplets
        self.local_iterations = local_iterations
        self.device = triplets.device
        self._local_kernel, stiffness_kernel = make_admm_kernels(triplets.energy)
        tr, der = triplets, triplets.der

        rest_state = tr.model.state()
        flatten_state(rest_state)
        stiffness = wp.zeros(tr.count, dtype=wp.vec3, device=self.device)
        wp.launch(
            stiffness_kernel,
            dim=tr.count,
            inputs=[
                rest_state.dismech.q, rest_state.particle_q, rest_state.dismech.edge_d1_q,
                rest_state.dismech.triplet_ref_twist_q, tr.conn, der.edge_length, tr.params, tr.rest,
                tr.num_node_dofs,
            ],
            outputs=[stiffness],
            device=self.device,
        )
        axial, transverse, twist = stiffness.numpy().astype(np.float64).T
        rho_x = np.where(transverse > 0.0, np.sqrt(axial * np.maximum(transverse, 0.0)), axial)
        rho_twist = np.where(twist > 0.0, twist, 1.0e-3 * rho_x * np.mean(der.edge_length.numpy()) ** 2)
        # The absolute angle only enters through the bend/twist coupling, so it gets a small weight.
        rho = rho_scale * np.column_stack((rho_x, rho_abs_ratio * rho_twist, rho_twist))
        self.rho = wp.array(rho.astype(np.float32), dtype=wp.vec3, device=self.device)

        self.z = wp.zeros(tr.count, dtype=vec8f, device=self.device)
        self.u = wp.zeros(tr.count, dtype=vec8f, device=self.device)
        self._initialized = False

    def penalty(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """COO ``(rows, cols, vals)`` of ``S^T P S``, both triangles."""
        dofs = self.triplets.dofs()  # [x0, theta_e, x1, theta_f, x2]
        r = self.rho.numpy().astype(np.float64)
        rows, cols, vals = [], [], []
        for k in range(3):
            for a, b in ((0, 4), (4, 8)):  # edge e = x1 - x0, edge f = x2 - x1
                i, j = dofs[:, a + k], dofs[:, b + k]
                rows += [i, j, i, j]
                cols += [i, j, j, i]
                vals += [r[:, 0], r[:, 0], -r[:, 0], -r[:, 0]]
        i, j = dofs[:, 3], dofs[:, 7]  # theta_e (weight rho_abs) and theta_f - theta_e (rho_twist)
        rows += [i, i, j, i, j]
        cols += [i, i, j, j, i]
        vals += [r[:, 1], r[:, 2], r[:, 2], -r[:, 2], -r[:, 2]]
        return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

    def begin_step(self, state_in: State) -> None:
        """On the first step: ``z = S q``, ``u = 0`` (the rod starts at rest); later steps warm-start."""
        if self._initialized:
            return
        tr = self.triplets
        e, f, n0, n1, n2 = tr.conn.numpy().T
        q = state_in.dismech.q.numpy()
        x = q[: tr.num_node_dofs].reshape(-1, 3)
        th = q[tr.num_node_dofs :]
        ee, ef = x[n1] - x[n0], x[n2] - x[n1]
        self.z.assign(np.column_stack((ee, th[e], ef, th[f] - th[e])).astype(np.float32))
        self._initialized = True

    def local(self, state_in: State, q: wp.array, dt: float, rhs: wp.array, stats: wp.array, update: int) -> None:
        """With ``update``: prox and dual step at ``q``, residual into ``stats[0]``. Always: ``rhs += S^T P (z - u)``."""
        tr, s_in = self.triplets, state_in.dismech
        wp.launch(
            self._local_kernel,
            dim=tr.count,
            inputs=[
                q, state_in.particle_q, s_in.edge_d1_q, s_in.triplet_ref_twist_q, tr.conn, tr.der.edge_length,
                tr.params, tr.rest, tr.strain_prev, dt, tr.num_node_dofs, self.rho, tr.dof_fixed,
                self.local_iterations, update,
            ],
            outputs=[self.z, self.u, rhs, stats],
            device=self.device,
        )


# -- kernels ------------------------------------------------------------------------------


@wp.kernel
def strain_kernel(
    node_q: wp.array[wp.vec3], edge_q: wp.array[float], edge_d1: wp.array[wp.vec3], triplet_ref_twist: wp.array[float],
    triplet_conn: wp.array[vec5i], edge_length: wp.array[float],
    # outputs
    strain: wp.array[vec5f],
):
    """Strains of a state whose frames are current (the transport is the identity)."""
    t = wp.tid()
    e, f, n0, n1, n2 = unpack_conn(triplet_conn[t])
    geom = triplet_geometry(
        node_q[n0], node_q[n1], node_q[n2], edge_q[e], edge_q[f],
        edge_d1[e], edge_direction(node_q, n0, n1), edge_d1[f], edge_direction(node_q, n1, n2),
        triplet_ref_twist[t], edge_length[e], edge_length[f],
    )
    strain[t] = geom.strain


@wp.kernel
def advance_ref_twist_kernel(
    node_q: wp.array[wp.vec3], edge_d1: wp.array[wp.vec3], triplet_conn: wp.array[vec5i],
    ref_twist_prev: wp.array[float],
    # outputs
    ref_twist: wp.array[float],
):
    t = wp.tid()
    e, f, n0, n1, n2 = unpack_conn(triplet_conn[t])
    ref_twist[t] = reference_twist(
        edge_d1[e], edge_direction(node_q, n0, n1), edge_d1[f], edge_direction(node_q, n1, n2), ref_twist_prev[t]
    )


@cache
def make_assemble_kernel(energy):
    """Gradient and upper-triangle Hessian assembly with ``energy`` compiled in."""

    @wp.kernel(module="unique")
    def assemble_kernel(
        node_q: wp.array[wp.vec3], node_q_old: wp.array[wp.vec3], edge_q: wp.array[float],
        edge_d1_old: wp.array[wp.vec3], triplet_ref_twist_old: wp.array[float], triplet_conn: wp.array[vec5i],
        edge_length: wp.array[float], triplet_params: wp.array[vec10f], triplet_rest: wp.array[wp.vec3],
        strain_prev: wp.array[vec5f], dt: float, theta_dof_offset: int, dof_fixed: wp.array[wp.int32],
        # outputs
        residual: wp.array[float], hess_indptr: wp.array[wp.int32], hess_indices: wp.array[wp.int32],
        hess_vals: wp.array[wp.float64],
    ):
        t = wp.tid()
        e, f, n0, n1, n2 = unpack_conn(triplet_conn[t])
        geom = triplet_geometry(
            node_q[n0], node_q[n1], node_q[n2], edge_q[e], edge_q[f],
            edge_d1_old[e], edge_direction(node_q_old, n0, n1), edge_d1_old[f], edge_direction(node_q_old, n1, n2),
            triplet_ref_twist_old[t], edge_length[e], edge_length[f],
        )
        sigma, C = energy(geom.strain, strain_prev[t], rest_strain(triplet_rest[t]), triplet_params[t], dt)
        J, K = strain_derivatives(geom, sigma, edge_length[e], edge_length[f])
        K = K + wp.transpose(J) * C * J
        g = wp.transpose(J) * sigma

        dofs = wp.vector(length=11, dtype=wp.int32)
        for k in range(3):
            dofs[k] = 3 * n0 + k
            dofs[4 + k] = 3 * n1 + k
            dofs[8 + k] = 3 * n2 + k
        dofs[3] = theta_dof_offset + e
        dofs[7] = theta_dof_offset + f

        for i in range(11):
            wp.atomic_add(residual, dofs[i], g[i])
        for i in range(11):
            for j in range(i, 11):
                v = K[i, j]
                if j > i:
                    v = 0.5 * (v + K[j, i])
                if v != 0.0 and dof_fixed[dofs[i]] == 0 and dof_fixed[dofs[j]] == 0:
                    row = wp.min(dofs[i], dofs[j])
                    col = wp.max(dofs[i], dofs[j])
                    wp.atomic_add(hess_vals, csr_slot(hess_indptr, hess_indices, row, col), wp.float64(v))

    return assemble_kernel


@cache
def make_residual_kernel(energy):
    """The gradient ``J^T sigma`` alone, on the flat DOF vector, differentiable with ``wp.Tape``
    (for the step adjoint, :mod:`~dismech_newton.adjoint`)."""

    @wp.kernel(module="unique")
    def residual_kernel(
        q: wp.array[float], q_old: wp.array[float], edge_d1_old: wp.array[wp.vec3],
        triplet_ref_twist_old: wp.array[float], triplet_conn: wp.array[vec5i], edge_length: wp.array[float],
        triplet_params: wp.array[vec10f], triplet_rest: wp.array[wp.vec3], strain_prev: wp.array[vec5f], dt: float,
        theta_dof_offset: int, dof_fixed: wp.array[wp.int32],
        # outputs
        residual: wp.array[float],
    ):
        t = wp.tid()
        e, f, n0, n1, n2 = unpack_conn(triplet_conn[t])
        ie = theta_dof_offset + e
        i_f = theta_dof_offset + f
        l0e = edge_length[e]
        l0f = edge_length[f]
        geom = triplet_geometry(
            node(q, n0), node(q, n1), node(q, n2), q[ie], q[i_f],
            edge_d1_old[e], wp.normalize(node(q_old, n1) - node(q_old, n0)),
            edge_d1_old[f], wp.normalize(node(q_old, n2) - node(q_old, n1)),
            triplet_ref_twist_old[t], l0e, l0f,
        )
        sigma, C = energy(geom.strain, strain_prev[t], rest_strain(triplet_rest[t]), triplet_params[t], dt)
        g = strain_gradient(geom, sigma, l0e, l0f)
        scatter_node(residual, dof_fixed, n0, wp.vec3(g[0], g[1], g[2]))
        scatter_node(residual, dof_fixed, n1, wp.vec3(g[4], g[5], g[6]))
        scatter_node(residual, dof_fixed, n2, wp.vec3(g[8], g[9], g[10]))
        scatter_dof(residual, dof_fixed, ie, g[3])
        scatter_dof(residual, dof_fixed, i_f, g[7])

    return residual_kernel


# -- ADMM local step ----------------------------------------------------------------------


@wp.func
def _reduction() -> mat11_8f:
    """``d q_triplet / d z``: ``x0 = 0, theta_e = z3, x1 = e, theta_f = z3 + z7, x2 = e + f``."""
    T = mat11_8f()
    for k in range(3):
        T[4 + k, k] = 1.0
        T[8 + k, k] = 1.0
        T[8 + k, 4 + k] = 1.0
    T[3, 3] = 1.0
    T[7, 3] = 1.0
    T[7, 7] = 1.0
    return T


@wp.func
def _local_geometry(
    z: vec8f, d1e_old: wp.vec3, te_old: wp.vec3, d1f_old: wp.vec3, tf_old: wp.vec3, ref_twist_old: float,
    l0e: float, l0f: float,
):
    e = wp.vec3(z[0], z[1], z[2])
    f = wp.vec3(z[4], z[5], z[6])
    return triplet_geometry(
        wp.vec3(0.0, 0.0, 0.0), e, e + f, z[3], z[3] + z[7], d1e_old, te_old, d1f_old, tf_old, ref_twist_old, l0e, l0f
    )


@wp.func
def _local_derivatives(geom: TripletGeometry, sigma: vec5f, C: mat55f, l0e: float, l0f: float):
    """Gradient, Hessian and Gauss-Newton Hessian of ``W`` with respect to ``z``."""
    J, K_geo = strain_derivatives(geom, sigma, l0e, l0f)
    T = _reduction()
    JT = J * T  # (5, 8)
    K_gn = wp.transpose(JT) * C * JT
    K = K_gn + wp.transpose(T) * K_geo * T
    g = wp.transpose(JT) * sigma
    return g, K, K_gn


@wp.func
def _cholesky_solve(A: mat88f, b: vec8f):
    """``A^{-1} b`` and whether ``A`` was positive definite."""
    L = mat88f()
    ok = int(1)
    for j in range(8):
        s = A[j, j]
        for k in range(j):
            s = s - L[j, k] * L[j, k]
        if s <= 1.0e-12 * wp.abs(A[j, j]) or s <= 0.0:
            ok = 0
            s = wp.max(wp.abs(A[j, j]), 1.0e-12)
        L[j, j] = wp.sqrt(s)
        for i in range(j + 1, 8):
            v = A[i, j]
            for k in range(j):
                v = v - L[i, k] * L[j, k]
            L[i, j] = v / L[j, j]
    y = vec8f()
    for i in range(8):
        v = b[i]
        for k in range(i):
            v = v - L[i, k] * y[k]
        y[i] = v / L[i, i]
    x = vec8f()
    for ii in range(8):
        i = 7 - ii
        v = y[i]
        for k in range(i + 1, 8):
            v = v - L[k, i] * x[k]
        x[i] = v / L[i, i]
    return x, ok


@wp.func
def _penalty(rho: wp.vec3) -> vec8f:
    """``(rho_x, rho_abs, rho_twist)`` spread over ``[e, theta_e, f, dtheta]``."""
    return vec8f(rho[0], rho[0], rho[0], rho[1], rho[0], rho[0], rho[0], rho[2])


@cache
def make_admm_kernels(energy):
    """Local-step and rest-stiffness kernels with ``energy`` compiled in."""

    @wp.kernel(module="unique", module_options={"fast_math": True})
    def local_kernel(
        q: wp.array[float], node_q_old: wp.array[wp.vec3], edge_d1_old: wp.array[wp.vec3],
        triplet_ref_twist_old: wp.array[float], triplet_conn: wp.array[vec5i], edge_length: wp.array[float],
        triplet_params: wp.array[vec10f], triplet_rest: wp.array[wp.vec3], strain_prev: wp.array[vec5f], dt: float,
        theta_dof_offset: int, rho: wp.array[wp.vec3], dof_fixed: wp.array[wp.int32], local_iterations: int,
        update: int,
        # outputs
        z: wp.array[vec8f], u: wp.array[vec8f], rhs: wp.array[float], stats: wp.array[float],
    ):
        """With ``update``: the prox of ``W`` (Newton from the previous ``z``) and the dual step.
        Always: ``rhs += S_f^T P (z - u - S q_fixed)`` for the next global step."""
        t = wp.tid()
        e, f, n0, n1, n2 = unpack_conn(triplet_conn[t])
        ie = theta_dof_offset + e
        i_f = theta_dof_offset + f
        P = _penalty(rho[t])
        zt = z[t]
        ut = u[t]

        if update != 0:
            x0 = node(q, n0)
            x1 = node(q, n1)
            x2 = node(q, n2)
            ee = x1 - x0
            ef = x2 - x1
            Sq = vec8f(ee[0], ee[1], ee[2], q[ie], ef[0], ef[1], ef[2], q[i_f] - q[ie])
            d = Sq + ut

            d1e = edge_d1_old[e]
            d1f = edge_d1_old[f]
            te_old = edge_direction(node_q_old, n0, n1)
            tf_old = edge_direction(node_q_old, n1, n2)
            l0e = edge_length[e]
            l0f = edge_length[f]
            rest = rest_strain(triplet_rest[t])
            z_old = zt
            for _it in range(local_iterations):
                geom = _local_geometry(zt, d1e, te_old, d1f, tf_old, triplet_ref_twist_old[t], l0e, l0f)
                sigma, C = energy(geom.strain, strain_prev[t], rest, triplet_params[t], dt)
                g, K, K_gn = _local_derivatives(geom, sigma, C, l0e, l0f)
                G = g + wp.cw_mul(P, zt - d)
                step, ok = _cholesky_solve(K + wp.diag(P), G)
                if ok == 0:  # indefinite: fall back to Gauss-Newton, positive definite
                    step, ok = _cholesky_solve(K_gn + wp.diag(P), G)
                # Trust region: far from the solution (fast motion) a full step can collapse an edge.
                s_e = wp.length(wp.vec3(step[0], step[1], step[2])) / (0.25 * l0e)
                s_f = wp.length(wp.vec3(step[4], step[5], step[6])) / (0.25 * l0f)
                s_t = wp.max(wp.abs(step[3]), wp.abs(step[7])) / 0.5
                zt = zt - step / wp.max(1.0, wp.max(s_t, wp.max(s_e, s_f)))
            ut = d - zt
            z[t] = zt
            u[t] = ut

            # Primal residual S q - z and the change of z, relative to the rest lengths for edges.
            res = Sq - zt
            m = wp.max(wp.abs(res[3]), wp.abs(res[7]))
            m = wp.max(m, wp.length(wp.vec3(res[0], res[1], res[2])) / l0e)
            m = wp.max(m, wp.length(wp.vec3(res[4], res[5], res[6])) / l0f)
            dz = zt - z_old
            m = wp.max(m, wp.max(wp.abs(dz[3]), wp.abs(dz[7])))
            m = wp.max(m, wp.length(wp.vec3(dz[0], dz[1], dz[2])) / l0e)
            m = wp.max(m, wp.length(wp.vec3(dz[4], dz[5], dz[6])) / l0f)
            wp.atomic_max(stats, 0, m)

        # rhs += S_f^T P (z - u - S q_c): the fixed DOFs' coupling moves to the right-hand side.
        c0 = fixed_node(q, dof_fixed, n0)
        c1 = fixed_node(q, dof_fixed, n1)
        c2 = fixed_node(q, dof_fixed, n2)
        ce = c1 - c0
        cf = c2 - c1
        th_e = float(0.0)
        th_f = float(0.0)
        if dof_fixed[ie] != 0:
            th_e = q[ie]
        if dof_fixed[i_f] != 0:
            th_f = q[i_f]
        w = wp.cw_mul(P, zt - ut - vec8f(ce[0], ce[1], ce[2], th_e, cf[0], cf[1], cf[2], th_f - th_e))
        we = wp.vec3(w[0], w[1], w[2])
        wf = wp.vec3(w[4], w[5], w[6])
        scatter_node(rhs, dof_fixed, n0, -we)
        scatter_node(rhs, dof_fixed, n1, we - wf)
        scatter_node(rhs, dof_fixed, n2, wf)
        scatter_dof(rhs, dof_fixed, ie, w[3] - w[7])
        scatter_dof(rhs, dof_fixed, i_f, w[7])

    @wp.kernel(module="unique")
    def rest_stiffness_kernel(
        q: wp.array[float], node_q: wp.array[wp.vec3], edge_d1: wp.array[wp.vec3], triplet_ref_twist: wp.array[float],
        triplet_conn: wp.array[vec5i], edge_length: wp.array[float], triplet_params: wp.array[vec10f],
        triplet_rest: wp.array[wp.vec3], theta_dof_offset: int,
        # outputs
        stiffness: wp.array[wp.vec3],
    ):
        """Per triplet ``(axial, transverse, twist)`` stiffness of ``W`` in ``z`` at rest."""
        t = wp.tid()
        e, f, n0, n1, n2 = unpack_conn(triplet_conn[t])
        ee = node_q[n1] - node_q[n0]
        ef = node_q[n2] - node_q[n1]
        th_e = q[theta_dof_offset + e]
        z0 = vec8f(ee[0], ee[1], ee[2], th_e, ef[0], ef[1], ef[2], q[theta_dof_offset + f] - th_e)
        te = wp.normalize(ee)
        tf = wp.normalize(ef)
        l0e = edge_length[e]
        l0f = edge_length[f]
        geom = _local_geometry(z0, edge_d1[e], te, edge_d1[f], tf, triplet_ref_twist[t], l0e, l0f)
        sigma, C = energy(geom.strain, geom.strain, rest_strain(triplet_rest[t]), triplet_params[t], 1.0)
        g, K, K_gn = _local_derivatives(geom, sigma, C, l0e, l0f)
        Kee = wp.mat33()
        Kff = wp.mat33()
        for i in range(3):
            for j in range(3):
                Kee[i, j] = K_gn[i, j]
                Kff[i, j] = K_gn[4 + i, 4 + j]
        ae = wp.dot(te, Kee * te)
        af = wp.dot(tf, Kff * tf)
        be = 0.5 * (wp.trace(Kee) - ae)
        bf = 0.5 * (wp.trace(Kff) - af)
        stiffness[t] = wp.vec3(0.5 * (ae + af), 0.5 * (be + bf), K_gn[7, 7])

    return local_kernel, rest_stiffness_kernel
