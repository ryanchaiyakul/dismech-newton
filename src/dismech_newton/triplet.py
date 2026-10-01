"""The triplet stencil: stretch, bend and twist of two consecutive edges.

One thread per triplet evaluates the strains, applies the energy and adds ``g = J^T sigma`` and the
upper triangle of ``K = J^T C J + sum_i sigma_i H_i`` straight into the CSR Hessian.

An energy is a ``wp.func (eps, eps_prev, rest, params, dt) -> (sigma, C)`` with
``sigma = dE/d eps`` (``vec5f``) and ``C = d^2E/d eps^2`` (``mat55f``). ``params`` is the triplet's
``[k, c]`` (per-strain stiffness and damping, ``vec10f``) and ``eps_prev`` its start-of-step strains.
Kernels are compiled per energy, so a solver's ``energy`` can be replaced (e.g. by a learned one).
"""

from functools import cache

import numpy as np
import warp as wp
from newton import Model, State

from .frames import reference_twist
from .linear import SymmetricCSR, csr_slot
from .strains import (
    edge_direction,
    mat55f,
    rest_strain,
    strain_derivatives,
    triplet_geometry,
    unpack_conn,
    vec5f,
    vec5i,
    vec10f,
)


@wp.func
def linear_energy(eps: vec5f, eps_prev: vec5f, rest: vec5f, p: vec10f, dt: float):
    """``E = 1/2 sum k_i (eps_i - rest_i)^2`` plus strain-rate viscosity ``c_i d eps_i / dt``."""
    rate = (eps - eps_prev) / dt
    sigma = vec5f()
    C = mat55f()
    for i in range(5):
        k = p[i]
        c = p[5 + i]
        sigma[i] = k * (eps[i] - rest[i]) + c * rate[i]
        C[i, i] = k + c / dt
    return sigma, C


@wp.kernel
def strain_kernel(
    node_q: wp.array[wp.vec3],
    edge_q: wp.array[float],
    edge_d1: wp.array[wp.vec3],
    triplet_ref_twist: wp.array[float],
    triplet_conn: wp.array[vec5i],
    edge_length: wp.array[float],
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
    node_q: wp.array[wp.vec3],
    edge_d1: wp.array[wp.vec3],
    triplet_conn: wp.array[vec5i],
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
        node_q: wp.array[wp.vec3],
        node_q_old: wp.array[wp.vec3],
        edge_q: wp.array[float],
        edge_d1_old: wp.array[wp.vec3],
        triplet_ref_twist_old: wp.array[float],
        triplet_conn: wp.array[vec5i],
        edge_length: wp.array[float],
        triplet_params: wp.array[vec10f],
        triplet_rest: wp.array[wp.vec3],
        strain_prev: wp.array[vec5f],
        dt: float,
        theta_dof_offset: int,
        dof_fixed: wp.array[wp.int32],
        # outputs
        residual: wp.array[float],
        hess_indptr: wp.array[wp.int32],
        hess_indices: wp.array[wp.int32],
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
        self._measure(model.state(), rest)
        self.rest = wp.array(rest.numpy()[:, 2:], dtype=wp.vec3, device=self.device)  # [kappa1, kappa2, tau]

    def dofs(self) -> np.ndarray:
        """``(T, 11)`` DOFs ``[x0, theta_e, x1, theta_f, x2]`` of every triplet."""
        e, f, n0, n1, n2 = self.conn.numpy().astype(np.int64).T
        th = self.num_node_dofs
        return np.column_stack([3 * n0, 3 * n0 + 1, 3 * n0 + 2, th + e, 3 * n1, 3 * n1 + 1, 3 * n1 + 2,
                                th + f, 3 * n2, 3 * n2 + 1, 3 * n2 + 2])

    def _measure(self, state: State, out: wp.array) -> None:
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
        self._measure(state_in, self.strain_prev)

    def assemble(self, state_in: State, state_out: State, residual: wp.array, hessian: SymmetricCSR, dt: float):
        """Add the gradient and upper-triangle Hessian at ``state_out`` into ``residual`` and ``hessian``."""
        s_in, s_out = state_in.dismech, state_out.dismech
        wp.launch(
            self._kernel,
            dim=self.count,
            inputs=[
                state_out.particle_q, state_in.particle_q, s_out.edge_q, s_in.edge_d1_q, s_in.triplet_ref_twist_q,
                self.conn, self.der.edge_length, self.params, self.rest, self.strain_prev, dt, self.num_node_dofs,
                self.dof_fixed,
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
