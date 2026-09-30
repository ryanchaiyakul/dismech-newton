"""Kernels of the triplet stencil.

One thread per triplet (DOFs ``[x0, theta_e, x1, theta_f, x2]``, 11) evaluates the five
strains ``[eps_e, eps_f, kappa1, kappa2, tau]``, applies the stencil class's energy and
accumulates the gradient and Hessian by the chain rule,
``g = J^T sigma`` and ``K = J^T C J + sum_i sigma_i H_i``. The Hessian is symmetric and only
its upper triangle is stored (:mod:`dismech_newton.system`): each unordered DOF pair is
added once, straight into the fixed-pattern CSR values, whose dtype the kernel follows.

Connectivity is one record per triplet (``vec5i``: ``e, f, n0, n1, n2``), so a thread finds
everything with a single load. The state keeps no edge tangents: the tangents of the
start-of-step configuration (``node_q_old``) are rebuilt from its node positions.

The assembly kernel is built once per concrete stencil class (:func:`make_assemble_kernel`),
which is how the subclass's energy is compiled into it.
"""

from functools import cache
from typing import Any

import warp as wp

from ...frames import reference_twist
from ...system import csr_slot
from .strains import deps, dkappa, dtau, triplet_geometry
from .strains.helper import edge_direction, mat5_11f, unpack_conn, vec5f, vec5i


@wp.kernel
def triplet_strain_kernel(
    node_q: wp.array[wp.vec3],
    node_q_old: wp.array[wp.vec3],
    edge_q: wp.array[float],
    edge_d1_old: wp.array[wp.vec3],
    triplet_ref_twist_old: wp.array[float],
    triplet_conn: wp.array[vec5i],
    edge_length: wp.array[float],
    # outputs
    strain: wp.array[vec5f],
):
    """Strain values only (rest strains and the start-of-step strains of rate-dependent energies)."""
    t = wp.tid()
    e, f, n0, n1, n2 = unpack_conn(triplet_conn[t])
    geom = triplet_geometry(
        node_q[n0], node_q[n1], node_q[n2], edge_q[e], edge_q[f],
        edge_d1_old[e], edge_direction(node_q_old, n0, n1), edge_d1_old[f], edge_direction(node_q_old, n1, n2),
        triplet_ref_twist_old[t], edge_length[e], edge_length[f],
    )
    strain[t] = geom.strain


@cache
def make_assemble_kernel(stencil_cls):
    """Fused assembly kernel of the concrete :class:`~dismech_newton.stencils.TripletStencil` subclass."""
    energy = stencil_cls.energy
    uses_rate = stencil_cls.uses_rate
    params_t = stencil_cls.params_type()

    @wp.kernel(module="unique")
    def assemble_triplet_kernel(
        node_q: wp.array[wp.vec3],
        node_q_old: wp.array[wp.vec3],
        edge_q: wp.array[float],
        edge_d1_old: wp.array[wp.vec3],
        triplet_ref_twist_old: wp.array[float],
        triplet_conn: wp.array[vec5i],
        edge_length: wp.array[float],
        triplet_params: wp.array[params_t],
        triplet_rest: wp.array[wp.vec3],
        strain_prev: wp.array[vec5f],
        dt: float,
        theta_dof_offset: int,
        dof_fixed: wp.array[wp.int32],
        # outputs
        residual: wp.array[float],
        hess_indptr: wp.array[wp.int32],
        hess_indices: wp.array[wp.int32],
        hess_vals: wp.array[Any],
    ):
        t = wp.tid()
        e, f, n0, n1, n2 = unpack_conn(triplet_conn[t])
        geom = triplet_geometry(
            node_q[n0], node_q[n1], node_q[n2], edge_q[e], edge_q[f],
            edge_d1_old[e], edge_direction(node_q_old, n0, n1), edge_d1_old[f], edge_direction(node_q_old, n1, n2),
            triplet_ref_twist_old[t], edge_length[e], edge_length[f],
        )
        # Stretch is measured against the rest length, so its rest strain is exactly zero.
        r = triplet_rest[t]
        rest = vec5f(0.0, 0.0, r[0], r[1], r[2])
        eps_prev = geom.strain
        if wp.static(uses_rate):
            eps_prev = strain_prev[t]
        sigma, C = energy(geom.strain, eps_prev, rest, triplet_params[t], dt)

        # Chain rule: strain Jacobian rows, then sigma-weighted strain Hessians.
        Jse, Jsf, Hse, Hsf = deps(geom.te, geom.tf, geom.ne, geom.nf, edge_length[e], edge_length[f])
        Jb1, Jb2, Hb1, Hb2 = dkappa(geom)
        Ja, Ha = dtau(geom)
        J = mat5_11f()
        for i in range(11):
            J[0, i] = Jse[i]
            J[1, i] = Jsf[i]
            J[2, i] = Jb1[i]
            J[3, i] = Jb2[i]
            J[4, i] = Ja[i]
        K = sigma[0] * Hse + sigma[1] * Hsf + sigma[2] * Hb1 + sigma[3] * Hb2 + sigma[4] * Ha
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
                    wp.atomic_add(hess_vals, csr_slot(hess_indptr, hess_indices, row, col), type(hess_vals[0])(v))

    return assemble_triplet_kernel


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
