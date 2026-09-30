"""The Newton system's vocabulary: the flat DOF vector and the sparse Hessian.

The unknowns are one flat vector. DOF ``3 * node + k`` is component ``k`` of
``particle_q[node]`` and DOF ``3 * N + edge`` is the twist angle ``edge_q[edge]``. A state
owns that vector as ``state.dismech.q`` (velocities: ``state.dismech.qd``), and
``particle_q``, ``edge_q`` and their velocity arrays are *views* into it, so kernels can work
on either form and integrators are plain elementwise 1-D kernels. :func:`flatten_state` sets a
state up this way, and :func:`dof_constants` gives the per-DOF mass and Dirichlet flags.

:class:`SymmetricCSR` is the Hessian: the upper triangle of a symmetric CSR matrix with a
pattern fixed by the topology (every stencil couples its DOFs, every DOF couples to itself
through the inertia). Kernels add straight into ``vals``, finding the slot of ``(row, col)``
by binary search in the sorted row (:func:`csr_slot`); there is no COO staging. Storing
only the upper triangle halves the values, the column indices and the cuDSS factor. The
values are float64 by default; float32 halves them and the factor again, at the price of a
less accurate solve on stiff systems (the assembly itself is float32 either way).
"""

from collections.abc import Sequence

import numpy as np
import warp as wp
from newton import Model, ParticleFlags, State

# -- DOF vector -------------------------------------------------------------------------


def dof_constants(model: Model) -> tuple[wp.array, wp.array]:
    """Per-DOF ``(mass, fixed)`` of ``model``: node mass / edge twist inertia, and the Dirichlet flag.

    A node DOF is fixed when its particle is not ``ACTIVE``, a twist DOF when ``edge_fixed`` is set.
    """
    der = model.dismech
    active = (model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0
    fixed = np.concatenate([np.repeat(~active, 3), der.edge_fixed.numpy() != 0])
    mass = np.concatenate([np.repeat(model.particle_mass.numpy(), 3), der.edge_inertia.numpy()])
    return (
        wp.array(mass.astype(np.float32), dtype=float, device=model.device),
        wp.array(fixed, dtype=wp.int32, device=model.device),
    )


def flatten_state(state: State) -> None:
    """Make ``state`` own the flat vectors ``dismech.q`` / ``dismech.qd`` (idempotent).

    Allocates them from the current values and rebinds ``particle_q``, ``edge_q``,
    ``particle_qd`` and ``edge_qd`` as views into them, so both forms stay in sync
    (``State.assign`` copies into the views and keeps the aliasing intact). Call it
    before any CUDA graph capture; the solver does so on the first step.
    """
    ns = state.dismech
    if getattr(ns, "q", None) is not None:
        return
    n = state.particle_q.shape[0]
    nd = 3 * n
    num_dofs = nd + ns.edge_q.shape[0]
    for flat_name, node_name, edge_name in (("q", "particle_q", "edge_q"), ("qd", "particle_qd", "edge_qd")):
        flat = wp.zeros(num_dofs, dtype=float, device=state.particle_q.device)
        nodes = flat[:nd].reshape((n, 3)).view(wp.vec3)
        edges = flat[nd:]
        nodes.assign(getattr(state, node_name))
        edges.assign(getattr(ns, edge_name))
        setattr(state, node_name, nodes)
        setattr(ns, edge_name, edges)
        setattr(ns, flat_name, flat)


# -- Hessian ----------------------------------------------------------------------------


class SymmetricCSR:
    """Upper-triangle CSR of a symmetric ``n x n`` matrix with a fixed pattern.

    ``dtype`` is the value type, ``wp.float64`` or ``wp.float32``; kernels that add into
    ``vals`` are generic over it.
    """

    def __init__(self, num_dofs: int, stencil_dofs: Sequence[np.ndarray], device, dtype=wp.float64) -> None:
        indptr, indices = self.upper_csr_pattern(num_dofs, stencil_dofs)
        self.n = num_dofs
        self.dtype = dtype
        self.indptr = wp.array(indptr, dtype=wp.int32, device=device)
        self.indices = wp.array(indices, dtype=wp.int32, device=device)
        self.vals = wp.zeros(len(indices), dtype=dtype, device=device)

    @property
    def nnz(self) -> int:
        return self.vals.shape[0]

    @staticmethod
    def upper_csr_pattern(num_dofs: int, stencil_dofs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Sorted upper-triangle CSR ``(indptr, indices)`` covering every stencil block and the diagonal.

        ``stencil_dofs`` holds one ``(count, dofs_per_stencil)`` array per stencil family.
        """
        keys = [np.arange(num_dofs, dtype=np.int64) * (num_dofs + 1)]
        for dofs in stencil_dofs:
            i, j = np.triu_indices(dofs.shape[1])  # the unordered DOF pairs of a stencil
            a, b = dofs[:, i], dofs[:, j]
            keys.append((np.minimum(a, b) * num_dofs + np.maximum(a, b)).ravel())
        keys = np.unique(np.concatenate(keys))
        indptr = np.zeros(num_dofs + 1, dtype=np.int32)
        np.cumsum(np.bincount(keys // num_dofs, minlength=num_dofs), out=indptr[1:])
        return indptr, (keys % num_dofs).astype(np.int32)


@wp.func
def csr_slot(indptr: wp.array[wp.int32], indices: wp.array[wp.int32], row: int, col: int) -> int:
    """Index into the CSR values of ``(row, col)``, which must be in the pattern."""
    lo = indptr[row]
    hi = indptr[row + 1] - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if indices[mid] < col:
            lo = mid + 1
        else:
            hi = mid
    return lo
