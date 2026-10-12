"""The flat DOF vector's helpers and the global matrix: an upper-triangle float64 CSR on the device."""

import numpy as np
import scipy.sparse as sp
import warp as wp


class SymmetricCSR:
    """Upper-triangle CSR; the pattern is the diagonal and every DOF pair within a row of each ``(count, width)``
    DOF array of ``stencils``."""

    def __init__(self, n: int, stencils: tuple[np.ndarray, ...], device) -> None:
        keys = [np.arange(n, dtype=np.int64) * (n + 1)] + [_pairs(d, n).ravel() for d in stencils]
        keys = np.unique(np.concatenate(keys))
        indptr = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(np.bincount(keys // n, minlength=n), out=indptr[1:])
        self.n = n
        self.pattern = (indptr, (keys % n).astype(np.int32))  # host copy
        self.indptr = wp.array(indptr, dtype=wp.int32, device=device)
        self.indices = wp.array(self.pattern[1], dtype=wp.int32, device=device)
        self.vals = wp.zeros(len(keys), dtype=wp.float64, device=device)
        self._keys = keys

    def slots(self, d: np.ndarray) -> wp.array:
        """``[p, r]``: the slot of row ``r``'s ``p``-th DOF pair of ``d`` (``np.triu_indices`` order, so kernels
        scatter without searching); ``d`` within the pattern."""
        slot = np.searchsorted(self._keys, _pairs(d, self.n)).T.astype(np.int32)
        return wp.array(np.ascontiguousarray(slot), dtype=wp.int32, device=self.vals.device)

    @property
    def nnz(self) -> int:
        return self.vals.shape[0]

    def to_scipy(self) -> sp.csr_matrix:
        """The full symmetric matrix on the host."""
        upper = sp.csr_matrix((self.vals.numpy(), self.indices.numpy(), self.indptr.numpy()), shape=(self.n, self.n))
        return (upper + upper.T - sp.diags(upper.diagonal())).tocsr()


def _pairs(d: np.ndarray, n: int) -> np.ndarray:
    i, j = np.triu_indices(d.shape[1])
    a, b = d[:, i].astype(np.int64), d[:, j].astype(np.int64)
    return np.minimum(a, b) * n + np.maximum(a, b)


@wp.func
def row_slot(indptr: wp.array[wp.int32], indices: wp.array[wp.int32], a: int, b: int) -> int:
    """The slot of ``(a, b)`` in either order, by searching its row; ``-1`` outside the pattern."""
    r = wp.min(a, b)
    c = wp.max(a, b)
    for s in range(indptr[r], indptr[r + 1]):
        if indices[s] == c:
            return s
    return -1


@wp.func
def scatter_node(b: wp.array[wp.float64], fixed: wp.array[wp.int32], n: int, v: wp.vec3):
    """``b[node n] += v`` on its free components."""
    for k in range(3):
        if fixed[3 * n + k] == 0:
            wp.atomic_add(b, 3 * n + k, wp.float64(v[k]))



def linear_solver(A: SymmetricCSR):
    """cuDSS on CUDA (with nvmath installed; float32: ``A`` is the step's chord matrix, an approximation the line
    search corrects), else SciPy; factorised on the solve after ``invalidate()``."""
    from ..linear import sparse_solver

    return sparse_solver(A, refactorize=False, single=True)
