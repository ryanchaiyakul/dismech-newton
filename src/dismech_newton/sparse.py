"""Device CSR matrices; kernels add into ``vals`` at :func:`csr_slot`."""

import numpy as np
import scipy.sparse as sp
import warp as wp


class _CSR:
    """The CSR arrays of an ``n x n`` float64 matrix."""

    def _set(self, n, indptr, indices, vals, device) -> None:
        self.n = n
        self.indptr = wp.array(indptr, dtype=wp.int32, device=device)
        self.indices = wp.array(indices, dtype=wp.int32, device=device)
        self.vals = wp.array(vals, dtype=wp.float64, device=device)

    @property
    def nnz(self) -> int:
        return self.vals.shape[0]


class SymmetricCSR(_CSR):
    """Upper-triangle CSR; the pattern is the diagonal and every DOF pair in a row of ``dofs``."""

    def __init__(self, n: int, dofs: np.ndarray, device) -> None:
        keys = [np.arange(n, dtype=np.int64) * (n + 1)]
        i, j = np.triu_indices(dofs.shape[1])
        a, b = dofs[:, i], dofs[:, j]
        keys.append((np.minimum(a, b) * n + np.maximum(a, b)).ravel())
        keys = np.unique(np.concatenate(keys))
        indptr = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(np.bincount(keys // n, minlength=n), out=indptr[1:])
        self._set(n, indptr, (keys % n).astype(np.int32), np.zeros(len(keys)), device)

    @classmethod
    def from_scipy(cls, H: sp.spmatrix, device) -> "SymmetricCSR":
        """The upper triangle of the symmetric ``H``."""
        H = sp.triu(H, format="csr")
        H.sort_indices()
        A = cls.__new__(cls)
        A._set(H.shape[0], H.indptr.astype(np.int32), H.indices.astype(np.int32), H.data, device)
        return A

    def to_scipy(self) -> sp.csr_matrix:
        """The full symmetric matrix on the host."""
        upper = sp.csr_matrix((self.vals.numpy(), self.indices.numpy(), self.indptr.numpy()), shape=(self.n, self.n))
        return (upper + upper.T - sp.diags(upper.diagonal())).tocsr()


class GeneralCSR(_CSR):
    """CSR of a general ``n x n`` float64 matrix."""

    def __init__(self, A: sp.spmatrix, device) -> None:
        A = sp.csr_matrix(A)
        A.sum_duplicates()
        A.sort_indices()
        self._set(A.shape[0], A.indptr.astype(np.int32), A.indices.astype(np.int32), A.data.astype(np.float64), device)


@wp.func
def csr_slot(indptr: wp.array[wp.int32], indices: wp.array[wp.int32], row: int, col: int) -> int:
    """Index of ``(row, col)``, which must be in the pattern."""
    lo = indptr[row]
    hi = indptr[row + 1] - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if indices[mid] < col:
            lo = mid + 1
        else:
            hi = mid
    return lo
