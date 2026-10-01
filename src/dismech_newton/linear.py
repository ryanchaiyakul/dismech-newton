"""The sparse symmetric matrix and its GPU solvers.

:class:`SymmetricCSR` stores the upper triangle (float64) with a fixed pattern; kernels add
straight into ``vals``, finding the slot of ``(row, col)`` with :func:`csr_slot`.
"""

import weakref
from typing import Any

import numpy as np
import scipy.sparse as sp
import warp as wp
from scipy.sparse.csgraph import connected_components

try:
    from nvmath.bindings import cudss
except ImportError:
    cudss = None


class SymmetricCSR:
    """Upper-triangle CSR of a symmetric ``n x n`` float64 matrix.

    The pattern covers the diagonal and every pair of DOFs in a row of ``dofs`` (``(count, k)``).
    """

    def __init__(self, n: int, dofs: np.ndarray, device) -> None:
        keys = [np.arange(n, dtype=np.int64) * (n + 1)]
        i, j = np.triu_indices(dofs.shape[1])
        a, b = dofs[:, i], dofs[:, j]
        keys.append((np.minimum(a, b) * n + np.maximum(a, b)).ravel())
        keys = np.unique(np.concatenate(keys))
        indptr = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(np.bincount(keys // n, minlength=n), out=indptr[1:])
        self._set(n, indptr, (keys % n).astype(np.int32), np.zeros(len(keys)), device)

    def _set(self, n, indptr, indices, vals, device) -> None:
        self.n = n
        self.indptr = wp.array(indptr, dtype=wp.int32, device=device)
        self.indices = wp.array(indices, dtype=wp.int32, device=device)
        self.vals = wp.array(vals, dtype=wp.float64, device=device)

    @classmethod
    def from_scipy(cls, H: sp.spmatrix, device) -> "SymmetricCSR":
        """The upper triangle of the symmetric ``H``, pattern and values."""
        H = sp.triu(H, format="csr")
        H.sort_indices()
        A = cls.__new__(cls)
        A._set(H.shape[0], H.indptr.astype(np.int32), H.indices.astype(np.int32), H.data, device)
        return A

    def to_scipy(self) -> sp.csr_matrix:
        """The full symmetric matrix on the host (both triangles)."""
        upper = sp.csr_matrix((self.vals.numpy(), self.indices.numpy(), self.indptr.numpy()), shape=(self.n, self.n))
        return (upper + upper.T - sp.diags(upper.diagonal())).tocsr()

    @property
    def nnz(self) -> int:
        return self.vals.shape[0]


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


# -- cuDSS --------------------------------------------------------------------------------

_CUDA_R_64F, _CUDA_R_32I = 1, 10  # cudaDataType codes


@wp.kernel
def _convert(src: wp.array[Any], dst: wp.array[Any]):
    i = wp.tid()
    dst[i] = type(dst[i])(src[i])


class CudssSolver:
    """``A x = b`` by LDL^T of ``A``'s upper triangle with cuDSS, on Warp's arrays and stream.

    The first solve runs the analysis. With ``refactorize`` every solve refactorises (the values of
    ``A`` change between solves); without, ``A`` is factorised once and solves are triangular only.
    """

    def __init__(self, A: SymmetricCSR, refactorize: bool = True) -> None:
        if cudss is None:
            raise ImportError("CudssSolver requires nvmath (install the 'gpu' extra)")
        self.A = A
        self.device = A.vals.device
        self.refactorize = refactorize
        self._b = wp.zeros(A.n, dtype=wp.float64, device=self.device)
        self._x = wp.zeros(A.n, dtype=wp.float64, device=self.device)
        self._handle = None
        self._factored = False

    def _setup(self) -> None:
        A = self.A
        self._handle = cudss.create()
        self._config = cudss.config_create()
        self._data = cudss.data_create(self._handle)
        self._A = cudss.matrix_create_csr(
            A.n, A.n, A.nnz, A.indptr.ptr, 0, A.indices.ptr, A.vals.ptr, _CUDA_R_32I, _CUDA_R_32I, _CUDA_R_64F,
            cudss.MatrixType.SYMMETRIC, cudss.MatrixViewType.UPPER, cudss.IndexBase.ZERO,
        )
        self._bm = cudss.matrix_create_dn(A.n, 1, A.n, self._b.ptr, _CUDA_R_64F, cudss.Layout.COL_MAJOR)
        self._xm = cudss.matrix_create_dn(A.n, 1, A.n, self._x.ptr, _CUDA_R_64F, cudss.Layout.COL_MAJOR)
        self._execute(cudss.Phase.ANALYSIS)
        weakref.finalize(self, _destroy, self._handle, self._config, self._data, (self._xm, self._bm, self._A))

    def _execute(self, phase) -> None:
        cudss.set_stream(self._handle, self.device.stream.cuda_stream)
        cudss.execute(self._handle, phase, self._config, self._data, self._A, self._xm, self._bm)

    def solve(self, b: wp.array, x: wp.array, reset: tuple[wp.array, wp.array] | None = None) -> None:
        """``x = A^{-1} b``; ``reset = (dst, src)`` also copies ``src`` into ``dst``."""
        wp.launch(_convert, dim=self.A.n, inputs=[b], outputs=[self._b], device=self.device)
        with wp.ScopedDevice(self.device):
            if self._handle is None:
                self._setup()
            if self.refactorize or not self._factored:
                self._execute(cudss.Phase.FACTORIZATION)
                self._factored = True
            self._execute(cudss.Phase.SOLVE)
        wp.launch(_convert, dim=self.A.n, inputs=[self._x], outputs=[x], device=self.device)
        if reset is not None:
            wp.copy(*reset)


def _destroy(handle, config, data, matrices) -> None:
    for m in matrices:
        cudss.matrix_destroy(m)
    cudss.data_destroy(handle, data)
    cudss.config_destroy(config)
    cudss.destroy(handle)


# -- dense block inverse ------------------------------------------------------------------


@wp.kernel
def _block_inverse_kernel(
    perm: wp.array[wp.int32],
    slot_block: wp.array[wp.int32],
    block_start: wp.array[wp.int32],
    block_size: wp.array[wp.int32],
    block_offset: wp.array[wp.int32],
    inv: wp.array[float],
    b: wp.array[float],
    reset_src: wp.array[float],
    # outputs
    x: wp.array[float],
    reset_dst: wp.array[float],
):
    """``x = A^{-1} b``, one thread per DOF in block order; also ``reset_dst = reset_src`` if given."""
    p = wp.tid()
    k = slot_block[p]
    s = block_start[k]
    n = block_size[k]
    col = block_offset[k] + p - s  # the inverse is symmetric: column p - s of the block
    acc = float(0.0)
    for j in range(n):
        acc += inv[col + j * n] * b[perm[s + j]]
    i = perm[p]
    x[i] = acc
    if reset_dst.shape[0] > 0:
        reset_dst[i] = reset_src[i]


class BlockInverseSolver:
    """``A x = b`` for a constant ``A`` through one dense inverse per connected component of ``A``.

    Suits matrices of many small blocks, like the ADMM global matrix (the x, y, z and twist DOFs of
    every rod are separate blocks). Identical blocks share one inverse; a solve is one kernel.
    """

    max_block = 4096  # largest block, and largest total of stored entries, that fits() accepts
    max_entries = 1 << 26

    def __init__(self, A: SymmetricCSR) -> None:
        self.A = A
        self.device = A.vals.device
        H = A.to_scipy()
        n_blocks, label = connected_components(H, directed=False)
        perm = np.argsort(label, kind="stable")
        size = np.bincount(label, minlength=n_blocks)
        start = np.concatenate(([0], np.cumsum(size)[:-1]))
        offset = np.empty(n_blocks, dtype=np.int64)
        inverses, seen, total = [], {}, 0
        for k in range(n_blocks):
            dofs = perm[start[k] : start[k] + size[k]]
            B = H[dofs][:, dofs].toarray()
            key = B.tobytes()
            if key not in seen:
                seen[key] = total
                inverses.append(np.linalg.inv(B).astype(np.float32).ravel())
                total += B.size
            offset[k] = seen[key]

        def ints(a):
            return wp.array(np.asarray(a).astype(np.int32), dtype=wp.int32, device=self.device)

        self.perm = ints(perm)
        self.slot_block = ints(np.repeat(np.arange(n_blocks), size))
        self.block_start = ints(start)
        self.block_size = ints(size)
        self.block_offset = ints(offset)
        self.inv = wp.array(np.concatenate(inverses), dtype=float, device=self.device)
        self._empty = wp.zeros(0, dtype=float, device=self.device)

    @classmethod
    def fits(cls, A: SymmetricCSR) -> bool:
        """Whether ``A``'s blocks are within :attr:`max_block` and :attr:`max_entries`."""
        size = np.bincount(connected_components(A.to_scipy(), directed=False)[1])
        return int(size.max()) <= cls.max_block and int(np.sum(size.astype(np.int64) ** 2)) <= cls.max_entries

    def solve(self, b: wp.array, x: wp.array, reset: tuple[wp.array, wp.array] | None = None) -> None:
        """``x = A^{-1} b``; ``reset = (dst, src)`` also copies ``src`` into ``dst`` in the same kernel."""
        dst, src = reset if reset is not None else (self._empty, self._empty)
        wp.launch(
            _block_inverse_kernel,
            dim=self.A.n,
            inputs=[self.perm, self.slot_block, self.block_start, self.block_size, self.block_offset, self.inv, b, src],
            outputs=[x, dst],
            device=self.device,
        )
