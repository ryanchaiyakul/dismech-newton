"""Sparse matrices on the device and their direct solves: cuDSS on CUDA (the 'gpu' extra), SciPy otherwise."""

import warnings
import weakref
from typing import Any

import numpy as np
import scipy.sparse as sp
import warp as wp
from scipy.sparse.linalg import splu

try:
    from nvmath.bindings import cudss
except ImportError:
    cudss = None

_CUDA_R_32F, _CUDA_R_64F, _CUDA_R_32I = 0, 1, 10  # cudaDataType codes


class _CSR:
    """``n``, ``indptr``, ``indices`` and float64 ``vals`` on the device."""

    @property
    def nnz(self) -> int:
        return self.vals.shape[0]

    def to_scipy(self) -> sp.csr_matrix:
        """The stored entries on the host."""
        return sp.csr_matrix((self.vals.numpy(), self.indices.numpy(), self.indptr.numpy()), shape=(self.n, self.n))


class SymmetricCSR(_CSR):
    """Upper-triangle float64 CSR; the pattern is the diagonal and every DOF pair within a row of each
    ``(count, width)`` DOF array of ``stencils``."""

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

    def entries(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Host ``(rows, cols, slot)`` of the full matrix's entries: the upper triangle mirrored."""
        indptr, indices = self.pattern
        rows = np.repeat(np.arange(self.n), np.diff(indptr))
        off = indices != rows
        return (np.concatenate([rows, indices[off]]), np.concatenate([indices, rows[off]]),
                np.concatenate([np.arange(len(indices)), np.flatnonzero(off)]))

    def to_scipy(self) -> sp.csr_matrix:
        """The full symmetric matrix on the host."""
        upper = super().to_scipy()
        return (upper + upper.T - sp.diags(upper.diagonal())).tocsr()


class GeneralCSR(_CSR):
    """CSR of a general float64 matrix (the pattern of ``A``, its values on the device)."""

    def __init__(self, A: sp.spmatrix, device) -> None:
        A = sp.csr_matrix(A)
        A.sum_duplicates()
        A.sort_indices()
        self.n = A.shape[0]
        self.indptr = wp.array(A.indptr.astype(np.int32), dtype=wp.int32, device=device)
        self.indices = wp.array(A.indices.astype(np.int32), dtype=wp.int32, device=device)
        self.vals = wp.array(A.data.astype(np.float64), dtype=wp.float64, device=device)


def _pairs(d: np.ndarray, n: int) -> np.ndarray:
    i, j = np.triu_indices(d.shape[1])
    a, b = d[:, i].astype(np.int64), d[:, j].astype(np.int64)
    return np.minimum(a, b) * n + np.maximum(a, b)


@wp.func
def triu_slot(a: int, c: int, D: int) -> int:
    """The index of the pair ``(a, c)``, ``a <= c``, among ``D`` DOFs in ``np.triu_indices`` order."""
    return a * D - (a * (a - 1)) // 2 + (c - a)


@wp.func
def row_slot(indptr: wp.array[wp.int32], indices: wp.array[wp.int32], a: int, b: int) -> int:
    """The slot of ``(a, b)`` in either order, by searching its row; ``-1`` outside the pattern."""
    r = wp.min(a, b)
    c = wp.max(a, b)
    for s in range(indptr[r], indptr[r + 1]):
        if indices[s] == c:
            return s
    return -1


class CudssSolver:
    """``A x = b`` by cuDSS LDL^T (LU for :class:`GeneralCSR`), analysed on the first solve and factorised on the
    first solve after :meth:`invalidate`; ``single``: in float32 (2.5x faster on a GeForce's float64, for an ``A``
    that only has to be approximate). Graph-capturable."""

    graph_capturable = True

    def __init__(self, A: SymmetricCSR | GeneralCSR, single: bool = False) -> None:
        self.A, self.device = A, A.vals.device
        dtype = wp.float32 if single else wp.float64
        self._type = _CUDA_R_32F if single else _CUDA_R_64F
        self._vals = wp.zeros(A.nnz, dtype=dtype, device=self.device) if single else A.vals
        self._b, self._x = (wp.zeros(A.n, dtype=dtype, device=self.device) for _ in range(2))
        self._handle = None
        self._factored = False

    def _setup(self) -> None:
        A = self.A
        self._handle = cudss.create()
        self._config = cudss.config_create()
        self._data = cudss.data_create(self._handle)
        if isinstance(A, GeneralCSR):
            kind = cudss.MatrixType.GENERAL, cudss.MatrixViewType.FULL
        else:
            kind = cudss.MatrixType.SYMMETRIC, cudss.MatrixViewType.UPPER
        self._A = cudss.matrix_create_csr(A.n, A.n, A.nnz, A.indptr.ptr, 0, A.indices.ptr, self._vals.ptr,
                                          _CUDA_R_32I, _CUDA_R_32I, self._type, *kind, cudss.IndexBase.ZERO)
        self._bm = cudss.matrix_create_dn(A.n, 1, A.n, self._b.ptr, self._type, cudss.Layout.COL_MAJOR)
        self._xm = cudss.matrix_create_dn(A.n, 1, A.n, self._x.ptr, self._type, cudss.Layout.COL_MAJOR)
        self._execute(cudss.Phase.ANALYSIS)
        weakref.finalize(self, _destroy, self._handle, self._config, self._data, (self._xm, self._bm, self._A))

    def _execute(self, phase) -> None:
        if phase == cudss.Phase.FACTORIZATION and self._vals is not self.A.vals:
            wp.launch(_convert, dim=self.A.nnz, inputs=[self.A.vals], outputs=[self._vals], device=self.device)
        cudss.set_stream(self._handle, self.device.stream.cuda_stream)
        cudss.execute(self._handle, phase, self._config, self._data, self._A, self._xm, self._bm)

    def invalidate(self) -> None:
        """``A``'s values changed: refactorise on the next solve."""
        self._factored = False

    def solve(self, b: wp.array, x: wp.array) -> None:
        """``x = A^{-1} b``."""
        wp.launch(_convert, dim=self.A.n, inputs=[b], outputs=[self._b], device=self.device)
        with wp.ScopedDevice(self.device):
            if self._handle is None:
                self._setup()
            if not self._factored:
                self._execute(cudss.Phase.FACTORIZATION)
                self._factored = True
            self._execute(cudss.Phase.SOLVE)
        wp.launch(_convert, dim=self.A.n, inputs=[self._x], outputs=[x], device=self.device)


def _destroy(handle, config, data, matrices) -> None:
    for m in matrices:
        cudss.matrix_destroy(m)
    cudss.data_destroy(handle, data)
    cudss.config_destroy(config)
    cudss.destroy(handle)


class ScipySolver:
    """``A x = b`` by SuperLU on the host, in float64; factorised on the first solve after :meth:`invalidate`. Not
    graph-capturable."""

    graph_capturable = False

    def __init__(self, A: SymmetricCSR | GeneralCSR) -> None:
        self.A = A
        self._lu = None
        if A.vals.device.is_cuda:
            warnings.warn("Solving a CUDA system with SciPy (nvmath is not installed: the 'gpu' extra): every solve "
                          "copies to the host and synchronizes, and CUDA graph capture is unavailable.",
                          RuntimeWarning, stacklevel=3)

    def invalidate(self) -> None:
        self._lu = None

    def solve(self, b: wp.array, x: wp.array) -> None:
        """``x = A^{-1} b``."""
        if self.A.vals.device.is_capturing:
            raise RuntimeError("ScipySolver solves on the host and cannot be captured in a graph")
        if self._lu is None:
            M = self.A.to_scipy().tocsc()
            try:
                self._lu = splu(M)
            except RuntimeError:  # exactly singular
                self._lu = splu(M + 1.0e-10 * abs(M).max() * sp.identity(self.A.n, format="csc"))
        x.assign(self._lu.solve(b.numpy().astype(np.float64)).astype(wp.dtype_to_numpy(x.dtype)))


def sparse_solver(A: SymmetricCSR | GeneralCSR, single: bool = False) -> CudssSolver | ScipySolver:
    """cuDSS for a CUDA ``A`` (with nvmath installed), else SciPy; ``single``: cuDSS in float32."""
    if A.vals.device.is_cuda and cudss is not None:
        return CudssSolver(A, single)
    return ScipySolver(A)


@wp.kernel
def _convert(src: wp.array[Any], dst: wp.array[Any]):
    i = wp.tid()
    dst[i] = type(dst[i])(src[i])
