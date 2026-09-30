import weakref
from typing import Any

import warp as wp

from ..system import SymmetricCSR
from .base import LinearSolverBase

try:
    from nvmath.bindings import cudss

    HAS_CUDSS = True
except ImportError:
    HAS_CUDSS = False

# cudaDataType codes.
_CUDA_R_32F, _CUDA_R_64F, _CUDA_R_32I = 0, 1, 10


@wp.kernel
def _convert(src: wp.array[Any], dst: wp.array[Any]):
    """Copy between the solver's float32 vectors and the matrix's dtype."""
    i = wp.tid()
    dst[i] = type(dst[i])(src[i])


class CudssSolver(LinearSolverBase):
    """LDL^T on the upper view, driven through the raw cuDSS bindings on Warp's own arrays.

    cuDSS reads the CSR arrays, the right-hand side and the solution in place: no
    per-solve allocation or copy. The factorisation runs in the matrix's dtype
    (``A.dtype``); for float32 the caller's ``b`` and ``x`` are used directly, for float64
    they are converted through two buffers of that dtype. The first :meth:`solve` runs
    reordering and symbolic analysis; after that every solve is refactorise and solve, all
    on Warp's current stream.
    """

    def __init__(self, A: SymmetricCSR, device: wp.DeviceLike) -> None:
        if not HAS_CUDSS:
            raise ImportError("CudssSolver requires nvmath (install the 'gpu' extra)")
        super().__init__(A, device)
        self._value_type = {wp.float32: _CUDA_R_32F, wp.float64: _CUDA_R_64F}[A.dtype]
        self._direct = A.dtype == wp.float32  # b and x need no conversion
        if not self._direct:
            self._b = wp.zeros(A.n, dtype=A.dtype, device=self.device)
            self._x = wp.zeros(A.n, dtype=A.dtype, device=self.device)
        self._handle = None

    def _setup(self, b: wp.array, x: wp.array) -> None:
        A = self.A
        with wp.ScopedDevice(self.device):
            self._handle = cudss.create()
            self._config = cudss.config_create()
            self._data = cudss.data_create(self._handle)
            self._A = cudss.matrix_create_csr(
                A.n, A.n, A.nnz, A.indptr.ptr, 0, A.indices.ptr, A.vals.ptr,
                _CUDA_R_32I, _CUDA_R_32I, self._value_type,
                cudss.MatrixType.SYMMETRIC, cudss.MatrixViewType.UPPER, cudss.IndexBase.ZERO,
            )
            self._bm = cudss.matrix_create_dn(A.n, 1, A.n, b.ptr, self._value_type, cudss.Layout.COL_MAJOR)
            self._xm = cudss.matrix_create_dn(A.n, 1, A.n, x.ptr, self._value_type, cudss.Layout.COL_MAJOR)
            self._execute(cudss.Phase.ANALYSIS)
        weakref.finalize(self, _destroy, self._handle, self._config, self._data, (self._xm, self._bm, self._A))

    def _execute(self, phase) -> None:
        cudss.set_stream(self._handle, self.device.stream.cuda_stream)
        cudss.execute(self._handle, phase, self._config, self._data, self._A, self._xm, self._bm)

    def solve(self, b: wp.array, x: wp.array) -> None:
        if self._direct:
            rhs, sol = b, x
        else:
            rhs, sol = self._b, self._x
            wp.launch(_convert, dim=self.A.n, inputs=[b], outputs=[rhs], device=self.device)
        if self._handle is None:
            self._setup(rhs, sol)
        elif self._direct:  # host-side pointer updates, in case the caller's arrays changed
            cudss.matrix_set_values(self._bm, rhs.ptr)
            cudss.matrix_set_values(self._xm, sol.ptr)
        with wp.ScopedDevice(self.device):
            self._execute(cudss.Phase.FACTORIZATION)
            self._execute(cudss.Phase.SOLVE)
        if not self._direct:
            wp.launch(_convert, dim=self.A.n, inputs=[sol], outputs=[x], device=self.device)


def _destroy(handle, config, data, matrices) -> None:
    for m in matrices:
        cudss.matrix_destroy(m)
    cudss.data_destroy(handle, data)
    cudss.config_destroy(config)
    cudss.destroy(handle)
