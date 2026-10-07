"""Solvers of ``A x = b`` for the matrices of :mod:`~dismech_newton.sparse`.

With ``increment``, ``solve(b, x)`` takes ``x`` as the current iterate and updates it by the correction,
``x += A^-1 (b - A x)``: the same answer, but the float32 solve rounds with the correction, not with ``b``.
ADMM's ``b`` (``M alpha q_pred + ...``) is large and cancels, so without it the solve leaves an iteration-
to-iteration noise floor that can sit above the convergence tolerance on long rods.
"""

import warnings
import weakref
from typing import Any

import numpy as np
import scipy.sparse as sp
import warp as wp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from .sparse import GeneralCSR, SymmetricCSR

try:
    from nvmath.bindings import cudss, cusparse
except ImportError:
    cudss = cusparse = None

_CUDA_R_64F, _CUDA_R_32I = 1, 10  # cudaDataType codes


def csr_entries(A: SymmetricCSR | GeneralCSR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Host ``(rows, cols, slot)`` of every entry of the full matrix, ``slot`` its index in ``A.vals`` (a
    :class:`SymmetricCSR`'s strict upper triangle mirrored after its stored entries)."""
    indptr, indices = A.indptr.numpy(), A.indices.numpy()
    rows = np.repeat(np.arange(A.n), np.diff(indptr))
    slot = np.arange(len(indices))
    if not isinstance(A, SymmetricCSR):
        return rows, indices, slot
    off = indices != rows
    return (np.concatenate([rows, indices[off]]), np.concatenate([indices, rows[off]]),
            np.concatenate([slot, slot[off]]))


class _Residual:
    """``r = b - A x`` in float64, one thread per row, reading ``A.vals`` live (full rows of a symmetric
    upper-triangle CSR through an index map)."""

    def __init__(self, A: SymmetricCSR | GeneralCSR) -> None:
        self.A = A
        n = A.n
        rows, cols, slot = csr_entries(A)
        order = np.lexsort((cols, rows))
        full_indptr = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(np.bincount(rows, minlength=n), out=full_indptr[1:])
        dev = A.vals.device
        self.indptr = wp.array(full_indptr, dtype=wp.int32, device=dev)
        self.cols = wp.array(cols[order].astype(np.int32), dtype=wp.int32, device=dev)
        self.slot = wp.array(slot[order].astype(np.int32), dtype=wp.int32, device=dev)

    def __call__(self, b: wp.array, x: wp.array, r: wp.array) -> None:
        wp.launch(_residual_kernel, dim=self.A.n, inputs=[self.indptr, self.cols, self.slot, self.A.vals, b, x],
                  outputs=[r], device=self.A.vals.device)


class CudssSolver:
    """``A x = b`` by cuDSS LDL^T (LU for :class:`GeneralCSR`); without ``refactorize``, factorised once. Capturable,
    but not inside a device-side loop (a conditional graph node fails to instantiate)."""

    graph_capturable = True
    loop_capturable = False

    def __init__(self, A: SymmetricCSR | GeneralCSR, refactorize: bool = True, increment: bool = False) -> None:
        if cudss is None:
            raise ImportError("CudssSolver requires nvmath (install the 'gpu' extra)")
        if not A.vals.device.is_cuda:
            raise ValueError(f"CudssSolver needs a CUDA matrix, got one on {A.vals.device}")
        self.A = A
        self.device = A.vals.device
        self.refactorize = refactorize
        self._residual = _Residual(A) if increment else None
        self._b = wp.zeros(A.n, dtype=wp.float64, device=self.device)
        self._x = wp.zeros(A.n, dtype=wp.float64, device=self.device)
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
        self._A = cudss.matrix_create_csr(A.n, A.n, A.nnz, A.indptr.ptr, 0, A.indices.ptr, A.vals.ptr, _CUDA_R_32I,
                                          _CUDA_R_32I, _CUDA_R_64F, *kind, cudss.IndexBase.ZERO)
        self._bm = cudss.matrix_create_dn(A.n, 1, A.n, self._b.ptr, _CUDA_R_64F, cudss.Layout.COL_MAJOR)
        self._xm = cudss.matrix_create_dn(A.n, 1, A.n, self._x.ptr, _CUDA_R_64F, cudss.Layout.COL_MAJOR)
        self._execute(cudss.Phase.ANALYSIS)
        weakref.finalize(self, _destroy, self._handle, self._config, self._data, (self._xm, self._bm, self._A))

    def _execute(self, phase) -> None:
        cudss.set_stream(self._handle, self.device.stream.cuda_stream)
        cudss.execute(self._handle, phase, self._config, self._data, self._A, self._xm, self._bm)

    def invalidate(self) -> None:
        """``A``'s values changed: refactorise on the next solve."""
        self._factored = False

    def solve(self, b: wp.array, x: wp.array, reset: tuple[wp.array, wp.array] | None = None) -> None:
        """``x = A^{-1} b``; ``reset = (dst, src)`` also copies ``src`` into ``dst``."""
        if self._residual is not None:
            self._residual(b, x, self._b)
        else:
            wp.launch(_convert, dim=self.A.n, inputs=[b], outputs=[self._b], device=self.device)
        with wp.ScopedDevice(self.device):
            if self._handle is None:
                self._setup()
            if self.refactorize or not self._factored:
                self._execute(cudss.Phase.FACTORIZATION)
                self._factored = True
            self._execute(cudss.Phase.SOLVE)
        wp.launch(_accumulate, dim=self.A.n, inputs=[self._x, int(self._residual is not None)], outputs=[x],
                  device=self.device)
        if reset is not None:
            wp.copy(*reset)


def _destroy(handle, config, data, matrices) -> None:
    for m in matrices:
        cudss.matrix_destroy(m)
    cudss.data_destroy(handle, data)
    cudss.config_destroy(config)
    cudss.destroy(handle)


class ScipySolver:
    """``A x = b`` by SuperLU on the CPU, in float64; not graph-capturable."""

    graph_capturable = loop_capturable = False

    def __init__(self, A: SymmetricCSR | GeneralCSR, refactorize: bool = True, increment: bool = False) -> None:
        self.A = A
        self.device = A.vals.device
        self.refactorize = refactorize
        self.increment = increment
        self._lu = None
        self._M = None
        if self.device.is_cuda:
            why = "nvmath is not installed (install the 'gpu' extra)" if cudss is None else "it was requested"
            warnings.warn(
                f"Solving a CUDA system with SciPy because {why}: every solve copies to the host and "
                "synchronizes, and CUDA graph capture is unavailable. cuDSS keeps it on the GPU.",
                RuntimeWarning, stacklevel=2,
            )

    def invalidate(self) -> None:
        self._lu = None

    def _factorize(self) -> None:
        A = self.A
        if isinstance(A, GeneralCSR):
            M = sp.csr_matrix((A.vals.numpy(), A.indices.numpy(), A.indptr.numpy()), shape=(A.n, A.n))
        else:
            M = A.to_scipy()
        self._M = M.tocsr()
        M = M.tocsc()
        try:
            self._lu = splu(M)
        except RuntimeError:  # exactly singular
            self._lu = splu((M + 1.0e-10 * abs(M).max() * sp.identity(A.n, format="csc")).tocsc())

    def solve(self, b: wp.array, x: wp.array, reset: tuple[wp.array, wp.array] | None = None) -> None:
        """``x = A^{-1} b``; ``reset = (dst, src)`` also copies ``src`` into ``dst``."""
        if self.device.is_capturing:
            raise RuntimeError("ScipySolver solves on the host and cannot be captured in a graph")
        if self.refactorize or self._lu is None:
            self._factorize()
        rhs = b.numpy().astype(np.float64)
        if self.increment:
            x0 = x.numpy().astype(np.float64)
            out = x0 + self._lu.solve(rhs - self._M @ x0)
        else:
            out = self._lu.solve(rhs)
        x.assign(out.astype(wp.dtype_to_numpy(x.dtype)))
        if reset is not None:
            wp.copy(*reset)


def sparse_solver(A: SymmetricCSR | GeneralCSR, refactorize: bool = True,
                  increment: bool = False) -> CudssSolver | ScipySolver:
    """cuDSS for a CUDA ``A`` (with nvmath installed), else SciPy."""
    if A.vals.device.is_cuda and cudss is not None:
        return CudssSolver(A, refactorize, increment)
    return ScipySolver(A, refactorize, increment)


class BlockInverseSolver:
    """``A x = b`` for a constant ``A`` by one dense inverse per connected component."""

    max_block = 4096  # largest block, and largest total of stored entries, that fits() accepts
    max_entries = 1 << 26
    graph_capturable = loop_capturable = True

    def __init__(self, A: SymmetricCSR, increment: bool = False, H: sp.spmatrix | None = None) -> None:
        """``H``: ``A.to_scipy()``, if already at hand."""
        self.A = A
        self.device = A.vals.device
        H = A.to_scipy() if H is None else H
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
        self._residual = _Residual(A) if increment else None
        self._r = wp.zeros(A.n if increment else 0, dtype=wp.float64, device=self.device)

    @classmethod
    def fits(cls, A: SymmetricCSR, H: sp.spmatrix | None = None) -> bool:
        """Whether ``A``'s blocks fit :attr:`max_block` and :attr:`max_entries`."""
        size = np.bincount(connected_components(A.to_scipy() if H is None else H, directed=False)[1])
        return int(size.max()) <= cls.max_block and int(np.sum(size.astype(np.int64) ** 2)) <= cls.max_entries

    def solve(self, b: wp.array, x: wp.array, reset: tuple[wp.array, wp.array] | None = None) -> None:
        """``x = A^{-1} b``; ``reset = (dst, src)`` also copies ``src`` into ``dst``."""
        dst, src = reset if reset is not None else (self._empty, self._empty)
        increment = self._residual is not None
        if increment:
            self._residual(b, x, self._r)
        wp.launch(
            _block_inverse_kernel,
            dim=self.A.n,
            inputs=[self.perm, self.slot_block, self.block_start, self.block_size, self.block_offset, self.inv, b,
                    self._r, int(increment), src],
            outputs=[x, dst],
            device=self.device,
        )


class TridiagonalSolver:
    """``A x = b`` for a constant ``A`` whose connected components are chains (one rod's x, y, z or twist DOFs),
    by batched cuSPARSE ``gtsv2StridedBatch`` (cyclic + parallel cyclic reduction): ``O(n)`` per solve.

    The chains are padded to the longest with identity rows; single-DOF components (fixed DOFs) are solved
    apart, in the scatter. With ``increment`` the residual is formed in the gather (no extra launch).
    """

    graph_capturable = loop_capturable = True

    def __init__(self, A: SymmetricCSR, increment: bool = False, H: sp.spmatrix | None = None) -> None:
        """``H``: ``A.to_scipy()``, if already at hand."""
        if cusparse is None:
            raise ImportError("TridiagonalSolver requires nvmath (install the 'gpu' extra)")
        if not A.vals.device.is_cuda:
            raise ValueError(f"TridiagonalSolver needs a CUDA matrix, got one on {A.vals.device}")
        self.A = A
        self.device = dev = A.vals.device
        self.increment = increment
        chains, singles = _chains(A.to_scipy() if H is None else H)
        self.batch = len(chains)
        self.m = m = max([3] + [len(c) for c, _ in chains])  # gtsv2 needs m >= 3
        total = max(self.batch * m, 1)
        slot_dof = np.full(total, -1, dtype=np.int32)
        dl, d, du = np.zeros(total), np.ones(total), np.zeros(total)
        for k, (chain, (lo, mid, up)) in enumerate(chains):
            s = k * m + np.arange(len(chain))
            slot_dof[s], dl[s], d[s], du[s] = chain, lo, mid, up
        dof_slot = np.full(A.n, -1, dtype=np.int32)
        dof_slot[slot_dof[slot_dof >= 0]] = np.flatnonzero(slot_dof >= 0)
        single_diag = np.zeros(A.n)
        single_diag[singles[0]] = singles[1]

        def arr(a, dtype):
            return wp.array(a, dtype=dtype, device=dev)

        self.dl, self.d, self.du = (arr(v.astype(np.float32), float) for v in (dl, d, du))  # factor (float32)
        # The residual's copy (float64), read only with increment.
        self.dl64, self.d64, self.du64 = (arr(v if increment else v[:0], wp.float64) for v in (dl, d, du))
        self.slot_dof, self.dof_slot = arr(slot_dof, wp.int32), arr(dof_slot, wp.int32)
        self.single_diag = arr(single_diag, wp.float64)
        self.X = wp.zeros(total, dtype=float, device=dev)
        self._empty = wp.zeros(0, dtype=float, device=dev)
        self._handle = None
        self._buffer = None

    @classmethod
    def fits(cls, A: SymmetricCSR, H: sp.spmatrix | None = None) -> bool:
        """Whether this solver can take ``A``: CUDA, nvmath, and every component a chain."""
        if cusparse is None or not A.vals.device.is_cuda:
            return False
        H = A.to_scipy() if H is None else H
        off = sp.triu(H, k=1, format="coo")
        off.eliminate_zeros()
        degree = np.bincount(np.concatenate([off.row, off.col]), minlength=A.n)
        n_blocks = connected_components(H, directed=False)[0]
        return degree.max(initial=0) <= 2 and off.nnz == A.n - n_blocks  # a forest of paths

    def _setup(self) -> None:
        self._handle = cusparse.create()
        weakref.finalize(self, cusparse.destroy, self._handle)
        size = cusparse.sgtsv2strided_batch_buffer_size_ext(self._handle, self.m, self.dl.ptr, self.d.ptr,
                                                             self.du.ptr, self.X.ptr, self.batch, self.m)
        self._buffer = wp.zeros(max(int(size), 1), dtype=wp.uint8, device=self.device)

    def solve(self, b: wp.array, x: wp.array, reset: tuple[wp.array, wp.array] | None = None) -> None:
        """``x = A^{-1} b``; ``reset = (dst, src)`` also copies ``src`` into ``dst``."""
        dst, src = reset if reset is not None else (self._empty, self._empty)
        inc = int(self.increment)
        if self.batch:
            if self._handle is None:
                self._setup()
            wp.launch(_tridiagonal_gather_kernel, dim=self.X.shape[0],
                      inputs=[self.slot_dof, self.dl64, self.d64, self.du64, b, x, inc], outputs=[self.X],
                      device=self.device)
            cusparse.set_stream(self._handle, self.device.stream.cuda_stream)
            cusparse.sgtsv2strided_batch(self._handle, self.m, self.dl.ptr, self.d.ptr, self.du.ptr, self.X.ptr,
                                         self.batch, self.m, self._buffer.ptr)
        wp.launch(_tridiagonal_scatter_kernel, dim=self.A.n,
                  inputs=[self.dof_slot, self.X, self.single_diag, b, inc, src], outputs=[x, dst],
                  device=self.device)


def _chains(H: sp.spmatrix):
    """The components of ``H`` in chain order with ``(dl, d, du)``, and the single-DOF ones ``(dofs, diag)``."""
    H = H.tocsr()
    n_blocks, label = connected_components(H, directed=False)
    off = H.copy()
    off.setdiag(0.0)
    off.eliminate_zeros()
    degree = np.diff(off.indptr)
    order = np.argsort(label, kind="stable")
    bounds = np.searchsorted(label[order], np.arange(n_blocks + 1))
    chains, single = [], []
    for k in range(n_blocks):
        members = order[bounds[k] : bounds[k + 1]]
        if len(members) == 1:
            single.append(members[0])
            continue
        ends = members[degree[members] == 1]
        if len(ends) != 2 or degree[members].max() > 2:
            raise ValueError("TridiagonalSolver needs every component of A to be a chain")
        chain, prev = [ends[0]], -1
        while len(chain) < len(members):
            row = off.indices[off.indptr[chain[-1]] : off.indptr[chain[-1] + 1]]
            prev, nxt = chain[-1], row[row != prev][0]
            chain.append(nxt)
        chain = np.asarray(chain)
        up = np.asarray(H[chain[:-1], chain[1:]]).ravel()
        chains.append((chain, (np.concatenate([[0.0], up]), np.asarray(H[chain, chain]).ravel(),
                               np.concatenate([up, [0.0]]))))
    single = np.asarray(single, dtype=np.int64)
    return chains, (single, H.diagonal()[single])


@wp.kernel
def _convert(src: wp.array[Any], dst: wp.array[Any]):
    i = wp.tid()
    dst[i] = type(dst[i])(src[i])


@wp.kernel
def _accumulate(src: wp.array[wp.float64], increment: int, dst: wp.array[float]):
    """``dst = src``, or ``dst += src`` with ``increment``."""
    i = wp.tid()
    v = float(src[i])
    if increment != 0:
        v = dst[i] + v
    dst[i] = v


@wp.kernel
def _residual_kernel(indptr: wp.array[wp.int32], cols: wp.array[wp.int32], slot: wp.array[wp.int32],
                     vals: wp.array[wp.float64], b: wp.array[float], x: wp.array[float], r: wp.array[wp.float64]):
    i = wp.tid()
    acc = wp.float64(b[i])
    for k in range(indptr[i], indptr[i + 1]):
        acc -= vals[slot[k]] * wp.float64(x[cols[k]])
    r[i] = acc


@wp.kernel
def _block_inverse_kernel(
    perm: wp.array[wp.int32],
    slot_block: wp.array[wp.int32],
    block_start: wp.array[wp.int32],
    block_size: wp.array[wp.int32],
    block_offset: wp.array[wp.int32],
    inv: wp.array[float],
    b: wp.array[float],
    r: wp.array[wp.float64],
    increment: int,
    reset_src: wp.array[float],
    # outputs
    x: wp.array[float],
    reset_dst: wp.array[float],
):
    """One thread per DOF in block order: ``x = inv b``, or ``x += inv r`` with ``increment``; also
    ``reset_dst = reset_src`` if given."""
    p = wp.tid()
    k = slot_block[p]
    s = block_start[k]
    n = block_size[k]
    col = block_offset[k] + p - s  # the inverse is symmetric: column p - s of the block
    acc = float(0.0)
    i = perm[p]
    if increment != 0:
        for j in range(n):
            acc += inv[col + j * n] * float(r[perm[s + j]])
        acc += x[i]
    else:
        for j in range(n):
            acc += inv[col + j * n] * b[perm[s + j]]
    x[i] = acc
    if reset_dst.shape[0] > 0:
        reset_dst[i] = reset_src[i]


@wp.kernel
def _tridiagonal_gather_kernel(slot_dof: wp.array[wp.int32], dl: wp.array[wp.float64], d: wp.array[wp.float64],
                               du: wp.array[wp.float64], b: wp.array[float], x: wp.array[float], increment: int,
                               X: wp.array[float]):
    """The padded right-hand side: ``b``, or with ``increment`` the residual ``b - A x`` (float64)."""
    s = wp.tid()
    i = slot_dof[s]
    if i < 0:
        X[s] = 0.0
        return
    acc = wp.float64(b[i])
    if increment != 0:
        acc -= d[s] * wp.float64(x[i])
        if dl[s] != wp.float64(0.0):
            acc -= dl[s] * wp.float64(x[slot_dof[s - 1]])
        if du[s] != wp.float64(0.0):
            acc -= du[s] * wp.float64(x[slot_dof[s + 1]])
    X[s] = float(acc)


@wp.kernel
def _tridiagonal_scatter_kernel(dof_slot: wp.array[wp.int32], X: wp.array[float], single_diag: wp.array[wp.float64],
                                b: wp.array[float], increment: int, reset_src: wp.array[float],
                                x: wp.array[float], reset_dst: wp.array[float]):
    """``x`` from the padded solution (single-DOF components: ``b / diag``); also the reset copy."""
    i = wp.tid()
    s = dof_slot[i]
    if s >= 0:
        v = X[s]
        if increment != 0:
            v = x[i] + v
    else:
        r = wp.float64(b[i])
        if increment != 0:
            r -= single_diag[i] * wp.float64(x[i])
        v = float(r / single_diag[i])
        if increment != 0:
            v = x[i] + v
    x[i] = v
    if reset_dst.shape[0] > 0:
        reset_dst[i] = reset_src[i]
