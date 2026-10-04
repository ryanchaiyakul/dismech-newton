"""Regularized Stokeslet segments (Cortez 2018): the viscous drag of the flagella example.

A port of ``RegularizedStokeslet.cpp`` (https://github.com/StructuresComp/rod-contact-sim; Tong, Choi et al.,
arXiv:2205.10309, Appendix A). Each step builds the dense mobility ``U = A f`` at the nodes, solves
``8 pi eta A f = U`` for the nodes' velocities (no slip; one single-precision Cholesky, cuSOLVER on CUDA,
SciPy on the CPU) and applies ``F = -8 pi eta A^-1 U`` for the step. :meth:`Stokeslets.implicit` takes the
drag backward-Euler instead: the explicit drag flips sign at the tips of a tight bundle once ``dt R / m > 2``.
cuSOLVER does not capture in a CUDA graph, so call :meth:`Stokeslets.compute` eagerly.
"""

import ctypes
import sys

import numpy as np
import scipy.linalg
import warp as wp


def _cusolver_library() -> ctypes.CDLL:
    """The cuSOLVER that CuPy loaded, called directly with buffers allocated once (CuPy's ``posv``
    copies ``A``, allocates and synchronizes on every call)."""
    from cupy.cuda import device as _cupy_device

    _cupy_device.get_cusolver_handle()  # makes CuPy load it
    names = ("cusolver64_12.dll", "cusolver64_11.dll") if sys.platform == "win32" else ("libcusolver.so.12", "libcusolver.so.11")
    for name in names:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        c, p = ctypes.c_int, ctypes.c_void_p
        lib.cusolverDnCreate.argtypes = [ctypes.POINTER(p)]
        lib.cusolverDnSetStream.argtypes = [p, p]
        lib.cusolverDnSpotrf_bufferSize.argtypes = [p, c, c, p, c, ctypes.POINTER(c)]
        lib.cusolverDnSpotrf.argtypes = [p, c, c, p, c, p, c, p]
        lib.cusolverDnSpotrs.argtypes = [p, c, c, c, p, c, p, c, p]
        return lib
    raise OSError("cuSOLVER not found")


_LOWER = 0  # CUBLAS_FILL_MODE_LOWER


def _check(status: int, what: str) -> None:
    if status != 0:
        raise RuntimeError(f"{what} failed with cuSOLVER status {status}")


@wp.func
def _segment_blocks(x: wp.vec3d, y0: wp.vec3d, y1: wp.vec3d, eps: wp.float64):
    """``(M1, M2)``: velocity at ``x`` per unit force density at ``y0`` and ``y1`` (linear along the segment)."""
    x0 = x - y0
    x1 = x - y1
    v = y0 - y1
    l = wp.length(v)
    e2 = eps * eps
    r0 = wp.sqrt(wp.dot(x0, x0) + e2)
    r1 = wp.sqrt(wp.dot(x1, x1) + e2)
    one = wp.float64(1.0)
    ll = l * l
    x0v = wp.dot(x0, v)
    x1v = wp.dot(x1, v)
    t0_1 = (wp.log(l * r1 + x1v) - wp.log(l * r0 + x0v)) / l
    t0_3 = -(one / (r1 * (l * r1 + x1v)) - one / (r0 * (l * r0 + x0v)))
    t1_1 = (r1 - r0) / ll - t0_1 * x0v / ll
    t1_3 = -(one / (r1 * ll) - one / (r0 * ll)) - t0_3 * x0v / ll
    t2_3 = -(one / (r1 * ll)) + t0_1 / ll - t1_3 * x0v / ll
    t3_3 = -(one / (r1 * ll)) + wp.float64(2.0) * t1_1 / ll - t2_3 * x0v / ll
    xx = wp.outer(x0, x0)
    xv = wp.outer(x0, v) + wp.outer(v, x0)
    vv = wp.outer(v, v)
    eye = wp.identity(3, dtype=wp.float64)
    m2 = (t1_1 + e2 * t1_3) * eye + t1_3 * xx + t2_3 * xv + t3_3 * vv
    m1 = (t0_1 + e2 * t0_3) * eye + t0_3 * xx + t1_3 * xv + t2_3 * vv - m2
    return m1, m2


@wp.kernel
def _assemble_kernel(
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    nv: int,
    eps: wp.float64,
    s: wp.array[float],
    scale: float,
    shift: float,
    # outputs
    A: wp.array2d[float],
    U: wp.array[float],
):
    """``S A S scale + shift I`` and ``S U``; block ``(i, k)`` of ``A`` for evaluation node ``i`` and
    force node ``k`` is ``M1`` of the segment starting at ``k`` plus ``M2`` of the segment ending at ``k``.
    Only the lower triangle is computed, and mirrored."""
    i, k = wp.tid()
    if k > i:
        return
    x = wp.vec3d(pos[i])
    kl = k % nv
    block = wp.mat33d()
    if kl < nv - 1:
        m1, m2 = _segment_blocks(x, wp.vec3d(pos[k]), wp.vec3d(pos[k + 1]), eps)
        block += m1
    if kl > 0:
        m1, m2 = _segment_blocks(x, wp.vec3d(pos[k - 1]), wp.vec3d(pos[k]), eps)
        block += m2
    for a in range(3):
        for b in range(3):
            r = 3 * i + a
            c = 3 * k + b
            if c <= r:
                w = float(block[a, b]) * scale * s[r] * s[c]
                if r == c:
                    w += shift
                A[r, c] = w
                A[c, r] = w
    if k == 0:
        v = vel[i]
        for a in range(3):
            U[3 * i + a] = s[3 * i + a] * v[a]


@wp.kernel
def _force_kernel(y: wp.array[float], s: wp.array[float], scale: float, force: wp.array[wp.vec3]):
    i = wp.tid()
    force[i] = scale * wp.vec3(s[3 * i] * y[3 * i], s[3 * i + 1] * y[3 * i + 1], s[3 * i + 2] * y[3 * i + 2])


class Stokeslets:
    """Drag on ``rods * nv`` nodes (rods of ``nv`` nodes, stored rod after rod)."""

    def __init__(self, rods: int, nv: int, viscosity: float, epsilon: float, device):
        self.n, self.nv, self.device = rods * nv, nv, device
        self.viscosity, self.epsilon = viscosity, epsilon
        n3 = 3 * self.n
        self.A = wp.zeros((n3, n3), dtype=float, device=device)  # overwritten by its factor
        self.U = wp.zeros(n3, dtype=float, device=device)  # the right-hand side, then the solution
        self.force = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self._s = wp.ones(n3, dtype=float, device=device)
        self._scale, self._shift, self._post = 1.0, 0.0, -8.0 * np.pi * viscosity  # explicit
        if not wp.get_device(device).is_cuda:
            return  # SciPy solves in place on the host arrays
        self._lib = lib = _cusolver_library()
        self._handle = ctypes.c_void_p()
        _check(lib.cusolverDnCreate(ctypes.byref(self._handle)), "cusolverDnCreate")
        lwork = ctypes.c_int()
        _check(lib.cusolverDnSpotrf_bufferSize(self._handle, _LOWER, n3, self.A.ptr, n3, ctypes.byref(lwork)),
               "spotrf_bufferSize")
        self._work = wp.zeros(max(lwork.value, 1), dtype=float, device=device)
        self._info = wp.zeros(1, dtype=wp.int32, device=device)  # nonzero if A is not positive definite

    def implicit(self, mass: np.ndarray, dt: float) -> None:
        """Take the drag implicitly over ``dt`` (see the module), ``mass`` per DOF (``3 rods nv``)."""
        self._s.assign(np.sqrt(np.asarray(mass, dtype=np.float64)).astype(np.float32))
        self._scale, self._shift, self._post = 1.0 / (8.0 * np.pi * self.viscosity), dt, -1.0

    def compute(self, pos: wp.array, vel: wp.array) -> wp.array:
        """Nodal drag ``F = -8 pi eta A(pos)^-1 vel`` (or its implicit form, see the module) into ``self.force``."""
        n3 = 3 * self.n
        wp.launch(_assemble_kernel, dim=(self.n, self.n),
                  inputs=[pos, vel, self.nv, wp.float64(self.epsilon), self._s, self._scale, self._shift],
                  outputs=[self.A, self.U], device=self.device)
        # A is symmetric: its row-major lower triangle is cuSOLVER's column-major upper, either will do.
        if self.A.device.is_cuda:
            lib = self._lib
            _check(lib.cusolverDnSetStream(self._handle, wp.get_stream(self.device).cuda_stream), "setStream")
            _check(lib.cusolverDnSpotrf(self._handle, _LOWER, n3, self.A.ptr, n3, self._work.ptr,
                                        self._work.shape[0], self._info.ptr), "spotrf")
            _check(lib.cusolverDnSpotrs(self._handle, _LOWER, n3, 1, self.A.ptr, n3, self.U.ptr, n3, self._info.ptr),
                   "spotrs")
        else:  # the host arrays are NumPy views: factor and solve in place
            U = self.U.numpy()
            factor = scipy.linalg.cho_factor(self.A.numpy().T, lower=True, overwrite_a=True, check_finite=False)
            U[:] = scipy.linalg.cho_solve(factor, U, check_finite=False)
        wp.launch(_force_kernel, dim=self.n, inputs=[self.U, self._s, self._post], outputs=[self.force],
                  device=self.device)
        return self.force
