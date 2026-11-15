"""Two-sided contact SPD PCG refinement kernels."""

import warp as wp

from ..worlds import Worlds
from .linesearch import SOLVER_MODULE

# -- SPD PCG on two-sided contacts: per world its scalars at c[w * _S + k] ------------------------
_RZ, _RZ0, _PW, _RZN, _ALPHA, _BETA, _STOP = range(7)
_S = 7


@wp.kernel(module=SOLVER_MODULE)
def _pcg_start_kernel(b: wp.array[wp.float64], dq: wp.array[float], wd: Worlds, x: wp.array[wp.float64],
                      r: wp.array[wp.float64], p: wp.array[wp.float64], c: wp.array[wp.float64]):
    i = wp.tid()
    w = wd.of[i]
    x[i] = wp.float64(0.0)
    r[i] = b[i]
    d = wp.float64(dq[i])
    p[i] = d
    if wd.active[w] != 0:
        wp.atomic_add(c, w * _S + _RZ0, b[i] * d)


@wp.kernel(module=SOLVER_MODULE)
def _pcg_init_kernel(wd: Worlds, c: wp.array[wp.float64]):
    w = wp.tid()
    o = w * _S
    rz0 = c[o + _RZ0]
    c[o + _RZ] = rz0
    c[o + _STOP] = wp.where(wd.active[w] != 0 and rz0 > wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))


@wp.kernel(module=SOLVER_MODULE)
def _spmv_kernel(indptr: wp.array[wp.int32], indices: wp.array[wp.int32], vals: wp.array[wp.float64],
                 p: wp.array[wp.float64], c: wp.array[wp.float64], wd: Worlds, y: wp.array[wp.float64]):
    """``y += A p`` for an upper-triangle symmetric CSR ``A`` in the running worlds."""
    i = wp.tid()
    if c[wd.of[i] * _S + _STOP] != wp.float64(0.0):
        return
    pi = p[i]
    yi = wp.float64(0.0)
    for slot in range(indptr[i], indptr[i + 1]):
        j = indices[slot]
        v = vals[slot]
        yi += v * p[j]
        if j > i:
            wp.atomic_add(y, j, v * pi)
    wp.atomic_add(y, i, yi)


@wp.kernel(module=SOLVER_MODULE)
def _pcg_rz_kernel(r: wp.array[wp.float64], z: wp.array[float], wd: Worlds, c: wp.array[wp.float64]):
    i = wp.tid()
    o = wd.of[i] * _S
    if c[o + _STOP] == wp.float64(0.0):
        wp.atomic_add(c, o + _RZN, r[i] * wp.float64(z[i]))


@wp.kernel(module=SOLVER_MODULE)
def _pcg_beta_kernel(c: wp.array[wp.float64]):
    w = wp.tid()
    o = w * _S
    if c[o + _STOP] != wp.float64(0.0):
        return
    rzn = c[o + _RZN]
    c[o + _RZN] = wp.float64(0.0)
    if rzn <= wp.float64(1.0e-4) * c[o + _RZ0]:
        c[o + _STOP] = wp.float64(1.0)
        return
    c[o + _BETA] = rzn / c[o + _RZ]
    c[o + _RZ] = rzn


@wp.kernel(module=SOLVER_MODULE)
def _pcg_dir_kernel(c: wp.array[wp.float64], z: wp.array[float], wd: Worlds, p: wp.array[wp.float64]):
    i = wp.tid()
    o = wd.of[i] * _S
    if c[o + _STOP] == wp.float64(0.0):
        p[i] = wp.float64(z[i]) + c[o + _BETA] * p[i]


@wp.kernel(module=SOLVER_MODULE)
def _pcg_pw_kernel(p: wp.array[wp.float64], w: wp.array[wp.float64], wd: Worlds, c: wp.array[wp.float64]):
    i = wp.tid()
    o = wd.of[i] * _S
    if c[o + _STOP] == wp.float64(0.0):
        wp.atomic_add(c, o + _PW, p[i] * w[i])


@wp.kernel(module=SOLVER_MODULE)
def _pcg_alpha_kernel(m: int, c: wp.array[wp.float64], go: wp.array[wp.int32]):
    w = wp.tid()
    o = w * _S
    c[o + _ALPHA] = wp.float64(0.0)
    if c[o + _STOP] != wp.float64(0.0):
        return
    pw = c[o + _PW]
    c[o + _PW] = wp.float64(0.0)
    rz = c[o + _RZ]
    if pw <= wp.float64(0.0):
        c[o + _STOP] = wp.float64(1.0)
        if m == 0:
            c[o + _ALPHA] = wp.float64(1.0)
        return
    c[o + _ALPHA] = rz / pw
    go[0] = 1


@wp.kernel(module=SOLVER_MODULE)
def _pcg_update_kernel(c: wp.array[wp.float64], p: wp.array[wp.float64], w: wp.array[wp.float64], wd: Worlds,
                       x: wp.array[wp.float64], r: wp.array[wp.float64]):
    i = wp.tid()
    a = c[wd.of[i] * _S + _ALPHA]
    if a != wp.float64(0.0):
        x[i] = x[i] + a * p[i]
        r[i] = r[i] - a * w[i]


@wp.kernel(module=SOLVER_MODULE)
def _pcg_end_kernel(x: wp.array[wp.float64], c: wp.array[wp.float64], wd: Worlds, dq: wp.array[float]):
    i = wp.tid()
    w = wd.of[i]
    if wd.active[w] != 0 and c[w * _S + _RZ0] > wp.float64(0.0):
        dq[i] = float(x[i])
