"""Origin-orthogonal odd random Fourier features (RFF) and 8-lane warp-shuffle reductions."""

import warp as wp

from ..constants import vec5f, vec6f, vec10f, vec12f


@wp.func
def iso_radial_quotient(u2: float) -> wp.vec2:
    """Singularity-free radial RFF quotient ``q(u^2) = sin(u)/u - exp(-u^2/2)`` and ``2 q'(u^2)``."""
    if u2 < 1.0e-4:
        return wp.vec2(u2 * (1.0 / 3.0 - (7.0 / 60.0) * u2), 2.0 / 3.0 - (7.0 / 15.0) * u2)
    u, e = wp.sqrt(u2), wp.exp(-0.5 * u2)
    q = wp.sin(u) / u - e
    return wp.vec2(q, (wp.cos(u) - (1.0 - u2) * e - q) / u2)


@wp.func
def psi_dpsi(u: float, cn: float) -> wp.vec2:
    """Origin-orthogonal odd RFF basis ``psi(u)`` and derivative ``psi'(u)``."""
    eu = wp.exp(-0.5 * u * u)
    return wp.vec2(cn * (wp.sin(u) - u * eu), cn * (wp.cos(u) - (1.0 - u * u) * eu))


@wp.func_native(
    snippet=(
        "#if defined(__CUDA_ARCH__)\n"
        "unsigned m = 0xffu << (threadIdx.x & 24);\n"
        "x += __shfl_down_sync(m, x, 4, 8);\n"
        "x += __shfl_down_sync(m, x, 2, 8);\n"
        "x += __shfl_down_sync(m, x, 1, 8);\n"
        "#endif\n"
        "return x;\n"
    ),
    adj_snippet=(
        "#if defined(__CUDA_ARCH__)\n"
        "adj_x += __shfl_sync(0xffu << (threadIdx.x & 24), adj_ret, 0, 8);\n"
        "#else\n"
        "adj_x += adj_ret;\n"
        "#endif\n"
    ),
)
def warp8_sum(x: float) -> float:
    """Sum ``x`` across an 8-lane sub-warp onto lane 0."""
    ...


@wp.func_native(
    snippet=(
        "#if defined(__CUDA_ARCH__)\n"
        "return __shfl_sync(0xffu << (threadIdx.x & 24), x, 0, 8);\n"
        "#else\n"
        "return x;\n"
        "#endif\n"
    ),
    adj_snippet=(
        "#if defined(__CUDA_ARCH__)\n"
        "unsigned m = 0xffu << (threadIdx.x & 24); float s = adj_ret;\n"
        "s += __shfl_down_sync(m, s, 4, 8);\n"
        "s += __shfl_down_sync(m, s, 2, 8);\n"
        "s += __shfl_down_sync(m, s, 1, 8);\n"
        "if ((threadIdx.x & 7) == 0) adj_x += s;\n"
        "#else\n"
        "adj_x += adj_ret;\n"
        "#endif\n"
    ),
)
def warp8_bcast(x: float) -> float:
    """Broadcast lane 0's scalar ``x`` to all 8 lanes in the sub-warp."""
    ...


@wp.func
def rod_rff_sum(
    z_a: vec5f,
    z_b: vec5f,
    fr: wp.array[vec10f],
    wr: wp.array2d[vec5f],
    w_idx: int,
    rff_dim: int,
    lane: int,
    stride: int,
    cn: float,
    iso: int,
):
    """Strided origin-orthogonal odd RFF sum across 5D rod strain and rate channels."""
    r2_k, r2_v = z_a[2] * z_a[2] + z_a[3] * z_a[3], z_b[2] * z_b[2] + z_b[3] * z_b[3]
    sp, dk, dc = vec5f(0.0), vec5f(0.0), vec5f(0.0)
    for k in range(wp.where(lane < rff_dim, (rff_dim - lane + stride - 1) // stride, 0)):
        f, w = fr[lane + k * stride], wr[w_idx, lane + k * stride]
        u = (
            f[0] * z_a[0]
            + f[1] * z_a[1]
            + f[4] * z_a[4]
            + f[5] * z_b[0]
            + f[6] * z_b[1]
            + f[9] * z_b[4]
            + wp.where(iso == 0, f[2] * z_a[2] + f[3] * z_a[3] + f[7] * z_b[2] + f[8] * z_b[3], 0.0)
        )
        pd = psi_dpsi(u, cn)
        s2, s3, c2, c3 = w[2] * pd[0], w[3] * pd[0], w[2] * f[7] * pd[1], w[3] * f[8] * pd[1]
        k2, k3 = w[2] * f[2] * pd[1], w[3] * f[3] * pd[1]
        if iso != 0:
            om2, ov2, ck, cv = f[2] * f[2], f[7] * f[7], cn * w[2] * f[2], cn * w[3] * f[7]
            qk, qv = iso_radial_quotient(om2 * r2_k), iso_radial_quotient(ov2 * r2_v)
            s2 = ck * qk[0] * z_a[2] + cv * qv[0] * z_b[2]
            s3 = ck * qk[0] * z_a[3] + cv * qv[0] * z_b[3]
            k2 = ck * (qk[0] + om2 * z_a[2] * z_a[2] * qk[1])
            k3 = ck * (qk[0] + om2 * z_a[3] * z_a[3] * qk[1])
            c2 = cv * (qv[0] + ov2 * z_b[2] * z_b[2] * qv[1])
            c3 = cv * (qv[0] + ov2 * z_b[3] * z_b[3] * qv[1])
        sp = sp + vec5f(w[0] * pd[0], w[1] * pd[0], s2, s3, w[4] * pd[0])
        dk = dk + vec5f(w[0] * f[0] * pd[1], w[1] * f[1] * pd[1], k2, k3, w[4] * f[4] * pd[1])
        dc = dc + vec5f(w[0] * f[5] * pd[1], w[1] * f[6] * pd[1], c2, c3, w[4] * f[9] * pd[1])
    return (
        vec5f(warp8_sum(sp[0]), warp8_sum(sp[1]), warp8_sum(sp[2]), warp8_sum(sp[3]), warp8_sum(sp[4])),
        vec5f(warp8_sum(dk[0]), warp8_sum(dk[1]), warp8_sum(dk[2]), warp8_sum(dk[3]), warp8_sum(dk[4])),
        vec5f(warp8_sum(dc[0]), warp8_sum(dc[1]), warp8_sum(dc[2]), warp8_sum(dc[3]), warp8_sum(dc[4])),
    )


@wp.func
def shell_rff_sum(
    z_a: vec6f,
    z_b: vec6f,
    fr: wp.array[vec12f],
    wr: wp.array2d[vec6f],
    w_idx: int,
    rff_dim: int,
    lane: int,
    stride: int,
    cn: float,
):
    """Strided origin-orthogonal odd RFF sum across 6D shell strain and rate channels."""
    sp, dk, dc = vec6f(0.0), vec6f(0.0), vec6f(0.0)
    for k in range(wp.where(lane < rff_dim, (rff_dim - lane + stride - 1) // stride, 0)):
        f, w = fr[lane + k * stride], wr[w_idx, lane + k * stride]
        fa = vec6f(f[0], f[1], f[2], f[3], f[4], f[5])
        fb = vec6f(f[6], f[7], f[8], f[9], f[10], f[11])
        pd = psi_dpsi(wp.dot(fa, z_a) + wp.dot(fb, z_b), cn)
        sp = sp + pd[0] * w
        dk = dk + pd[1] * wp.cw_mul(w, fa)
        dc = dc + pd[1] * wp.cw_mul(w, fb)
    return (
        vec6f(
            warp8_sum(sp[0]),
            warp8_sum(sp[1]),
            warp8_sum(sp[2]),
            warp8_sum(sp[3]),
            warp8_sum(sp[4]),
            warp8_sum(sp[5]),
        ),
        vec6f(
            warp8_sum(dk[0]),
            warp8_sum(dk[1]),
            warp8_sum(dk[2]),
            warp8_sum(dk[3]),
            warp8_sum(dk[4]),
            warp8_sum(dk[5]),
        ),
        vec6f(
            warp8_sum(dc[0]),
            warp8_sum(dc[1]),
            warp8_sum(dc[2]),
            warp8_sum(dc[3]),
            warp8_sum(dc[4]),
            warp8_sum(dc[5]),
        ),
    )
