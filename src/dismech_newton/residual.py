"""Pure-Warp constitutive residual operators (5D DER rods and 6D BAC shells)."""

from dataclasses import dataclass
from functools import cache
from types import SimpleNamespace
from typing import Literal

import numpy as np
import warp as wp
from newton import Model

from .builder import _P, vec5f, vec6f, vec10f, vec12f
from .rod import Rod, _dkappa, _dphi, _geometry, _oriented, rod_dofs, vec11f
from .shell import Shell, _angles, _forms, _points, _row, _whiten, shell_dofs, vec21f, vec21i

_ADJ_OPTS = {"enable_backward": True, "max_unroll": 32, "optimization_level": 2}
_MEM_MODES = {"none": 0, "dahl": 0, "elastoplastic": 1, "prony": 2, "bouc_wen": 3}
_MEM_ATTRS = {1: ("flow", "yield_weight"), 2: ("k_mem", "tau"), 3: ("k_mem", "flow", "bw_sat")}

@dataclass
class MemorySpec:
    """Configuration for carried internal constitutive memory ``h_t`` (elastoplastic, Prony, Bouc-Wen)."""
    mode: Literal["none", "dahl", "elastoplastic", "bouc_wen", "prony"] = "none"
    yield_eps: float = 0.0
    yield_delta: float = 1.0e-3
    enabled_dims: tuple[int, ...] = (2, 3, 4)
    k_mem: float | np.ndarray | wp.array = 0.25
    flow: float | np.ndarray | wp.array = 0.5
    tau: float | np.ndarray | wp.array = 0.1
    bw_sat: float | np.ndarray | wp.array = 0.1
    yield_weight: float | np.ndarray | wp.array = 1.0

@dataclass
class ResidualSpec:
    """Configuration and optional initial weights for pure-Warp constitutive residuals."""
    rod_rff_dim: int = 64
    rff_bandwidth: float = 1.5
    iso_bending: bool = True
    clamp_lo: float = -0.35
    clamp_hi: float = 8.0
    rod_scales: tuple[float, ...] = (0.05, 0.05, 5.0, 5.0, 5.0, 0.5, 0.5, 50.0, 50.0, 50.0)
    seed: int = 0
    w_rod: np.ndarray | wp.array | None = None
    spatial_rest_delta: np.ndarray | wp.array | None = None
    shell_rff_dim: int = 64
    shell_scales: tuple[float, ...] = (0.05, 0.05, 0.05, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 20.0, 20.0, 20.0)
    w_shell: np.ndarray | wp.array | None = None
    shell_spatial_rest_delta: np.ndarray | wp.array | None = None
    memory: MemorySpec | None = None
    shell_memory: MemorySpec | None = None

def _as_vec_grad(val, dt, nv: int, n0: int, n1: int | None, dev, name: str) -> wp.array:
    """Validate/convert scalar, NumPy, or Warp array to a differentiable vector ``wp.array`` on ``dev``."""
    mem, r0, rg = (n0 < 0), max(n0, 0), (n1 is not None)
    if val is None:
        return wp.zeros((1, r0) if (not mem and rg) else 1, dtype=dt, device=dev, requires_grad=rg)
    w = isinstance(val, wp.array)
    a = val if w else np.asarray(val, np.float32)
    if not w and mem:
        a = np.full((1, nv), a.item(), "f4") if a.size == 1 else (a[None] if a.shape == (nv,) else a)
    elif not mem and rg and a.shape in ((r0,), (r0, nv)):
        a = a.reshape((1,) + a.shape) if w else a[None]
    a = ((a.reshape(-1) if mem and a.ndim == 2 else a).view(dt) if a.dtype == wp.float32 and a.shape[-1] == nv
         else (a if a.dtype == dt else None)) if w else a
    sh = ((1,), (n1,)) if mem else (((r0,),) if not rg else ((1, r0), (n1, r0)))
    if a is None or a.shape not in (sh if w else tuple(s + (nv,) for s in sh)):
        raise ValueError(f"{name} shape/dtype mismatch: shape={getattr(val, 'shape', np.shape(val))}")
    out = a if (w and a.device == dev) else wp.array(a.numpy() if w else a, dtype=dt, device=dev)
    out.requires_grad = True
    return out

@wp.func
def _iso_qh2(u2: float) -> wp.vec2:
    """Singularity-free radial RFF quotient ``q(u^2) = sin(u)/u - exp(-u^2/2)`` and ``2 q'(u^2)``."""
    if u2 < 1.0e-4:
        return wp.vec2(u2 * (1.0 / 3.0 - (7.0 / 60.0) * u2), 2.0 / 3.0 - (7.0 / 15.0) * u2)
    u, e = wp.sqrt(u2), wp.exp(-0.5 * u2)
    q = wp.sin(u) / u - e
    return wp.vec2(q, (wp.cos(u) - (1.0 - u2) * e - q) / u2)

@wp.func
def _psi_dpsi(u: float, cn: float) -> wp.vec2:
    """Origin-orthogonal odd RFF basis ``psi(u)`` and derivative ``psi'(u)``."""
    eu = wp.exp(-0.5 * u * u)
    return wp.vec2(cn * (wp.sin(u) - u * eu), cn * (wp.cos(u) - (1.0 - u * u) * eu))

@wp.func_native(snippet=(
    "#if defined(__CUDA_ARCH__)\nunsigned m = 0xffu << (threadIdx.x & 24);\nx += __shfl_down_sync(m, x, 4, 8); "
    "x += __shfl_down_sync(m, x, 2, 8); x += __shfl_down_sync(m, x, 1, 8);\n#endif\nreturn x;\n"),
    adj_snippet=("#if defined(__CUDA_ARCH__)\nadj_x += __shfl_sync(0xffu << (threadIdx.x & 24), adj_ret, 0, 8);\n"
                 "#else\nadj_x += adj_ret;\n#endif\n"))
def _ts(x: float) -> float: ...

@wp.func_native(snippet=(
    "#if defined(__CUDA_ARCH__)\nreturn __shfl_sync(0xffu << (threadIdx.x & 24), x, 0, 8);\n"
    "#else\nreturn x;\n#endif\n"),
    adj_snippet=("#if defined(__CUDA_ARCH__)\nunsigned m = 0xffu << (threadIdx.x & 24); float s = adj_ret;\n"
                 "s += __shfl_down_sync(m, s, 4, 8); s += __shfl_down_sync(m, s, 2, 8); "
                 "s += __shfl_down_sync(m, s, 1, 8);\nif ((threadIdx.x & 7) == 0) adj_x += s;\n"
                 "#else\nadj_x += adj_ret;\n#endif\n"))
def _bcast8(x: float) -> float: ...

@wp.func
def _clamp_ch(k0: float, c0: float, sig: float, dkr: float, dcr: float, r0: float, er: float, ed: float,
              mem_i: float, d_c: float, clo: float, chi: float, dt: float) -> wp.vec3:
    """SPD tangent clamping + semi-implicit rest shift for one strain channel."""
    dk = wp.where(k0 > 0.0, wp.clamp(dkr, clo * k0, chi * k0), 0.0)
    ks, den = k0 + dk, wp.max(k0 + dk, 1.0e-30)
    dc = wp.where(c0 > 0.0, wp.clamp(dcr, clo * c0, chi * c0),
                  wp.where(k0 > 0.0, wp.clamp(dcr, 0.0, chi * k0 * dt), 0.0))
    return wp.vec3(ks, c0 + dc + d_c, wp.where(ks > 0.0, r0 - mem_i / den - (sig - dk * er - dc * ed) / den, r0))

@cache
def _mem_ops(N: int, md: int):
    """Mode-specialized ``module='unique'`` memory step helper and carry kernel for 5D rods and 6D shells."""
    vt, pt = (vec5f, vec10f) if N == 5 else (vec6f, vec12f)
    @wp.func(name=f"_gv_{N}_{md}")
    def _gv(a: wp.array[vt], w: int) -> vt:
        return a[wp.where(a.shape[0] > 1, w, 0)]
    @wp.func(name=f"_yg_{N}_{md}")
    def _yg(d: vt, mk: vt, yv: vt, ye: float, yd: float) -> wp.vec3:
        r2 = float(0.0)
        for j in range(N):
            r2 = r2 + yv[j] * yv[j] * mk[j] * d[j] * d[j]
        if r2 <= ye * ye:
            return wp.vec3(0.0)
        ex = wp.sqrt(r2) - ye
        return wp.vec3(ex + ye, ex * ex / (ex + yd), ex * (ex + 2.0 * yd) / ((ex + yd) * (ex + yd)))
    @wp.func(name=f"_mem_step_{N}_{md}")
    def mem_step(pb: pt, rb_i: vt, dr_i: vt, hdr: int, za: vt, ep: vt, h: vt, sp: vt, dk: vt, dc: vt, ls: vt, sa: vt,
                 sb: vt, dm0: vt, dc0: vt, w: int, km: wp.array[vt], fl: wp.array[vt], ta: wp.array[vt],
                 bs: wp.array[vt], yw: wp.array[vt], mk: vt, ye: float, yd: float, clo: float, chi: float, dt: float):
        po, ro = pt(), vt()
        r0 = rb_i + dr_i if hdr != 0 else rb_i
        er, ef = wp.cw_mul(za, sa), vt(0.0)
        if not wp.isnan(ep[0]):
            ef = (er + r0 - ep) / dt
        if wp.static(md == 1):
            flv, yg = _gv(fl, w), _yg(er - h, mk, _gv(yw, w), ye, yd)
        if wp.static(md == 2):
            kmv, tav = _gv(km, w), _gv(ta, w)
        if wp.static(md == 3):
            kmv, flv, bsv = _gv(km, w), _gv(fl, w), _gv(bs, w)
        for j in range(N):
            k0, dkj, dcj, mvj = pb[j], 0.0, 0.0, 0.0
            if wp.static(md != 0):
                if mk[j] != 0.0:
                    if wp.static(md == 1):
                        dkj, mvj = -k0 * wp.max(flv[j], 0.0) * yg[2], -k0 * h[j]
                    if wp.static(md == 2):
                        dkj = wp.max(kmv[j], 0.0) * k0 * wp.exp(-dt / wp.max(tav[j], 1.0e-6))
                        mvj = dkj * (er[j] - h[j])
                    if wp.static(md == 3):
                        kp, u = wp.max(kmv[j], 0.0) * k0, wp.clamp(h[j] / wp.max(bsv[j], 1.0e-6), -1.0, 1.0)
                        dcj, mvj = wp.min(dt * kp * wp.max(flv[j], 0.0) * (1.0 - 0.5 * u * u), chi * k0 * dt), kp * h[j]
            r = _clamp_ch(k0, pb[N + j], ls[j] * sp[j], ls[j] * dk[j] / sa[j] + dkj, ls[j] * dc[j] / sb[j],
                          r0[j], er[j], ef[j], dm0[j] + mvj, dc0[j] + dcj, clo, chi, dt)
            po[j] = r[0]
            po[N + j] = r[1]
            ro[j] = r[2]
        return po, ro
    if not md:
        return mem_step, None
    @wp.kernel(name=f"_mem_carry_{N}_{md}", module="unique", module_options=_ADJ_OPTS)
    def mem_carry(prev: wp.array[vt], strain: wp.array[vt], rb: wp.array[vt], dr: wp.array[vt], hdr: int,
                  mem: wp.array[vt], ew: wp.array[wp.int32], fl: wp.array[vt], ta: wp.array[vt], bs: wp.array[vt],
                  yw: wp.array[vt], mk: vt, ye: float, yd: float, dt: float, mem_out: wp.array[vt]):
        t = wp.tid()
        h, out = mem[t], vt(0.0)
        if dt > 0.0:
            ep, w, r0 = strain[t], ew[t], (rb[t] + dr[t] if hdr != 0 else rb[t])
            if wp.static(md == 1):
                flv, yg = _gv(fl, w), _yg(ep - r0 - h, mk, _gv(yw, w), ye, yd)
            if wp.static(md == 2):
                tav = _gv(ta, w)
            if wp.static(md == 3):
                flv, bsv = _gv(fl, w), _gv(bs, w)
            for j in range(N):
                v = h[j]
                if mk[j] != 0.0:
                    if wp.static(md == 1):
                        if yg[0] > 0.0:
                            v = h[j] + wp.max(flv[j], 0.0) * yg[1] * (ep[j] - r0[j] - h[j]) / yg[0]
                    if wp.static(md == 2):
                        v = h[j] + (1.0 - wp.exp(-dt / wp.max(tav[j], 1.0e-6))) * (ep[j] - r0[j] - h[j])
                    if wp.static(md == 3):
                        d, B, ad = ep[j] - prev[t][j], wp.max(bsv[j], 1.0e-6), wp.abs(ep[j] - prev[t][j])
                        if ad > ye:
                            sm = ad if ye <= 0.0 else (ad - ye) * (ad - ye) / (ad - ye + yd)
                            u, sn = wp.clamp(h[j] / B, -1.0, 1.0), wp.where(d >= 0.0, 1.0, -1.0)
                            f = 1.0 - 0.5 * u * u * (1.0 + sn * wp.where(h[j] >= 0.0, 1.0, -1.0))
                            v = wp.clamp(h[j] + sn * sm * wp.max(flv[j], 0.0) * f, -B, B)
                out[j] = v
        mem_out[t] = out if dt > 0.0 else h
    return mem_step, mem_carry

@wp.func
def _rod_strain_rate(x: vec11f, xd: vec11f, v: Rod, t: int):
    """Triplet strain and directional strain rate ``J * xd`` without forming the full ``5x11`` matrix."""
    g, xo = _geometry(_oriented(x, v.sign[t]), v, t), _oriented(xd, v.sign[t])
    ze, zf = wp.vec3(xo[4] - xo[0], xo[5] - xo[1], xo[6] - xo[2]), wp.vec3(xo[8] - xo[4], xo[9] - xo[5], xo[10] - xo[6])
    ge, gf = _dphi(g.te_old, g.te, g.ne), _dphi(g.tf_old, g.tf, g.nf)
    De1, Df1, a1, b1 = _dkappa(g, g.strain[2], g.m2e, g.m2f, -g.m1e, -g.m1f)
    De2, Df2, a2, b2 = _dkappa(g, g.strain[3], -g.m1e, -g.m1f, -g.m2e, -g.m2f)
    pe, pf = wp.dot(ge, ze) + xo[3], wp.dot(gf, zf) + xo[7]
    return g.strain, vec5f(wp.dot(g.te, ze) / g.l0e, wp.dot(g.tf, zf) / g.l0f,
        wp.dot(De1, ze) + wp.dot(Df1, zf) + a1 * pe + b1 * pf, wp.dot(De2, ze) + wp.dot(Df2, zf) + a2 * pe + b2 * pf,
        0.5 * wp.dot(g.kb / g.ne, ze) + 0.5 * wp.dot(g.kb / g.nf, zf) - pe + pf)

@wp.func
def _rff_loop(z_a: vec5f, z_b: vec5f, fr: wp.array[vec10f], wr: wp.array2d[vec5f],
              w_idx: int, R_: int, lane: int, stride: int, cn: float, iso: int):
    """Strided origin-orthogonal odd RFF sum in a helper function (avoids Warp kernel-loop AD bugs)."""
    r2_k, r2_v = z_a[2] * z_a[2] + z_a[3] * z_a[3], z_b[2] * z_b[2] + z_b[3] * z_b[3]
    sp, dk, dc = vec5f(0.0), vec5f(0.0), vec5f(0.0)
    for k in range(wp.where(lane < R_, (R_ - lane + stride - 1) // stride, 0)):
        f, w = fr[lane + k * stride], wr[w_idx, lane + k * stride]
        u = (f[0] * z_a[0] + f[1] * z_a[1] + f[4] * z_a[4] + f[5] * z_b[0] + f[6] * z_b[1] + f[9] * z_b[4]
             + wp.where(iso == 0, f[2] * z_a[2] + f[3] * z_a[3] + f[7] * z_b[2] + f[8] * z_b[3], 0.0))
        pd = _psi_dpsi(u, cn)
        s2, s3, c2, c3 = w[2] * pd[0], w[3] * pd[0], w[2] * f[7] * pd[1], w[3] * f[8] * pd[1]
        k2, k3 = w[2] * f[2] * pd[1], w[3] * f[3] * pd[1]
        if iso != 0:
            om2, ov2, ck, cv = f[2] * f[2], f[7] * f[7], cn * w[2] * f[2], cn * w[3] * f[7]
            qk, qv = _iso_qh2(om2 * r2_k), _iso_qh2(ov2 * r2_v)
            s2, s3 = ck * qk[0] * z_a[2] + cv * qv[0] * z_b[2], ck * qk[0] * z_a[3] + cv * qv[0] * z_b[3]
            k2, k3 = ck * (qk[0] + om2 * z_a[2] * z_a[2] * qk[1]), ck * (qk[0] + om2 * z_a[3] * z_a[3] * qk[1])
            c2, c3 = cv * (qv[0] + ov2 * z_b[2] * z_b[2] * qv[1]), cv * (qv[0] + ov2 * z_b[3] * z_b[3] * qv[1])
        sp = sp + vec5f(w[0] * pd[0], w[1] * pd[0], s2, s3, w[4] * pd[0])
        dk = dk + vec5f(w[0] * f[0] * pd[1], w[1] * f[1] * pd[1], k2, k3, w[4] * f[4] * pd[1])
        dc = dc + vec5f(w[0] * f[5] * pd[1], w[1] * f[6] * pd[1], c2, c3, w[4] * f[9] * pd[1])
    return (vec5f(_ts(sp[0]), _ts(sp[1]), _ts(sp[2]), _ts(sp[3]), _ts(sp[4])),
            vec5f(_ts(dk[0]), _ts(dk[1]), _ts(dk[2]), _ts(dk[3]), _ts(dk[4])),
            vec5f(_ts(dc[0]), _ts(dc[1]), _ts(dc[2]), _ts(dc[3]), _ts(dc[4])))

@wp.func
def _shell_rff_loop(z_a: vec6f, z_b: vec6f, fr: wp.array[vec12f], wr: wp.array2d[vec6f],
                    w_idx: int, R_: int, lane: int, stride: int, cn: float):
    """Strided origin-orthogonal odd RFF sum across 6D shell strain and rate channels."""
    sp, dk, dc = vec6f(0.0), vec6f(0.0), vec6f(0.0)
    for k in range(wp.where(lane < R_, (R_ - lane + stride - 1) // stride, 0)):
        f, w = fr[lane + k * stride], wr[w_idx, lane + k * stride]
        fa, fb = vec6f(f[0], f[1], f[2], f[3], f[4], f[5]), vec6f(f[6], f[7], f[8], f[9], f[10], f[11])
        pd = _psi_dpsi(wp.dot(fa, z_a) + wp.dot(fb, z_b), cn)
        sp, dk, dc = sp + pd[0] * w, dk + pd[1] * wp.cw_mul(w, fa), dc + pd[1] * wp.cw_mul(w, fb)
    return (vec6f(_ts(sp[0]), _ts(sp[1]), _ts(sp[2]), _ts(sp[3]), _ts(sp[4]), _ts(sp[5])),
            vec6f(_ts(dk[0]), _ts(dk[1]), _ts(dk[2]), _ts(dk[3]), _ts(dk[4]), _ts(dk[5])),
            vec6f(_ts(dc[0]), _ts(dc[1]), _ts(dc[2]), _ts(dc[3]), _ts(dc[4]), _ts(dc[5])))

@cache
def _rod_step_fn(md: int):
    """Fused 5D DER rod strain/rate, RFF reduction, SPD clamping, Dahl, and memory kernel."""
    mem_step, _ = _mem_ops(5, md)
    @wp.kernel(name=f"_rod_step_m{md}", module="unique", module_options=_ADJ_OPTS)
    def step(q: wp.array[float], u: wp.array[float], v: Rod, pb: wp.array[vec10f], rb: wp.array[vec5f],
             sq: wp.array[vec5f], dr: wp.array[vec5f], has_dr: int, dahl: wp.array[wp.vec2], fr: wp.array[vec10f],
             wr: wp.array2d[vec5f], ew: wp.array[wp.int32], share: wp.array[wp.vec2], mem: wp.array[vec5f],
             km: wp.array[vec5f], fl: wp.array[vec5f], ta: wp.array[vec5f], bs: wp.array[vec5f], yw: wp.array[vec5f],
             mk: vec5f, ye: float, yd: float, sc: vec10f, iso: int, st: int, clo: float, chi: float, dt: float,
             pout: wp.array[vec10f], rout: wp.array[vec5f]):
        t, lane, za, zb = wp.tid() // st, wp.tid() % st, vec5f(0.0), vec5f(0.0)
        le, lf, dl = float(0.0), float(0.0), float(0.0)
        if lane == 0:
            c, d = v.conn[t], rod_dofs(v, t)
            x = vec11f(q[d[0]], q[d[1]], q[d[2]], q[d[3]], q[d[4]], q[d[5]],
                       q[d[6]], q[d[7]], q[d[8]], q[d[9]], q[d[10]])
            xd = vec11f(u[d[0]], u[d[1]], u[d[2]], u[d[3]], u[d[4]], u[d[5]],
                        u[d[6]], u[d[7]], u[d[8]], u[d[9]], u[d[10]])
            eps, ed = _rod_strain_rate(x, xd, v, t)
            er = eps - (rb[t] + dr[t] if has_dr != 0 else rb[t])
            le, lf = float(v.edge_length[c[0]]), float(v.edge_length[c[1]])
            dl = 0.5 * (le + lf)
            za = vec5f(er[0] / sc[0], er[1] / sc[1], er[2] / (dl * sc[2]), er[3] / (dl * sc[3]), er[4] / (dl * sc[4]))
            zb = vec5f(ed[0] / sc[5], ed[1] / sc[6], ed[2] / (dl * sc[7]), ed[3] / (dl * sc[8]), ed[4] / (dl * sc[9]))
        z_a, z_b = za, zb
        if st > 1:
            z_a = vec5f(_bcast8(za[0]), _bcast8(za[1]), _bcast8(za[2]), _bcast8(za[3]), _bcast8(za[4]))
            z_b = vec5f(_bcast8(zb[0]), _bcast8(zb[1]), _bcast8(zb[2]), _bcast8(zb[3]), _bcast8(zb[4]))
        R_, wi = fr.shape[0], (ew[t] if wr.shape[0] > 1 else 0)
        sp, dk, dc = vec5f(0.0), vec5f(0.0), vec5f(0.0)
        if R_ > 0:
            sp, dk, dc = _rff_loop(z_a, z_b, fr, wr, wi, R_, lane, st, wp.sqrt(2.0 / float(R_)), iso)
        if lane == 0:
            sv, dv, p0, mv = share[t], dahl[t], pb[t], mem[t]
            ls = vec5f(sv[0] * le, sv[1] * lf, 1.0, 1.0, 1.0)
            on = wp.where(dv[0] > 0.0 and dv[1] > 0.0, 1.0, 0.0)
            rt = on * dt * dv[0] / wp.max(dv[1], 1.0e-30)
            dm0 = vec5f(0.0, 0.0, on * mv[2], on * mv[3], on * mv[4])
            dc0 = vec5f(0.0, 0.0, rt * p0[2], rt * p0[3], rt * p0[4])
            sa = vec5f(sc[0], sc[1], dl * sc[2], dl * sc[3], dl * sc[4])
            sb = vec5f(sc[5], sc[6], dl * sc[7], dl * sc[8], dl * sc[9])
            po, ro = mem_step(p0, rb[t], dr[t], has_dr, z_a, sq[t], mv, sp, dk, dc, ls, sa, sb, dm0, dc0,
                              ew[t], km, fl, ta, bs, yw, mk, ye, yd, clo, chi, dt)
            pout[t] = po
            rout[t] = ro
    return step

@wp.func
def _v21(a: wp.array[float], d: vec21i) -> vec21f:
    return vec21f(a[d[0]], a[d[1]], a[d[2]], a[d[3]], a[d[4]], a[d[5]], a[d[6]], a[d[7]], a[d[8]], a[d[9]], a[d[10]],
                  a[d[11]], a[d[12]], a[d[13]], a[d[14]], a[d[15]], a[d[16]], a[d[17]], a[d[18]], a[d[19]], a[d[20]])

@wp.func
def _shell_strain_rate(x: vec21f, xd: vec21f, v: Shell, f: int):
    """Face strain and directional strain rate ``J * xd`` without forming the full ``6x21`` matrix."""
    P, dP, th, dth = _points(x), _points(xd), _angles(x), _angles(xd)
    eu, ev, deu, dev = _row(P, 1)-_row(P, 0), _row(P, 2)-_row(P, 0), _row(dP, 1)-_row(dP, 0), _row(dP, 2)-_row(dP, 0)
    dm = wp.vec3(2.0 * wp.dot(eu, deu), wp.dot(eu, dev) + wp.dot(ev, deu), 2.0 * wp.dot(ev, dev))
    m, g = _forms(P, th, v.sign[f], v.flap[f])
    n, dn = wp.cross(eu, ev), wp.cross(deu, ev) + wp.cross(eu, dev)
    nn = wp.length(n)
    dnn, dg = wp.dot(n / nn, dn), wp.vec3()
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        pj, pk, dpj, dpk = _row(P, j), _row(P, k), _row(dP, j), _row(dP, k)
        t, dpsi = pk - pj, float(0.0)
        L = wp.length(t)
        tn, h, c = t / L, nn / L, g[i] * L / nn
        if v.flap[f][i] != 0.0:
            fl, dfl, pi, dpi = _row(P, 3 + i), _row(dP, 3 + i), _row(P, i), _row(dP, i)
            ng = wp.cross(pj - pk, fl - pk)
            u1, u2 = n / wp.dot(n, n), ng / wp.dot(ng, ng)
            dpsi = -0.5 * (wp.dot(u1, L * dpi + wp.dot(pi - pk, tn) * dpj - wp.dot(pi - pj, tn) * dpk)
                           + wp.dot(u2, L * dfl + wp.dot(fl - pk, tn) * dpj - wp.dot(fl - pj, tn) * dpk))
        dg[i] = c * (dnn - h * wp.dot(tn, dpk - dpj)) / L + h * (1.0 + c * c) * (dpsi + float(v.sign[f][i]) * dth[i])
    return _whiten(v, f, m, g), _whiten(v, f, dm, dg)

@cache
def _shell_step_fn(md: int):
    """Fused 6D BAC shell strain/rate, RFF reduction, SPD clamping, and memory kernel."""
    mem_step, _ = _mem_ops(6, md)
    @wp.kernel(name=f"_shell_step_m{md}", module="unique", module_options=_ADJ_OPTS)
    def step(q: wp.array[float], qd: wp.array[float], v: Shell, pb: wp.array[vec12f], rb: wp.array[vec6f],
             sq: wp.array[vec6f], dr: wp.array[vec6f], has_dr: int, ar: wp.array[float], fr: wp.array[vec12f],
             ws: wp.array2d[vec6f], ew: wp.array[wp.int32], mem: wp.array[vec6f], km: wp.array[vec6f],
             fl: wp.array[vec6f], ta: wp.array[vec6f], bs: wp.array[vec6f], yw: wp.array[vec6f], mk: vec6f,
             ye: float, yd: float, sc: vec12f, st: int, clo: float, chi: float, dt: float, pout: wp.array[vec12f],
             rout: wp.array[vec6f]):
        f, lane, za, zb = wp.tid() // st, wp.tid() % st, vec6f(0.0), vec6f(0.0)
        if lane == 0:
            d = shell_dofs(v, f)
            eps, ed = _shell_strain_rate(_v21(q, d), _v21(qd, d), v, f)
            er = eps - (rb[f] + dr[f] if has_dr != 0 else rb[f])
            za = vec6f(er[0] / sc[0], er[1] / sc[1], er[2] / sc[2], er[3] / sc[3], er[4] / sc[4], er[5] / sc[5])
            zb = vec6f(ed[0] / sc[6], ed[1] / sc[7], ed[2] / sc[8], ed[3] / sc[9], ed[4] / sc[10], ed[5] / sc[11])
        z_a, z_b = za, zb
        if st > 1:
            z_a = vec6f(_bcast8(za[0]), _bcast8(za[1]), _bcast8(za[2]), _bcast8(za[3]), _bcast8(za[4]), _bcast8(za[5]))
            z_b = vec6f(_bcast8(zb[0]), _bcast8(zb[1]), _bcast8(zb[2]), _bcast8(zb[3]), _bcast8(zb[4]), _bcast8(zb[5]))
        R_, wi = fr.shape[0], (ew[f] if ws.shape[0] > 1 else 0)
        sp, dk, dc = vec6f(0.0), vec6f(0.0), vec6f(0.0)
        if R_ > 0:
            sp, dk, dc = _shell_rff_loop(z_a, z_b, fr, ws, wi, R_, lane, st, wp.sqrt(2.0 / float(R_)))
        if lane == 0:
            sa, sb = vec6f(sc[0], sc[1], sc[2], sc[3], sc[4], sc[5]), vec6f(sc[6], sc[7], sc[8], sc[9], sc[10], sc[11])
            po, ro = mem_step(pb[f], rb[f], dr[f], has_dr, z_a, sq[f], mem[f], sp, dk, dc, vec6f(ar[f]), sa, sb,
                              vec6f(0.0), vec6f(0.0), ew[f], km, fl, ta, bs, yw, mk, ye, yd, clo, chi, dt)
            pout[f] = po
            rout[f] = ro
    return step

class ConstitutiveResidual:
    """Pre-allocated pure-Warp 5D DER rod and 6D BAC shell constitutive residual and memory evaluator."""
    def __init__(self, model: Model, spec: ResidualSpec, device=None):
        if (min(spec.rod_rff_dim, spec.shell_rff_dim) < 0 or min(*spec.rod_scales, *spec.shell_scales) <= 0.0
                or len(spec.rod_scales) != 10 or len(spec.shell_scales) != 12
                or not (-1.0 < spec.clamp_lo <= spec.clamp_hi)):
            raise ValueError("Invalid ResidualSpec parameters")
        self.spec, self.device = spec, wp.get_device(model.device if device is None else device)
        self.clamp_lo, self.clamp_hi = float(spec.clamp_lo), float(spec.clamp_hi)
        self.scales, self.shell_scales = vec10f(*map(float, spec.rod_scales)), vec12f(*map(float, spec.shell_scales))
        self.iso_bending = int(bool(spec.iso_bending))
        d, pw = model.dismech, (model.particle_world.numpy() if model.particle_count else np.zeros(0, np.int32))
        gm = model.world_count > 1 and (pw < 0).any()
        self._nw = 1 if (gm or not pw.size) else int(np.maximum(pw, 0).max(initial=0)) + 1
        self.rod_count = nt = int(model.custom_frequency_counts.get(_P + "triplet", 0))
        self.shell_count = nf = int(model.custom_frequency_counts.get(_P + "face", 0))
        e, f = (d.triplet_edge0.numpy(), d.triplet_edge1.numpy()) if nt else ((), ())
        cnt = np.bincount(np.concatenate([e, f]), minlength=d.edge_length.shape[0]) if nt else None
        es = np.column_stack([1.0 / cnt[e], 1.0 / cnt[f]]).astype(np.float32) if nt else np.zeros((0, 2), np.float32)
        self.edge_share = wp.array(es, dtype=wp.vec2, device=self.device)
        rc = self._elem(nt, d.edge_node0.numpy()[e] if nt else None, pw, gm, False)
        sc = self._elem(nf, d.face_node0.numpy() if nf else None, pw, gm, True)
        (self.elem_world, self.params_step, self.rest_step, self.rff_freq, self.w_rod, self.has_d_rest,
         self.d_rest, self.mem_rod) = rc
        (self.shell_elem_world, self.shell_params_step, self.shell_rest_step, self.shell_rff_freq,
         self.w_shell, self.has_d_rest_shell, self.d_rest_shell, self.mem_shell) = sc
        self.rod_mem_mode, self.shell_mem_mode, self._ctx = self.mem_rod.mode, self.mem_shell.mode, (rc, sc)
        self.has_rod = bool(nt and (spec.rod_rff_dim > 0 or self.has_d_rest or self.rod_mem_mode != 0))
        self.has_shell = bool(nf and (spec.shell_rff_dim > 0 or self.has_d_rest_shell or self.shell_mem_mode != 0))

    def _elem(self, n: int, nodes, pw, gm, shell: bool):
        dev, sp, (nv, p_dt, v_dt) = self.device, self.spec, (6, vec12f, vec6f) if shell else (5, vec10f, vec5f)
        dim, seed, w_v, dr_v, ms = ((sp.shell_rff_dim, sp.seed + 1, sp.w_shell, sp.shell_spatial_rest_delta,
            sp.shell_memory) if shell else (sp.rod_rff_dim, sp.seed, sp.w_rod, sp.spatial_rest_delta, sp.memory))
        ms = ms or MemorySpec()
        if (ms.mode not in _MEM_MODES or (shell and ms.mode == "dahl") or ms.yield_eps < 0.0 or ms.yield_delta <= 0.0
                or any(i < 0 or i >= nv for i in ms.enabled_dims)):
            raise ValueError(f"Invalid MemorySpec: {ms}")
        md, pfx, hdr = _MEM_MODES[ms.mode], ("shell_" if shell else ""), int(dr_v is not None and n > 0)
        mem = SimpleNamespace(mode=md, mk=v_dt(*(float(i in ms.enabled_dims) for i in range(nv))),
            ye=float(ms.yield_eps), yd=float(ms.yield_delta), pr=wp.empty(n if md == 3 else 0, dtype=v_dt,
            device=dev, requires_grad=True), **{k: _as_vec_grad(getattr(ms, k), v_dt, nv, -1, self._nw, dev, k)
            for k in ("k_mem", "flow", "tau", "bw_sat", "yield_weight")})
        rff = np.random.default_rng(seed).normal(0.0, sp.rff_bandwidth, (dim, 2 * nv)).astype(np.float32)
        ew = np.zeros(n, np.int32) if (gm or not n) else np.maximum(pw[nodes], 0).astype(np.int32)
        return (wp.array(ew, dtype=wp.int32, device=dev), wp.empty(n, dtype=p_dt, device=dev),
                wp.empty(n, dtype=v_dt, device=dev), wp.array(rff, dtype=p_dt, device=dev),
                _as_vec_grad(w_v, v_dt, nv, dim, self._nw, dev, "w_shell" if shell else "w_rod"), hdr,
                _as_vec_grad(dr_v if hdr else None, v_dt, nv, n, None, dev, f"{pfx}spatial_rest_delta"), mem)

    def grad_inputs(self) -> list[wp.array]:
        """Differentiable parameter arrays tracked by ``SolverDiSMech._record``."""
        c = ((self.spec.rod_rff_dim and (self.rod_count or self.spec.w_rod is not None), self.w_rod),
             (self.has_d_rest, self.d_rest), (self.has_d_rest_shell, self.d_rest_shell),
             (self.spec.shell_rff_dim and (self.shell_count or self.spec.w_shell is not None), self.w_shell))
        return [a for k, a in c if k] + [getattr(m, k) for n, m in ((self.rod_count, self.mem_rod),
                (self.shell_count, self.mem_shell)) if n for k in _MEM_ATTRS.get(m.mode, ())]

    def _eval(self, shell: bool, el, s, dt: float, pb, rb, po, ro) -> None:
        if not el.count:
            return
        ew, dp, dr, rff, w_arr, hdr, d_rest, m = self._ctx[shell]
        st, bd = (1, 1) if self.device.is_cpu else ((1, 128) if not rff.shape[0] else (8, 128))
        ma = (getattr(s, el.memory), m.k_mem, m.flow, m.tau, m.bw_sat, m.yield_weight, m.mk, m.ye, m.yd)
        sk, xs = (_shell_step_fn(m.mode), [el.area, rff, w_arr, ew, *ma, self.shell_scales]) if shell else (
            _rod_step_fn(m.mode), [el.dahl, rff, w_arr, ew, self.edge_share, *ma, self.scales, self.iso_bending])
        wp.launch(sk, el.count * st, block_dim=bd, device=self.device, outputs=[dp if po is None else po,
                  dr if ro is None else ro], inputs=[s.q, s.qd, el.bind(s), pb, rb, getattr(s, el.stored),
                  d_rest, hdr, *xs, st, self.clamp_lo, self.clamp_hi, dt])

    def _carry(self, shell: bool, el, p, s, dt: float) -> None:
        ew, _, _, _, _, hdr, d_rest, m = self._ctx[shell]
        if not el.count or not m.mode:
            return
        out, prev = getattr(s, el.memory), getattr(p, el.stored)
        if m.mode == 3 and dt > 0.0:
            if out.requires_grad and m.pr.grad is not None:
                m.pr.grad.zero_()
            wp.launch(el.kernels.previous, el.count, inputs=[prev, p.q, el.bind(p)], outputs=[m.pr], device=self.device)
            prev = m.pr
        wp.launch(_mem_ops(6 if shell else 5, m.mode)[1], el.count, block_dim=1 if self.device.is_cpu else 128,
                  device=self.device, outputs=[out], inputs=[prev, getattr(s, el.stored), el.rest, d_rest, hdr,
                  getattr(p, el.memory), ew, m.flow, m.tau, m.bw_sat, m.yield_weight, m.mk, m.ye, m.yd, dt])

    def eval_rod(self, rods, s, dt: float, params_base, rest_base, params_out=None, rest_out=None) -> None:
        self._eval(False, rods, s, dt, params_base, rest_base, params_out, rest_out)
    def eval_shell(self, shells, s, dt: float, params_base, rest_base, params_out=None, rest_out=None) -> None:
        self._eval(True, shells, s, dt, params_base, rest_base, params_out, rest_out)
    def carry_rod(self, rods, p, s, dt: float) -> None:
        self._carry(False, rods, p, s, dt)
    def carry_shell(self, shells, p, s, dt: float) -> None:
        self._carry(True, shells, p, s, dt)
