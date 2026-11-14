"""Pure-Warp constitutive residual operators (5D DER rods and 6D BAC shells)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from newton import Model

from .builder import _P, vec5f, vec6f, vec10f, vec12f
from .rod import Rod, _dkappa, _dphi, _geometry, _oriented, rod_dofs, vec11f
from .shell import Shell, _angles, _forms, _points, _row, _whiten, shell_dofs, vec21f, vec21i

_ADJ_OPTS = {"enable_backward": True, "max_unroll": 32, "optimization_level": 2}

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

def _as_vec_grad(val, dt, nv: int, n0: int, n1: int | None, dev, name: str) -> wp.array:
    """Validate/convert a NumPy or Warp array to a differentiable vector ``wp.array`` on ``dev``."""
    if val is None:
        return wp.zeros((1, n0) if n1 is not None else 1, dtype=dt, device=dev, requires_grad=n1 is not None)
    sh, w = (((n0,),) if n1 is None else ((1, n0), (n1, n0))), isinstance(val, wp.array)
    a = val if w else np.asarray(val, dtype=np.float32)
    a = (a.reshape((1,) + a.shape) if w else a[None]) if n1 is not None and a.shape in ((n0,), (n0, nv)) else a
    a = (a if a.dtype == dt else (a.view(dt) if a.dtype == wp.float32 and a.shape[-1] == nv else None)) if w else a
    if a is None or not (a.shape in sh if w else a.shape in tuple(s + (nv,) for s in sh)):
        raise ValueError(f"{name} shape/dtype mismatch: shape={val.shape}")
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

@wp.func
def _ts(x: float) -> float:
    return wp.tile_sum(wp.tile(x))[0]

@wp.func
def _clamp_ch(k0: float, c0: float, sig: float, dkr: float, dcr: float, r0: float, er: float, ed: float,
              mem_i: float, d_c: float, clo: float, chi: float, dt: float) -> wp.vec3:
    """SPD tangent clamping + semi-implicit rest shift for one strain channel."""
    dk = wp.where(k0 > 0.0, wp.clamp(dkr, clo * k0, chi * k0), 0.0)
    ks, den = k0 + dk, wp.max(k0 + dk, 1.0e-30)
    dc = wp.where(c0 > 0.0, wp.clamp(dcr, clo * c0, chi * c0),
                  wp.where(k0 > 0.0, wp.clamp(dcr, 0.0, chi * k0 * dt), 0.0))
    return wp.vec3(ks, c0 + dc + d_c, wp.where(ks > 0.0, r0 - mem_i / den - (sig - dk * er - dc * ed) / den, r0))

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
    return sp, dk, dc

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
    return sp, dk, dc

@wp.kernel(module="unique", module_options=_ADJ_OPTS)
def _rod_prep_kernel(q: wp.array[float], u: wp.array[float], v: Rod, rb: wp.array[vec5f], sq: wp.array[vec5f],
                     dr: wp.array[vec5f], sc: vec10f, has_dr: int, dt: float, za: wp.array[vec5f],
                     zb: wp.array[vec5f], ef: wp.array[vec5f], r0o: wp.array[vec5f]):
    """Compute dimensionless triplet strain/rate coordinates ``(z_a, z_b)`` and rest state per rod triplet."""
    t = wp.tid()
    d = rod_dofs(v, t)
    x = vec11f(q[d[0]], q[d[1]], q[d[2]], q[d[3]], q[d[4]], q[d[5]], q[d[6]], q[d[7]], q[d[8]], q[d[9]], q[d[10]])
    xd = vec11f(u[d[0]], u[d[1]], u[d[2]], u[d[3]], u[d[4]], u[d[5]], u[d[6]], u[d[7]], u[d[8]], u[d[9]], u[d[10]])
    eps, ed = _rod_strain_rate(x, xd, v, t)
    ep, r0, c = sq[t], (rb[t] + dr[t] if has_dr != 0 else rb[t]), v.conn[t]
    ef[t] = vec5f(0.0)
    if not wp.isnan(ep[0]):
        ef[t] = (eps - ep) / dt
    r0o[t] = r0
    er, dl = eps - r0, 0.5 * (float(v.edge_length[c[0]]) + float(v.edge_length[c[1]]))
    za[t] = vec5f(er[0] / sc[0], er[1] / sc[1], er[2] / (dl * sc[2]), er[3] / (dl * sc[3]), er[4] / (dl * sc[4]))
    zb[t] = vec5f(ed[0] / sc[5], ed[1] / sc[6], ed[2] / (dl * sc[7]), ed[3] / (dl * sc[8]), ed[4] / (dl * sc[9]))

@wp.kernel(module="unique", module_options=_ADJ_OPTS)
def _rod_constit_step_kernel(v: Rod, p_base: wp.array[vec10f], dahl: wp.array[wp.vec2], mem: wp.array[vec5f],
    fr: wp.array[vec10f], w_rod: wp.array2d[vec5f], ew: wp.array[wp.int32], share: wp.array[wp.vec2],
    za_in: wp.array[vec5f], zb_in: wp.array[vec5f], ef_in: wp.array[vec5f], r0_in: wp.array[vec5f], sc: vec10f,
    iso: int, stride: int, clo: float, chi: float, dt: float, p_out: wp.array[vec10f], r_out: wp.array[vec5f]):
    """Tiled 5D DER rod RFF evaluation, warp-tile reduction, SPD tangent clamping, and additive Dahl."""
    t, lane = wp.tid()
    R_, z_a, wi = fr.shape[0], za_in[t], wp.where(w_rod.shape[0] > 1, ew[t], 0)
    sp, dk, dc = _rff_loop(z_a, zb_in[t], fr, w_rod, wi, R_, lane, stride, wp.sqrt(2.0 / float(R_)), iso)
    sp = vec5f(_ts(sp[0]), _ts(sp[1]), _ts(sp[2]), _ts(sp[3]), _ts(sp[4]))
    dk = vec5f(_ts(dk[0]), _ts(dk[1]), _ts(dk[2]), _ts(dk[3]), _ts(dk[4]))
    dc = vec5f(_ts(dc[0]), _ts(dc[1]), _ts(dc[2]), _ts(dc[3]), _ts(dc[4]))
    if lane == 0:
        c, sv, dv = v.conn[t], share[t], dahl[t]
        le, lf = float(v.edge_length[c[0]]), float(v.edge_length[c[1]])
        dl, ls = 0.5 * (le + lf), vec5f(sv[0] * le, sv[1] * lf, 1.0, 1.0, 1.0)
        sa = vec5f(sc[0], sc[1], dl * sc[2], dl * sc[3], dl * sc[4])
        sb = vec5f(sc[5], sc[6], dl * sc[7], dl * sc[8], dl * sc[9])
        on, po, ro = wp.where(dv[0] > 0.0 and dv[1] > 0.0, 1.0, 0.0), vec10f(), vec5f()
        rate, pb, r0, ef, mv = on * dt * dv[0] / wp.max(dv[1], 1.0e-30), p_base[t], r0_in[t], ef_in[t], mem[t]
        for i in range(5):
            r = _clamp_ch(pb[i], pb[5 + i], ls[i] * sp[i], ls[i] * dk[i] / sa[i], ls[i] * dc[i] / sb[i],
                          r0[i], z_a[i] * sa[i], ef[i], on * mv[i] if i >= 2 else 0.0,
                          rate * pb[i] if i >= 2 else 0.0, clo, chi, dt)
            po[i] = r[0]
            po[5 + i] = r[1]
            ro[i] = r[2]
        p_out[t] = po
        r_out[t] = ro

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

@wp.kernel(module="unique", module_options=_ADJ_OPTS)
def _shell_prep_kernel(q: wp.array[float], qd: wp.array[float], v: Shell, rb: wp.array[vec6f], sq: wp.array[vec6f],
                       dr: wp.array[vec6f], sc: vec12f, has_dr: int, dt: float, za: wp.array[vec6f],
                       zb: wp.array[vec6f], ef: wp.array[vec6f], r0o: wp.array[vec6f]):
    """Compute dimensionless face strain/rate coordinates ``(z_a, z_b)`` and rest state per shell face."""
    f = wp.tid()
    d = shell_dofs(v, f)
    eps, ed = _shell_strain_rate(_v21(q, d), _v21(qd, d), v, f)
    ep, r0 = sq[f], (rb[f] + dr[f] if has_dr != 0 else rb[f])
    ef[f] = vec6f(0.0)
    if not wp.isnan(ep[0]):
        ef[f] = (eps - ep) / dt
    r0o[f] = r0
    er = eps - r0
    za[f] = vec6f(er[0] / sc[0], er[1] / sc[1], er[2] / sc[2], er[3] / sc[3], er[4] / sc[4], er[5] / sc[5])
    zb[f] = vec6f(ed[0] / sc[6], ed[1] / sc[7], ed[2] / sc[8], ed[3] / sc[9], ed[4] / sc[10], ed[5] / sc[11])

@wp.kernel(module="unique", module_options=_ADJ_OPTS)
def _shell_constit_step_kernel(pb: wp.array[vec12f], ar: wp.array[float], fr: wp.array[vec12f], ws: wp.array2d[vec6f],
    ew: wp.array[wp.int32], za: wp.array[vec6f], zb: wp.array[vec6f], ef: wp.array[vec6f], r0i: wp.array[vec6f],
    sc: vec12f, st: int, clo: float, chi: float, dt: float, pout: wp.array[vec12f], rout: wp.array[vec6f]):
    """Tiled 6D BAC shell RFF evaluation, warp-tile reduction, area scaling, and SPD tangent clamping."""
    f, lane = wp.tid()
    R_, z_a, wi = fr.shape[0], za[f], wp.where(ws.shape[0] > 1, ew[f], 0)
    sp, dk, dc = _shell_rff_loop(z_a, zb[f], fr, ws, wi, R_, lane, st, wp.sqrt(2.0 / float(R_)))
    sp = vec6f(_ts(sp[0]), _ts(sp[1]), _ts(sp[2]), _ts(sp[3]), _ts(sp[4]), _ts(sp[5]))
    dk = vec6f(_ts(dk[0]), _ts(dk[1]), _ts(dk[2]), _ts(dk[3]), _ts(dk[4]), _ts(dk[5]))
    dc = vec6f(_ts(dc[0]), _ts(dc[1]), _ts(dc[2]), _ts(dc[3]), _ts(dc[4]), _ts(dc[5]))
    if lane == 0:
        af, p0, r0, ef_, po, ro = ar[f], pb[f], r0i[f], ef[f], vec12f(), vec6f()
        for i in range(6):
            r = _clamp_ch(p0[i], p0[6 + i], af * sp[i], af * dk[i] / sc[i], af * dc[i] / sc[6 + i],
                          r0[i], z_a[i] * sc[i], ef_[i], 0.0, 0.0, clo, chi, dt)
            po[i] = r[0]
            po[6 + i] = r[1]
            ro[i] = r[2]
        pout[f] = po
        rout[f] = ro

class ConstitutiveResidual:
    """Pre-allocated pure-Warp 5D DER rod and 6D BAC shell constitutive residual evaluator."""
    def __init__(self, model: Model, spec: ResidualSpec, device=None):
        if (min(spec.rod_rff_dim, spec.shell_rff_dim, *spec.rod_scales, *spec.shell_scales) <= 0.0
                or len(spec.rod_scales) != 10 or len(spec.shell_scales) != 12
                or not (-1.0 < spec.clamp_lo <= spec.clamp_hi)):
            raise ValueError("Invalid ResidualSpec parameters")
        self.spec, self.device = spec, wp.get_device(model.device if device is None else device)
        self.clamp_lo, self.clamp_hi = float(spec.clamp_lo), float(spec.clamp_hi)
        self.scales, self.shell_scales = vec10f(*map(float, spec.rod_scales)), vec12f(*map(float, spec.shell_scales))
        self._rods, self._shells, self.iso_bending = None, None, int(bool(spec.iso_bending))
        d, pw = model.dismech, (model.particle_world.numpy() if model.particle_count else np.zeros(0, np.int32))
        gm = model.world_count > 1 and (pw < 0).any()
        self._nw = 1 if (gm or not pw.size) else int(np.maximum(pw, 0).max(initial=0)) + 1
        self.rod_count = nt = int(model.custom_frequency_counts.get(_P + "triplet", 0))
        self.shell_count = nf = int(model.custom_frequency_counts.get(_P + "face", 0))
        e, f = (d.triplet_edge0.numpy(), d.triplet_edge1.numpy()) if nt else ((), ())
        cnt = np.bincount(np.concatenate([e, f]), minlength=d.edge_length.shape[0]) if nt else None
        es = np.column_stack([1.0 / cnt[e], 1.0 / cnt[f]]).astype(np.float32) if nt else np.zeros((0, 2), np.float32)
        self.edge_share = wp.array(es, dtype=wp.vec2, device=self.device)
        rn, fn = (d.edge_node0.numpy()[e] if nt else None), (d.face_node0.numpy() if nf else None)
        (self.elem_world, self.params_step, self.rest_step, self._rb, self.rff_freq, self.w_rod, self.has_d_rest,
         self.d_rest) = self._elem(nt, vec10f, vec5f, 5, rn, pw, gm, False)
        (self.shell_elem_world, self.shell_params_step, self.shell_rest_step, self._sb, self.shell_rff_freq,
         self.w_shell, self.has_d_rest_shell, self.d_rest_shell) = self._elem(nf, vec12f, vec6f, 6, fn, pw, gm, True)

    def _elem(self, n: int, p_dt, v_dt, nv: int, nodes, pw, gm, shell: bool):
        dev, sp = self.device, self.spec
        dim, seed, w_v, dr_v, tag = ((sp.shell_rff_dim, sp.seed + 1, sp.w_shell, sp.shell_spatial_rest_delta, "shell")
                                     if shell else (sp.rod_rff_dim, sp.seed, sp.w_rod, sp.spatial_rest_delta, "rod"))
        rff = np.random.default_rng(seed).normal(0.0, sp.rff_bandwidth, (dim, 2 * nv))
        bufs = tuple(wp.empty(n, dtype=v_dt, device=dev, requires_grad=True) for _ in range(4))
        ew = np.zeros(n, np.int32) if (gm or not n) else np.maximum(pw[nodes], 0)
        has_dr = int(dr_v is not None and n > 0)
        return (wp.array(ew.astype(np.int32), dtype=wp.int32, device=dev), wp.empty(n, dtype=p_dt, device=dev),
                wp.empty(n, dtype=v_dt, device=dev), bufs, wp.array(rff.astype(np.float32), dtype=p_dt, device=dev),
                _as_vec_grad(w_v, v_dt, nv, dim, self._nw, dev, f"w_{tag}"), has_dr,
                _as_vec_grad(dr_v if has_dr else None, v_dt, nv, n, None, dev, f"{tag}_spatial_rest_delta"))

    def grad_inputs(self) -> list[wp.array]:
        """Differentiable parameter arrays tracked by ``SolverDiSMech._record``."""
        return [a for c, a in ((self.rod_count or self.spec.w_rod is not None, self.w_rod),
                               (self.has_d_rest, self.d_rest), (self.has_d_rest_shell, self.d_rest_shell),
                               (self.shell_count or self.spec.w_shell is not None, self.w_shell)) if c]

    def _eval(self, shell: bool, el, s, dt: float, pb, rb, po, ro) -> None:
        if not hasattr(el, "bind"):
            el, s, dt, pb, rb, po, ro = (self._shells if shell else self._rods), el, s, dt, pb, rb, po
        if not el.count:
            return
        d_p, d_r, bufs, pk, sk = (
            (self.shell_params_step, self.shell_rest_step, self._sb, _shell_prep_kernel, _shell_constit_step_kernel)
            if shell else (self.params_step, self.rest_step, self._rb, _rod_prep_kernel, _rod_constit_step_kernel))
        po, ro = (d_p if po is None else po), (d_r if ro is None else ro)
        for b in (b for b in (bufs if po.requires_grad or ro.requires_grad else ()) if b.grad is not None):
            b.grad.zero_()
        eb, t = el.bind(s), (1 if self.device.is_cpu else 32, self.clamp_lo, self.clamp_hi, dt)
        pin = ([s.q, s.qd, eb, rb, s.face_strain_q, self.d_rest_shell, self.shell_scales, self.has_d_rest_shell, dt]
               if shell else [s.q, s.qd, eb, rb, s.triplet_strain_q, self.d_rest, self.scales, self.has_d_rest, dt])
        sin = ([pb, el.area, self.shell_rff_freq, self.w_shell, self.shell_elem_world, *bufs, self.shell_scales, *t]
               if shell else [eb, pb, el.dahl, s.triplet_mem_q, self.rff_freq, self.w_rod, self.elem_world,
                              self.edge_share, *bufs, self.scales, self.iso_bending, *t])
        wp.launch(pk, dim=el.count, device=self.device, inputs=pin, outputs=list(bufs))
        wp.launch_tiled(sk, dim=[el.count], block_dim=32, device=self.device, inputs=sin, outputs=[po, ro])

    def eval_rod(self, rods, s, dt: float, params_base, rest_base, params_out=None, rest_out=None) -> None:
        """Compute step-effective ``(params_out, rest_out)`` for DER rods from state ``s``."""
        self._eval(False, rods, s, dt, params_base, rest_base, params_out, rest_out)

    def eval_shell(self, shells, s, dt: float, params_base, rest_base, params_out=None, rest_out=None) -> None:
        """Compute step-effective ``(params_out, rest_out)`` for BAC shells from state ``s``."""
        self._eval(True, shells, s, dt, params_base, rest_base, params_out, rest_out)
