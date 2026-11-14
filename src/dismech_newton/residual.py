"""Pure-Warp constitutive residual operators (Phase 1: 5D DER rods)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from newton import Model

from .builder import vec5f, vec10f
from .rod import Rod, _dkappa, _dphi, _geometry, _oriented, rod_dofs, vec11f

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


def _as_vec_grad(val, vec_dtype, n_vec: int, shapes: tuple[tuple[int, ...], ...], dev, name: str) -> wp.array:
    """Validate/convert a NumPy or Warp weight array to a differentiable vector ``wp.array`` on ``dev``."""
    f_shapes = tuple(s + (n_vec,) for s in shapes)
    if isinstance(val, wp.array):
        if val.dtype == vec_dtype and val.shape in shapes:
            out = val if val.device == dev else wp.array(val.numpy(), dtype=vec_dtype, device=dev, requires_grad=True)
        else:
            if val.dtype == wp.float32 and len(shapes[0]) == 2 and val.shape == (shapes[0][1], n_vec):
                val = val.reshape((1, shapes[0][1], n_vec))
            if val.dtype != wp.float32 or val.shape not in f_shapes:
                raise ValueError(f"{name} shape/dtype mismatch: shape={val.shape}, dtype={val.dtype}")
            if val.device == dev:
                out = val.view(vec_dtype)
            else:
                out = wp.array(val.numpy().astype(np.float32), dtype=vec_dtype, device=dev, requires_grad=True)
        out.requires_grad = True
        return out
    arr = np.asarray(val, dtype=np.float32)
    arr = arr[None] if len(shapes[0]) == 2 and arr.shape == (shapes[0][1], n_vec) else arr
    if arr.shape not in f_shapes:
        raise ValueError(f"{name} shape mismatch: {arr.shape}")
    return wp.array(arr, dtype=vec_dtype, device=dev, requires_grad=True)


@wp.func
def _iso_qh2(u2: float) -> wp.vec2:
    """Singularity-free radial RFF quotient ``q(u^2) = sin(u)/u - exp(-u^2/2)`` and ``2 q'(u^2)``."""
    if u2 < 1.0e-4:
        return wp.vec2(u2 * (1.0 / 3.0 - (7.0 / 60.0) * u2), 2.0 / 3.0 - (7.0 / 15.0) * u2)
    u, e = wp.sqrt(u2), wp.exp(-0.5 * u2)
    q = wp.sin(u) / u - e
    return wp.vec2(q, (wp.cos(u) - (1.0 - u2) * e - q) / u2)


@wp.func
def _clamp_pack(
    pb: vec10f, sig: vec5f, dk_raw: vec5f, dc_raw: vec5f, r0: vec5f, eps_rel: vec5f, eff_rate: vec5f,
    mem: vec5f, on: float, rate: float, clamp_lo: float, clamp_hi: float, dt: float,
):
    """SPD tangent clamping + optional additive Dahl hysteresis across all 5 DER rod channels."""
    p_out, r_out = vec10f(), vec5f()
    for i in range(5):
        k0, c0 = pb[i], pb[5 + i]
        dk = wp.where(k0 > 0.0, wp.clamp(dk_raw[i], clamp_lo * k0, clamp_hi * k0), 0.0)
        k_step = k0 + dk
        dc = wp.where(
            c0 > 0.0,
            wp.clamp(dc_raw[i], clamp_lo * c0, clamp_hi * c0),
            wp.where(k0 > 0.0, wp.clamp(dc_raw[i], 0.0, clamp_hi * k0 * dt), 0.0),
        )
        dahl_t = on * mem[i] / wp.max(k_step, 1.0e-30) if i >= 2 else 0.0
        p_out[i] = k_step
        p_out[5 + i] = c0 + dc + (rate * k0 if i >= 2 else 0.0)
        cor = (sig[i] - dk * eps_rel[i] - dc * eff_rate[i]) / wp.max(k_step, 1.0e-30)
        r_out[i] = wp.where(k_step > 0.0, r0[i] - dahl_t - cor, r0[i])
    return p_out, r_out


@wp.func
def _rod_strain_rate(x: vec11f, xd: vec11f, v: Rod, t: int):
    """Triplet strain and directional strain rate ``J * xd`` without forming the full ``5x11`` matrix."""
    s = v.sign[t]
    g, xo = _geometry(_oriented(x, s), v, t), _oriented(xd, s)
    ze, zf = wp.vec3(xo[4] - xo[0], xo[5] - xo[1], xo[6] - xo[2]), wp.vec3(xo[8] - xo[4], xo[9] - xo[5], xo[10] - xo[6])
    ge, gf = _dphi(g.te_old, g.te, g.ne), _dphi(g.tf_old, g.tf, g.nf)
    De1, Df1, a1, b1 = _dkappa(g, g.strain[2], g.m2e, g.m2f, -g.m1e, -g.m1f)
    De2, Df2, a2, b2 = _dkappa(g, g.strain[3], -g.m1e, -g.m1f, -g.m2e, -g.m2f)
    pe, pf = wp.dot(ge, ze) + xo[3], wp.dot(gf, zf) + xo[7]
    return g.strain, vec5f(
        wp.dot(g.te, ze) / g.l0e,
        wp.dot(g.tf, zf) / g.l0f,
        wp.dot(De1, ze) + wp.dot(Df1, zf) + a1 * pe + b1 * pf,
        wp.dot(De2, ze) + wp.dot(Df2, zf) + a2 * pe + b2 * pf,
        0.5 * wp.dot(g.kb / g.ne, ze) + 0.5 * wp.dot(g.kb / g.nf, zf) - pe + pf,
    )


@wp.func
def _rff_loop(
    z_a: vec5f, z_b: vec5f, fr: wp.array[vec10f], wr: wp.array2d[vec5f],
    w_idx: int, R_: int, lane: int, stride: int, c_norm: float, iso_bending: int,
):
    """Strided origin-orthogonal odd RFF sum in a helper function (avoids Warp kernel-loop AD bugs)."""
    r2_k, r2_v = z_a[2] * z_a[2] + z_a[3] * z_a[3], z_b[2] * z_b[2] + z_b[3] * z_b[3]
    sp, dk, dc = vec5f(0.0), vec5f(0.0), vec5f(0.0)
    for k in range(wp.where(lane < R_, (R_ - lane + stride - 1) // stride, 0)):
        f, w = fr[lane + k * stride], wr[w_idx, lane + k * stride]
        u = f[0] * z_a[0] + f[1] * z_a[1] + f[4] * z_a[4] + f[5] * z_b[0] + f[6] * z_b[1] + f[9] * z_b[4]
        if iso_bending == 0:
            u = u + f[2] * z_a[2] + f[3] * z_a[3] + f[7] * z_b[2] + f[8] * z_b[3]
        eu = wp.exp(-0.5 * u * u)
        psi, dpsi = c_norm * (wp.sin(u) - u * eu), c_norm * (wp.cos(u) - (1.0 - u * u) * eu)
        s2, s3 = w[2] * psi, w[3] * psi
        k2, k3, c2, c3 = w[2] * f[2] * dpsi, w[3] * f[3] * dpsi, w[2] * f[7] * dpsi, w[3] * f[8] * dpsi
        if iso_bending != 0:
            om, ov = f[2], f[7]
            qk, qv, ck, cv = _iso_qh2(om * om * r2_k), _iso_qh2(ov * ov * r2_v), c_norm * w[2] * om, c_norm * w[3] * ov
            s2, s3 = ck * qk[0] * z_a[2] + cv * qv[0] * z_b[2], ck * qk[0] * z_a[3] + cv * qv[0] * z_b[3]
            k2, k3 = ck * (qk[0] + om * om * z_a[2] * z_a[2] * qk[1]), ck * (qk[0] + om * om * z_a[3] * z_a[3] * qk[1])
            c2, c3 = cv * (qv[0] + ov * ov * z_b[2] * z_b[2] * qv[1]), cv * (qv[0] + ov * ov * z_b[3] * z_b[3] * qv[1])
        sp = sp + vec5f(w[0] * psi, w[1] * psi, s2, s3, w[4] * psi)
        dk = dk + vec5f(w[0] * f[0] * dpsi, w[1] * f[1] * dpsi, k2, k3, w[4] * f[4] * dpsi)
        dc = dc + vec5f(w[0] * f[5] * dpsi, w[1] * f[6] * dpsi, c2, c3, w[4] * f[9] * dpsi)
    return sp, dk, dc


@wp.kernel(module="unique", module_options=_ADJ_OPTS)
def _rod_prep_kernel(
    q: wp.array[float], qd: wp.array[float], v: Rod, rest_base: wp.array[vec5f], strain_q: wp.array[vec5f],
    d_rest: wp.array[vec5f], scales: vec10f, has_d_rest: int, dt: float,
    za_out: wp.array[vec5f], zb_out: wp.array[vec5f], ef_out: wp.array[vec5f], r0_out: wp.array[vec5f],
):
    """Compute dimensionless triplet strain/rate coordinates ``(z_a, z_b)`` and rest state per rod triplet."""
    t = wp.tid()
    d = rod_dofs(v, t)
    x = vec11f(q[d[0]], q[d[1]], q[d[2]], q[d[3]], q[d[4]], q[d[5]], q[d[6]], q[d[7]], q[d[8]], q[d[9]], q[d[10]])
    xd = vec11f(
        qd[d[0]], qd[d[1]], qd[d[2]], qd[d[3]], qd[d[4]],
        qd[d[5]], qd[d[6]], qd[d[7]], qd[d[8]], qd[d[9]], qd[d[10]],
    )
    eps, eps_dot = _rod_strain_rate(x, xd, v, t)
    eps_prev, r0 = strain_q[t], (rest_base[t] + d_rest[t] if has_d_rest != 0 else rest_base[t])
    ef_out[t] = vec5f(0.0) if wp.isnan(eps_prev[0]) else (eps - eps_prev) / dt
    eps_rel, c = eps - r0, v.conn[t]
    dl = 0.5 * (float(v.edge_length[c[0]]) + float(v.edge_length[c[1]]))
    za_out[t] = vec5f(
        eps_rel[0] / scales[0], eps_rel[1] / scales[1],
        eps_rel[2] / (dl * scales[2]), eps_rel[3] / (dl * scales[3]), eps_rel[4] / (dl * scales[4]),
    )
    zb_out[t] = vec5f(
        eps_dot[0] / scales[5], eps_dot[1] / scales[6],
        eps_dot[2] / (dl * scales[7]), eps_dot[3] / (dl * scales[8]), eps_dot[4] / (dl * scales[9]),
    )
    r0_out[t] = r0


@wp.kernel(module="unique", module_options=_ADJ_OPTS)
def _rod_constit_step_kernel(
    v: Rod, params_base: wp.array[vec10f], dahl: wp.array[wp.vec2], mem: wp.array[vec5f], rff_freq: wp.array[vec10f],
    w_rod: wp.array2d[vec5f], elem_world: wp.array[wp.int32], edge_share: wp.array[wp.vec2], za_in: wp.array[vec5f],
    zb_in: wp.array[vec5f], ef_in: wp.array[vec5f], r0_in: wp.array[vec5f], scales: vec10f, iso_bending: int,
    stride: int, clamp_lo: float, clamp_hi: float, dt: float, params_out: wp.array[vec10f], rest_out: wp.array[vec5f],
):
    """Tiled 5D DER rod RFF evaluation, warp-tile reduction, SPD tangent clamping, and additive Dahl."""
    t, lane = wp.tid()
    R_ = rff_freq.shape[0]
    w_idx, z_a, cn = wp.where(w_rod.shape[0] > 1, elem_world[t], 0), za_in[t], wp.sqrt(2.0 / float(R_))
    sp, dk, dc = _rff_loop(z_a, zb_in[t], rff_freq, w_rod, w_idx, R_, lane, stride, cn, iso_bending)
    sp0, sp1, sp2 = wp.tile_sum(wp.tile(sp[0]))[0], wp.tile_sum(wp.tile(sp[1]))[0], wp.tile_sum(wp.tile(sp[2]))[0]
    sp3, sp4 = wp.tile_sum(wp.tile(sp[3]))[0], wp.tile_sum(wp.tile(sp[4]))[0]
    dk0, dk1, dk2 = wp.tile_sum(wp.tile(dk[0]))[0], wp.tile_sum(wp.tile(dk[1]))[0], wp.tile_sum(wp.tile(dk[2]))[0]
    dk3, dk4 = wp.tile_sum(wp.tile(dk[3]))[0], wp.tile_sum(wp.tile(dk[4]))[0]
    dc0, dc1, dc2 = wp.tile_sum(wp.tile(dc[0]))[0], wp.tile_sum(wp.tile(dc[1]))[0], wp.tile_sum(wp.tile(dc[2]))[0]
    dc3, dc4 = wp.tile_sum(wp.tile(dc[3]))[0], wp.tile_sum(wp.tile(dc[4]))[0]
    if lane == 0:
        c, sv = v.conn[t], edge_share[t]
        le, lf = float(v.edge_length[c[0]]), float(v.edge_length[c[1]])
        dl, ls0, ls1 = 0.5 * (le + lf), sv[0] * le, sv[1] * lf
        eps_rel = vec5f(
            z_a[0] * scales[0], z_a[1] * scales[1],
            z_a[2] * dl * scales[2], z_a[3] * dl * scales[3], z_a[4] * dl * scales[4],
        )
        sig_v = vec5f(ls0 * sp0, ls1 * sp1, sp2, sp3, sp4)
        dk_v = vec5f(
            ls0 * dk0 / scales[0], ls1 * dk1 / scales[1],
            dk2 / (dl * scales[2]), dk3 / (dl * scales[3]), dk4 / (dl * scales[4]),
        )
        dc_v = vec5f(
            ls0 * dc0 / scales[5], ls1 * dc1 / scales[6],
            dc2 / (dl * scales[7]), dc3 / (dl * scales[8]), dc4 / (dl * scales[9]),
        )
        dv = dahl[t]
        on = wp.where(dv[0] > 0.0 and dv[1] > 0.0, 1.0, 0.0)
        p_step, r_step = _clamp_pack(
            params_base[t], sig_v, dk_v, dc_v, r0_in[t], eps_rel, ef_in[t], mem[t],
            on, on * dt * dv[0] / wp.max(dv[1], 1.0e-30), clamp_lo, clamp_hi, dt,
        )
        params_out[t] = p_step
        rest_out[t] = r_step


class ConstitutiveResidual:
    """Pre-allocated pure-Warp 5D DER rod constitutive residual evaluator."""

    def __init__(self, model: Model, spec: ResidualSpec, device=None):
        if spec.rod_rff_dim <= 0 or len(spec.rod_scales) != 10 or any(s <= 0.0 for s in spec.rod_scales):
            raise ValueError("Invalid rod_rff_dim or rod_scales in ResidualSpec")
        if not (-1.0 < spec.clamp_lo <= spec.clamp_hi):
            raise ValueError(f"require -1 < clamp_lo <= clamp_hi, got ({spec.clamp_lo}, {spec.clamp_hi})")
        self.spec, self.device = spec, wp.get_device(model.device if device is None else device)
        self.iso_bending = int(bool(spec.iso_bending))
        self.clamp_lo, self.clamp_hi = float(spec.clamp_lo), float(spec.clamp_hi)
        self.scales, self._rods = vec10f(*[float(s) for s in spec.rod_scales]), None
        self._stride = 1 if self.device.is_cpu else 32
        d = model.dismech
        e, f = d.triplet_edge0.numpy(), d.triplet_edge1.numpy()
        self.rod_count = n_t = int(e.shape[0])
        pw = model.particle_world.numpy() if model.particle_count else np.zeros(0, dtype=np.int32)
        gm = model.world_count > 1 and (pw < 0).any()
        num_w = 1 if (gm or not pw.size) else int(np.maximum(pw, 0).max(initial=0)) + 1
        n_e = d.edge_length.shape[0]
        cnt = np.bincount(np.concatenate([e, f]), minlength=n_e) if n_t else np.zeros(0, dtype=np.int64)
        share = (np.column_stack([1.0 / cnt[e], 1.0 / cnt[f]]) if n_t else np.zeros((0, 2))).astype(np.float32)
        ew = np.zeros(n_t, dtype=np.int32) if (gm or not n_t) else np.maximum(pw[d.edge_node0.numpy()[e]], 0)
        self.edge_share = wp.array(share, dtype=wp.vec2, device=self.device)
        self.elem_world = wp.array(ew.astype(np.int32), dtype=wp.int32, device=self.device)
        self.params_step = wp.empty(n_t, dtype=vec10f, device=self.device)
        self.rest_step = wp.empty(n_t, dtype=vec5f, device=self.device)
        self._za, self._zb, self._ef, self._r0 = (
            wp.empty(n_t, dtype=vec5f, device=self.device, requires_grad=True) for _ in range(4)
        )
        dim = spec.rod_rff_dim
        freq = np.random.default_rng(spec.seed).normal(0.0, spec.rff_bandwidth, size=(dim, 10)).astype(np.float32)
        self.rff_freq = wp.array(freq, dtype=vec10f, device=self.device)
        if spec.w_rod is None:
            self.w_rod = wp.zeros((1, dim), dtype=vec5f, device=self.device, requires_grad=True)
        else:
            self.w_rod = _as_vec_grad(spec.w_rod, vec5f, 5, ((1, dim), (num_w, dim)), self.device, "w_rod")
        has_dr = spec.spatial_rest_delta is not None and n_t > 0
        self.has_d_rest = int(has_dr)
        self.d_rest = (
            _as_vec_grad(spec.spatial_rest_delta, vec5f, 5, ((n_t,),), self.device, "spatial_rest_delta")
            if has_dr
            else wp.zeros(1, dtype=vec5f, device=self.device, requires_grad=False)
        )

    def grad_inputs(self) -> list[wp.array]:
        """Differentiable parameter arrays tracked by ``SolverDiSMech._record``."""
        return [self.w_rod, self.d_rest] if self.has_d_rest != 0 else [self.w_rod]

    def eval_rod(self, rods, s, dt: float, params_base, rest_base, params_out, rest_out=None) -> None:
        """Compute step-effective ``(params_out, rest_out)`` for DER rods from state ``s``."""
        if not hasattr(rods, "bind"):
            rods, s, dt, params_base, rest_base, params_out, rest_out = (
                self._rods, rods, s, dt, params_base, rest_base, params_out,
            )
        if not rods.count:
            return
        if params_out.requires_grad or (rest_out is not None and rest_out.requires_grad):
            for buf in (self._za, self._zb, self._ef, self._r0):
                if buf.grad is not None:
                    buf.grad.zero_()
        rb = rods.bind(s)
        wp.launch(
            _rod_prep_kernel, dim=rods.count, device=self.device,
            inputs=[s.q, s.qd, rb, rest_base, s.triplet_strain_q, self.d_rest, self.scales, self.has_d_rest, dt],
            outputs=[self._za, self._zb, self._ef, self._r0],
        )
        wp.launch_tiled(
            _rod_constit_step_kernel, dim=[rods.count], block_dim=32, device=self.device,
            inputs=[
                rb, params_base, rods.dahl, s.triplet_mem_q, self.rff_freq, self.w_rod, self.elem_world,
                self.edge_share, self._za, self._zb, self._ef, self._r0, self.scales, self.iso_bending,
                self._stride, self.clamp_lo, self.clamp_hi, dt,
            ],
            outputs=[params_out, rest_out],
        )
