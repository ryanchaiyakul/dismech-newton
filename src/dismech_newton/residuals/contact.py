"""Single-dataclass O(M) compositional tribology and surface-embedding contact residual."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp
from newton import Model

from ..constants import ADJ_MODULE_OPTS, vec10f, vec11f
from ..contact import (
    Arm,
    Bodies,
    Mesh,
    Offset,
    Slots,
    _slot_edge_t,
    barrier,
    contact_point,
    pad_energy,
    shape_of,
    slot_map,
    slot_surface_pair,
)
from ..dofs import node
from .base import as_rff_weights


@wp.func
def _mt_add(u: wp.vec3, al: float, b1: wp.vec3, b2: wp.vec3) -> wp.vec2:
    p1, p2 = wp.dot(u, b1), wp.dot(u, b2)
    return wp.where(
        p1 * p1 + p2 * p2 > 1.0e-12,
        al * wp.vec2(p1 * p1 - p2 * p2, 2.0 * p1 * p2) / wp.max(p1 * p1 + p2 * p2, 1.0e-12),
        wp.vec2(0.0),
    )


@wp.func
def _slot_vrel(
    s: Slots,
    qd: wp.array[float],
    poses: wp.array[wp.transform],
    b: Bodies,
    bqd: wp.array[wp.spatial_vector],
    key0: int,
    i: int,
) -> wp.vec3:
    vrel, vs = wp.vec3(0.0), wp.vec3(0.0)
    for k in range(4):
        if s.pair[i][k] >= 0:
            vrel += slot_map(s, i, k) * node(qd, s.pair[i][k])
    if s.key[i] >= key0:
        jb = b.body[shape_of(s.key[i], key0)]
        if jb >= 0 and bqd.shape[0] > jb:
            bv, lv = bqd[jb], s.anchor[i] - wp.transform_point(poses[jb], b.body_com[jb])
            vs = wp.spatial_top(bv) + wp.cross(wp.spatial_bottom(bv), lv)
    return vrel - vs


@wp.func
def _contact_rff(
    e: wp.array3d[float],
    fe: wp.array[vec11f],
    fu: wp.array[wp.vec4],
    fb: wp.array[float],
    wc: wp.array3d[float],
    we: int,
    a: int,
    b: int,
    uv: wp.vec4,
    ur: wp.vec4,
    ww: int,
    rff_dim: int,
) -> wp.vec2:
    eta = vec11f(
        e[we, a, 0] * e[we, b, 0],
        e[we, a, 1] * e[we, b, 1],
        e[we, a, 2] * e[we, b, 4] + e[we, b, 2] * e[we, a, 4],
        e[we, a, 3] * e[we, b, 5] + e[we, b, 3] * e[we, a, 5],
        e[we, a, 0] + e[we, b, 0],
        e[we, a, 1] + e[we, b, 1],
        e[we, a, 2] + e[we, b, 2],
        e[we, a, 3] + e[we, b, 3],
        e[we, a, 4] + e[we, b, 4],
        e[we, a, 5] + e[we, b, 5],
        e[we, a, 6] + e[we, b, 6],
    )
    dm, dp = float(0.0), float(0.0)
    for r in range(rff_dim):
        p0, fur = wp.dot(fe[r], eta) + fb[r], fu[r]
        dm += wc[ww, r, 0] * (wp.cos(p0 + wp.dot(fur, uv)) - wp.cos(p0 + wp.dot(fur, ur)))
        dp += wc[ww, r, 1] * (wp.cos(p0 + fur[3] * uv[3]) - wp.cos(p0) + uv[3] * fur[3] * wp.sin(p0))
    return wp.vec2(dm, dp)


def _make_contact_kernel(replay: bool):
    @wp.kernel(module="unique", module_options=ADJ_MODULE_OPTS)
    def _kern(
        s: Slots,
        q: wp.array[float],
        q0: wp.array[float],
        dt: float,
        qd: wp.array[float],
        m: Mesh,
        b: Bodies,
        ps: wp.array[wp.transform],
        bq: wp.array[wp.spatial_vector],
        nn: int,
        key0: int,
        em: wp.array3d[float],
        fe: wp.array[vec11f],
        fu: wp.array[wp.vec4],
        fb: wp.array[float],
        wc: wp.array3d[float],
        rff_dim: int,
        rp: vec10f,
        mo: wp.array[float],
        po: wp.array[wp.vec4],
        ao: wp.array[wp.vec4],
    ):
        i = wp.tid()
        if s.active[i] == 0:
            return
        n, th, p, kh = s.normal[i], s.thickness[i], s.pad[i], s.base[i][1]
        d, rho, ta, tb = wp.dot(n, s.c0[i]), float(s.rho[i]), wp.vec3(s.ta[i]), wp.vec3(s.tb[i])
        if wp.static(replay):
            dq = contact_point(q, s.pair[i], s.w[i], wp.vec3(0.0))
            d = wp.dot(n, s.c0[i] + dq - contact_point(q0, s.pair[i], s.w[i], wp.vec3(0.0)))
            rho = -barrier(d, th, s.stiffness[i], s.lam[i])[1] - pad_energy(d, th, wp.vec4(kh, p[1], p[2], p[3]))[1]
            ta, tb = _slot_edge_t(q, s.pair[i], s.w[i], n, 0, ta), _slot_edge_t(q, s.pair[i], s.w[i], n, 1, tb)

        we = wp.max(s.world[i], 0) if em.shape[0] > 1 else 0
        ww = wp.max(s.world[i], 0) if wc.shape[0] > 1 else 0
        ma, mb, ua, ub = slot_surface_pair(q, ps, m, b, nn, key0, s.own[i], s.key[i], ta, tb)
        ma, mb = wp.clamp(ma, 0, em.shape[1] - 1), wp.clamp(mb, 0, em.shape[1] - 1)
        b1 = wp.normalize(wp.cross(n, wp.where(wp.abs(n[0]) < 0.9, wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0))))
        b2 = wp.cross(n, b1)
        ab = _mt_add(ua, em[we, ma, 7], b1, b2) + _mt_add(ub, em[we, mb, 7], b1, b2)
        a, bt, ani = ab[0], ab[1], wp.vec4(0.0)
        if a * a + bt * bt > 1.0e-16 and rp[5] > 0.0:
            r = wp.sqrt(a * a + bt * bt)
            vx, vy = wp.where(a >= 0.0, r + a, bt), wp.where(a >= 0.0, bt, r - a)
            e1 = (vx * b1 + vy * b2) / wp.sqrt(wp.max(vx * vx + vy * vy, 1.0e-24))
            ani = wp.vec4(e1[0], e1[1], e1[2], wp.min(r, rp[5]))

        vrel = _slot_vrel(s, qd, ps, b, bq, key0, i)
        v1, v2, iv2 = wp.dot(vrel, b1), wp.dot(vrel, b2), 1.0 / (rp[1] * rp[1])
        sv = (v1 * v1 + v2 * v2) * iv2
        san = (a * (v1 * v1 - v2 * v2) + 2.0 * bt * v1 * v2) * iv2
        dt_n = wp.max(0.0, th + p[2] - d) / wp.max(wp.max(p[2], th), 1.0e-5)
        uv = wp.vec4(wp.log(1.0 + wp.max(rho, 0.0) / rp[0]), sv, san, dt_n)
        ur = wp.vec4(wp.log(2.0), 1.0, 0.0, 0.0)
        dg = _contact_rff(em, fe, fu, fb, wc, we, ma, mb, uv, ur, ww, rff_dim)
        z = rp[6] * dg[0] + rp[7]
        sp = wp.max(z, 0.0) + wp.log(1.0 + wp.exp(-wp.abs(z)))
        s0 = rp[7] + wp.log(1.0 + wp.exp(-rp[7]))
        pe = wp.vec4(kh * wp.clamp(1.0 + rp[6] * dg[1], rp[3], rp[4]), p[1], p[2], p[3]) if kh > 0.0 else p
        mo[i] = s.base[i][0] * wp.max(rp[2], 1.0 + (1.0 - rp[2]) * (sp / s0 - 1.0))
        po[i] = pe
        ao[i] = ani
        if wp.static(not replay) and kh > 0.0 and dg[1] != 0.0:
            pf = -pad_energy(d, th, pe)[1]
            s.damp[i] = pe[3] * pf / dt
            s.rho[i] = -barrier(d, th, s.stiffness[i], s.lam[i])[1] + pf

    return _kern


_contact_apply_kernel, _contact_replay_kernel = _make_contact_kernel(False), _make_contact_kernel(True)


@dataclass
class ContactResidual:
    """Pure-Warp O(M) compositional tribology and surface-embedding contact residual."""

    rff_dim: int = 32
    rff_bandwidth: float = 1.5
    rho_ref: float = 1.0
    v_ref: float = 0.1
    mu_min_mult: float = 0.1
    pad_min_mult: float = 0.1
    pad_max_mult: float = 8.0
    gamma_max: float = 1.5
    num_materials: int = 0
    embeddings: np.ndarray | wp.array | None = None
    w_contact: np.ndarray | wp.array | None = None
    seed: int = 29

    device: Any = field(default=None, init=False, repr=False)
    fr_eta: wp.array | None = field(default=None, init=False, repr=False)
    fr_u: wp.array | None = field(default=None, init=False, repr=False)
    fr_b: wp.array | None = field(default=None, init=False, repr=False)
    params: tuple[wp.array, ...] = field(default=(), init=False, repr=False)
    _rp: vec10f = field(default_factory=vec10f, init=False, repr=False)
    _src: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _bound: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            self.rff_dim < 0
            or self.rff_bandwidth < 0.0
            or min(self.rho_ref, self.v_ref, self.pad_min_mult) <= 0.0
            or not (0.0 < self.mu_min_mult <= 1.0)
            or self.pad_max_mult < self.pad_min_mult
            or self.gamma_max < 0.0
            or self.num_materials < 0
        ):
            raise ValueError("Invalid ContactResidual parameters")

    def bind(self, model: Model, device=None) -> "ContactResidual":
        """Allocate/bind compositional contact residual buffers onto ``model`` on ``device``."""
        self._src = {"embeddings": self.embeddings, "w_contact": self.w_contact}
        f = getattr(model, "_parent", model)
        dev = wp.get_device(f.device if device is None else device)
        self.device = dev

        mid = max(
            [0]
            + [
                int(a.numpy().max())
                for k in ("surface_id", "shape_surface_id")
                if (a := getattr(getattr(f, "dismech", None), k, None)) is not None and a.shape[0]
            ]
        )
        em = 0 if (e := self.embeddings) is None else (e.shape if isinstance(e, wp.array) else np.shape(e))[-2]
        if e is not None and em <= mid:
            raise ValueError(f"embeddings has {em} materials, need > max surface_id {mid}")

        has_global = bool((nw := f.world_count) > 1 and f.particle_count and (f.particle_world.numpy() < 0).any())
        n_worlds = 1 if has_global else max(1, nw)
        n_mat = max(1, mid + 1, int(self.num_materials), em)
        n_rff = max(int(self.rff_dim), 1)
        rng = np.random.default_rng(self.seed)
        raw = rng.normal(0.0, self.rff_bandwidth, (n_rff, 15)).astype(np.float32)
        u0 = rng.uniform(0.0, 2.0 * np.pi, n_rff).astype(np.float32)

        self.fr_eta = wp.array(raw[:, :11], dtype=vec11f, device=dev)
        self.fr_u = wp.array(raw[:, 11:], dtype=wp.vec4, device=dev)
        self.fr_b = wp.array(u0, dtype=float, device=dev)
        self.embeddings = as_rff_weights(e, float, n_mat, 8, n_worlds, dev, "embeddings")
        self.w_contact = as_rff_weights(self.w_contact, float, n_rff, 2, n_worlds, dev, "w_contact")
        self._rp = vec10f(
            self.rho_ref,
            self.v_ref,
            self.mu_min_mult,
            self.pad_min_mult,
            self.pad_max_mult,
            self.gamma_max,
            (2.0 / self.rff_dim) ** 0.5 if self.rff_dim else 0.0,
            float(np.log(np.e - 1.0)),
            0.0,
            0.0,
        )
        self.params = (self.w_contact, self.embeddings)
        self._bound = True
        return self

    def _launch(self, kern, dim: int, outs, s: Slots, q, q0, dt: float, qd, ps, bq, src: Offset, *tail) -> None:
        wp.launch(
            kern,
            dim,
            device=self.device,
            outputs=outs,
            inputs=[
                s,
                q,
                q0,
                dt,
                qd,
                src.scene.m,
                src.bodies,
                ps,
                bq,
                src.num_nodes,
                src.shape_key0,
                self.embeddings,
                self.fr_eta,
                self.fr_u,
                self.fr_b,
                self.w_contact,
                int(self.rff_dim),
                self._rp,
                *tail,
            ],
        )

    def apply_contact(self, s: Slots, state_in, source: Offset, arm: Arm, dt: float) -> None:
        """Modulate active contact slots once per step at step start (pure-Warp, FrameGraph-safe)."""
        q = state_in.dismech.q
        b = source.bodies
        self._launch(_contact_apply_kernel, s.active.shape[0], [], s, q, q, dt, state_in.dismech.qd, b.body_q, b.body_qd, source, s.mu, s.pad, s.aniso)

    def replay_contact(self, c: Slots, q_in, qd_in, poses, bqd, source: Offset, dt: float, q_in0=None):
        """Differentiably replay modulated ``(mu, pad, aniso)`` for ``StepAdjoint``."""
        q0 = wp.array(ptr=q_in.ptr, dtype=float, shape=q_in.shape, device=q_in.device) if q_in0 is None else q_in0
        mu, pad, ani = (wp.zeros(len(c.active), t, self.device, requires_grad=True) for t in (float, wp.vec4, wp.vec4))
        self._launch(_contact_replay_kernel, len(c.active), [mu, pad, ani], c, q_in, q0, dt, qd_in, poses, bqd, source)
        return mu, pad, ani

    def grad_inputs(self) -> list[wp.array]:
        """Differentiable parameter arrays tracked by adjoint."""
        return [self.w_contact, self.embeddings]
