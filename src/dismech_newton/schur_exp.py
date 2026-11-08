"""EXPERIMENT (proxy-coupling-principled idea 1): the proxies' interface inertia from the source's step response.

``DM_SCHUR=full|body|diag`` (installed by ``__init__``). Each coupled step, before the coupler's passes, the source
(MuJoCo) is stepped ``k + 1`` times from the same input with the proxy forces ``F0`` and ``F0 + delta e_j``: the
proxies' end velocities give ``R = dv/dF`` per world, and the source's step operator condensed on the proxy DOFs is
``Minv_s = R / h`` (``J A^-1 J^T``: articulation, implicit damping, constraints). ProxyArm then uses
``M = (Minv_s + eps Minv_view)^-1`` (``full``: the world's k x k block; ``body``: its 6 x 6 blocks; ``diag``), its pred
rewinds the applied force with ``Minv_s`` (``x0 + h v1 - h^2 Minv_s F``: the source's own free end pose) and the
harvest is ``M (z - pred) / h^2``. Env: ``DM_SCHUR_EPS`` (0.01), ``DM_SCHUR_DELTA`` (1 N; torques 0.05x),
``DM_SCHUR_EVERY`` (1: probe every step)."""

import os

import numpy as np
from newton import StateFlags

MODE = os.environ.get("DM_SCHUR", "")
EPS = float(os.environ.get("DM_SCHUR_EPS", "0.01"))
DELTA = float(os.environ.get("DM_SCHUR_DELTA", "1.0"))
EVERY = int(os.environ.get("DM_SCHUR_EVERY", "1"))
ROBIN = os.environ.get("DM_ROBIN", "")  # the source takes F(x1) = F_ref + K (x1 - x_ref), implicitly: 1 by the
# probes before the passes; 2 in our harvest (no probes: view inertia, the source's free velocity carried over)
STATS = {"probes": 0}


def _maps(arm, proxy):
    """Per arm proxy p: its source-local body and its column block (world, 6 slot)."""
    if getattr(arm, "_schur_maps", None) is None:
        d2g = proxy.destination_local_to_proxy_global.numpy()
        s2g = proxy.source_local_to_proxy_global.numpy()
        g2s = {int(g): i for i, g in enumerate(s2g) if g >= 0}
        body = arm._body.numpy()
        glob = np.array([d2g[b] for b in body])
        src = np.array([g2s[int(g)] for g in glob])
        arm._schur_maps = (body, glob, src, arm._world, arm._slot)
    return arm._schur_maps


def _probe(coupler, state_in, control, contacts, dt):
    for (src_name, dst_name), group in coupler._proxy_groups.items():
        if not group["bodies"]:
            continue
        src, dst = coupler._entries[src_name], coupler._entries[dst_name]
        arm = getattr(dst.solver, "_proxy_arm", None)
        if arm is None or not arm.k:
            continue
        proxy = group["bodies"][0]
        n = getattr(arm, "_schur_steps", 0)
        arm._schur_steps = n + 1
        if n % EVERY and getattr(arm, "_schur", None) is not None:
            continue
        body, glob, sidx, world, slot = _maps(arm, proxy)
        W, k = arm._arm.M.shape[0], arm.k
        F0 = proxy.coupling_forces.numpy().copy()
        restart = arm.restart
        q0 = src.state_0.body_q.numpy()[sidx].astype(np.float64)  # the synced start poses (freshly distributed)

        def run(F):
            coupler._distribute_state(state_in, dt=dt, iteration_restart=True)
            proxy.coupling_forces.assign(F)
            coupler._clear_body_force_input(src)
            coupler._add_body_force_input(src, src.body_local_to_global, state_in.body_f)
            coupler._add_body_force_input(src, proxy.source_local_to_proxy_global, proxy.coupling_forces)
            coupler._notify_input_state_update(src, StateFlags.BODY_F, dt=dt)
            coupler._step_entry(src, control, contacts, dt)
            return src.state_1.body_qd.numpy()[sidx].astype(np.float64)

        v0 = run(F0)
        R = np.zeros((W, k, k))
        for s in range(k // 6):
            on = slot == s
            for r in range(6):
                d = DELTA if r < 3 else 0.05 * DELTA
                F = F0.copy()
                F[glob[on], r] += d
                dv = (run(F) - v0) / d  # [P, 6]
                for p in range(len(body)):
                    R[world[p], 6 * slot[p]:6 * slot[p] + 6, 6 * s + r] = dv[p]
        STATS["probes"] += k + 1
        Fg = F0
        if ROBIN == "1" and getattr(arm, "_robin_K", None) is not None and getattr(arm, "_robin_pose", None) is not None:
            K, (com_ref, q_ref) = arm._robin_K, arm._robin_pose
            com0 = q0[:, :3] + _qrot(q0[:, 3:], arm._com_local)
            dx = np.concatenate([com0 - com_ref, _rotvec(_qmul(q0[:, 3:], _qconj(q_ref)))], axis=1)
            F0s, xs, vs = (np.zeros((W, k)) for _ in range(3))
            for p in range(len(body)):
                sl = slice(6 * slot[p], 6 * slot[p] + 6)
                F0s[world[p], sl], xs[world[p], sl], vs[world[p], sl] = F0[glob[p]], dx[p], v0[p]
            Fs = np.empty_like(F0s)
            for w in range(W):  # (I - h K R) F = F_ref + K (dx + h v0) - h K R F0
                KR = dt * K[w] @ R[w]
                Fs[w] = np.linalg.solve(np.eye(k) - KR, F0s[w] + K[w] @ (xs[w] + dt * vs[w]) - KR @ F0s[w])
            Fg = F0.copy()
            for p in range(len(body)):
                Fg[glob[p]] = Fs[world[p], 6 * slot[p]:6 * slot[p] + 6]
            STATS["robin"] = STATS.get("robin", 0) + 1
        proxy.coupling_forces.assign(Fg)
        coupler._distribute_state(state_in, dt=dt, iteration_restart=True)
        arm.restart = restart
        arm._schur = 0.5 * (R + R.transpose(0, 2, 1)) / dt
        arm._schur_asym = np.abs(R - R.transpose(0, 2, 1)).max() / max(np.abs(R).max(), 1e-30)


def _qmul(a, b):
    v1, w1, v2, w2 = a[:, :3], a[:, 3:], b[:, :3], b[:, 3:]
    return np.concatenate([w1 * v2 + w2 * v1 + np.cross(v1, v2), w1 * w2 - (v1 * v2).sum(1, keepdims=True)], axis=1)


def _qconj(q):
    return np.concatenate([-q[:, :3], q[:, 3:]], axis=1)


def _qrot(q, v):
    return _qmul(_qmul(q, np.concatenate([v, np.zeros((len(v), 1))], axis=1)), _qconj(q))[:, :3]


def _rotvec(q):
    q = np.where(q[:, 3:] < 0, -q, q)
    s = np.linalg.norm(q[:, :3], axis=1, keepdims=True)
    return np.where(s > 1e-12, 2 * np.arctan2(s, q[:, 3:]) / np.maximum(s, 1e-30), 2.0) * q[:, :3]


def _select(S):
    if MODE == "full":
        return S
    out = np.zeros_like(S)
    for b in range(S.shape[1] // 6):
        sl = slice(6 * b, 6 * b + 6)
        out[:, sl, sl] = S[:, sl, sl] if MODE == "body" else np.einsum("wii->wi", S[:, sl, sl])[:, :, None] * np.eye(6)
    return out


def _robin_harvest(arm, local_to_global, out_body_f):
    """The next step's force for the source, ``F = F_ref + K (x1 + h v(F) - x_ref)``, ``v(F) = v_free + h Minv F``:
    x1 the source's end pose this step (synced pose + h v1), v_free = v1 + h a_own with the source's own
    acceleration carried over: a_own = (v1 - v0) / h - Minv F_applied (v0 the last step's v1)."""
    q, qd, h, Minv = arm._sync
    within = arm._robin_pass < int(os.environ.get("DM_PASSES", "1")) - 1  # a pass that the step repeats
    qd0 = getattr(arm, "_robin_v0", qd)
    if not within:
        arm._robin_v0 = qd
    com_ref, q_ref = arm._robin_pose
    K = arm._robin_K
    W, k = K.shape[0], arm.k
    body, world, slot = arm._body.numpy(), arm._world, arm._slot
    glob = local_to_global.numpy()[body]
    com_local = arm.model.body_com.numpy()[body].astype(np.float64)
    com1 = q[:, :3] + _qrot(q[:, 3:], com_local) + h * qd[:, :3]
    om = h * qd[:, 3:]
    ang = np.linalg.norm(om, axis=1, keepdims=True)
    turn = np.concatenate([np.where(ang > 0, np.sin(ang / 2) / np.maximum(ang, 1e-30), 0.5) * om, np.cos(ang / 2)], 1)
    q1 = _qmul(turn, q[:, 3:])
    dx = np.concatenate([com1 - com_ref, _rotvec(_qmul(q1, _qconj(q_ref)))], axis=1)
    Fg = out_body_f.numpy()
    Fref, xs, vs, v0s = (np.zeros((W, k)) for _ in range(4))
    Fa = getattr(arm, "_schur_F", np.zeros((W, k)))
    for p in range(len(body)):
        sl = slice(6 * slot[p], 6 * slot[p] + 6)
        Fref[world[p], sl], xs[world[p], sl], vs[world[p], sl], v0s[world[p], sl] = Fg[glob[p]], dx[p], qd[p], qd0[p]
    out = Fg.copy()
    debt = os.environ.get("DM_ROBIN_DEBT") == "1"  # momentum: the source pays back what it took beyond our contact
    if debt:
        arm._robin_debt = getattr(arm, "_robin_debt", np.zeros((W, k))) + Fa - Fref
    for w in range(W):
        if within:  # the same step again: x1(F) = x1 + h^2 Minv (F - F_applied)
            rhs = Fref[w] + K[w] @ (xs[w] - h * h * Minv[w] @ Fa[w])
        else:  # the next step: x2(F) = x1 + h v_free + h^2 Minv F
            rhs = Fref[w] + K[w] @ (xs[w] + h * (2 * vs[w] - v0s[w] - h * Minv[w] @ Fa[w]))
        F = np.linalg.solve(np.eye(k) - h * h * K[w] @ Minv[w], rhs)
        if debt:
            F = F - arm._robin_debt[w]
        for p in np.flatnonzero(world == w):
            out[glob[p]] = F[6 * slot[p]:6 * slot[p] + 6]
    out_body_f.assign(out)
    STATS["robin"] = STATS.get("robin", 0) + 1


def install():
    from newton.solvers.experimental.coupled import SolverCoupledProxy

    if os.environ.get("DM_PASSES"):  # every coupler's passes (examples included)
        init = SolverCoupledProxy.Config.__init__

        def _init(self, *a, **kw):
            init(self, *a, **kw)
            object.__setattr__(self, "iterations", int(os.environ["DM_PASSES"]))

        SolverCoupledProxy.Config.__init__ = _init
    if not MODE and not ROBIN:
        return
    SolverCoupledProxy.graph_capturable = False  # host-side numpy every step

    from . import coupling
    from .solver import SolverDiSMech

    step_coupled = SolverCoupledProxy._step_coupled

    def _step_coupled(self, state_in, state_out, control, contacts, dt):
        if MODE:
            _probe(self, state_in, control, contacts, dt)
        step_coupled(self, state_in, state_out, control, contacts, dt)

    SolverCoupledProxy._step_coupled = _step_coupled

    rewind = SolverDiSMech.coupling_rewind_proxy_body

    def _rewind(self, local_to_global, state, coupling_forces, gravity, dt):
        rewind(self, local_to_global, state, coupling_forces, gravity, dt)
        arm = self._proxy_arm
        if arm is not None and arm.k and (getattr(arm, "_schur", None) is not None or ROBIN == "2"):
            body, world, slot = arm._body.numpy(), arm._world, arm._slot
            glob = local_to_global.numpy()[body]
            Fg = coupling_forces.numpy()
            F = np.zeros((arm._arm.M.shape[0], arm.k))
            for p in range(len(body)):
                F[world[p], 6 * slot[p]:6 * slot[p] + 6] = Fg[glob[p]]
            arm._schur_F = F

    SolverDiSMech.coupling_rewind_proxy_body = _rewind

    update = coupling.ProxyArm.update

    def _update(self, state, h):
        self._robin_pass = getattr(self, "_robin_pass", 0) + 1 if self.restart else 0
        out = update(self, state, h)
        if ROBIN == "2":  # the synced start pose and end velocity (the source's), and the view's inverse inertia
            b = self._body.numpy()
            self._sync = (state.body_q.numpy()[b].astype(np.float64), state.body_qd.numpy()[b].astype(np.float64), h,
                          self._arm.Minv.numpy().astype(np.float64))
        S = getattr(self, "_schur", None)
        if S is None or MODE == "probe":
            return out
        a = self._arm
        W, k = a.M.shape[0], self.k
        Minv_old = a.Minv.numpy().astype(np.float64)
        Minv_s = _select(S)
        Minv = Minv_s + EPS * Minv_old
        M = np.linalg.inv(Minv)
        pred = a.pred.numpy().astype(np.float64).reshape(W, k)
        F = getattr(self, "_schur_F", np.zeros((W, k)))
        rewind = Minv_s if os.environ.get("DM_SCHUR_REWIND", "s") == "s" else Minv  # the source's own response
        pred += h * h * np.einsum("wij,wj->wi", Minv_old - rewind, F)
        a.M.assign(M.astype(np.float32))
        a.Minv.assign(Minv.astype(np.float32))
        a.pred.assign(pred.reshape(-1).astype(np.float32))
        self._schur_step = (M, pred, h)
        return out

    coupling.ProxyArm.update = _update

    finish = coupling.ProxyArm.finish

    def _finish(self, z, state_in, state_out, dt, h):
        finish(self, z, state_in, state_out, dt, h)
        if ROBIN:  # the reference pose of the Robin force: the proxies' end poses
            b = self._body.numpy()
            self._com_local = self.model.body_com.numpy()[b].astype(np.float64)
            qo = state_out.body_q.numpy()[b].astype(np.float64)
            self._robin_pose = (qo[:, :3] + _qrot(qo[:, 3:], self._com_local), qo[:, 3:])
        if getattr(self, "_schur_step", None) is not None:
            a = self._arm
            W, k = a.M.shape[0], self.k
            self._schur_z = z.numpy()[a.start:a.start + W * k].astype(np.float64).reshape(W, k)

    coupling.ProxyArm.finish = _finish

    harvest = SolverDiSMech.coupling_harvest_proxy_wrenches

    def _harvest(self, local_to_global, out_body_f, **kw):
        arm = self._proxy_arm
        if arm is None or getattr(arm, "_schur_step", None) is None:
            harvest(self, local_to_global, out_body_f, **kw)
            if ROBIN == "2" and arm is not None and getattr(arm, "_robin_K", None) is not None:
                _robin_harvest(arm, local_to_global, out_body_f)
            return
        M, pred, h = arm._schur_step
        Fn = np.einsum("wij,wj->wi", M, arm._schur_z - pred) / (h * h)
        body, glob, _, world, slot = arm._schur_maps
        out = out_body_f.numpy()
        for p in range(len(body)):
            out[glob[p]] = Fn[world[p], 6 * slot[p]:6 * slot[p] + 6]
        out_body_f.assign(out)
        arm._schur_step = None

    SolverDiSMech.coupling_harvest_proxy_wrenches = _harvest

    solve = SolverDiSMech._solve

    def _solve(self, state_in, q, dt):
        solve(self, state_in, q, dt)
        arm = self._proxy_arm
        if not ROBIN or arm is None or not arm.k:
            return
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        if os.environ.get("DM_ROBIN_K", "end") == "end":
            self._assemble(q, self.dofs)  # the step's matrix at its end (else its last factorized)
        indptr, indices = self.A.pattern
        U = sp.csr_matrix((self.A.vals.numpy(), indices, indptr), shape=(self.A.n, self.A.n))
        A = (U + U.T - sp.diags(U.diagonal())).tocsc()
        a = arm._arm
        W, k = a.M.shape[0], arm.k
        z = np.arange(a.start, a.start + W * k)
        rest = np.setdiff1d(np.arange(self.A.n), z)
        Azn = A[z][:, rest].toarray()
        S = A[z][:, z].toarray() - Azn @ spla.splu(A[rest][:, rest].tocsc()).solve(Azn.T)
        M = a.M.numpy().astype(np.float64) * a.inv_h2
        K = np.stack([-(S[w * k:(w + 1) * k, w * k:(w + 1) * k] - M[w]) for w in range(W)])
        arm._robin_K = 0.5 * (K + K.transpose(0, 2, 1))

    SolverDiSMech._solve = _solve
