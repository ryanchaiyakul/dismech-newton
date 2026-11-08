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
        proxy.coupling_forces.assign(F0)
        coupler._distribute_state(state_in, dt=dt, iteration_restart=True)
        arm.restart = restart
        arm._schur = 0.5 * (R + R.transpose(0, 2, 1)) / dt
        arm._schur_asym = np.abs(R - R.transpose(0, 2, 1)).max() / max(np.abs(R).max(), 1e-30)


def _select(S):
    if MODE == "full":
        return S
    out = np.zeros_like(S)
    for b in range(S.shape[1] // 6):
        sl = slice(6 * b, 6 * b + 6)
        out[:, sl, sl] = S[:, sl, sl] if MODE == "body" else np.einsum("wii->wi", S[:, sl, sl])[:, :, None] * np.eye(6)
    return out


def install():
    from newton.solvers.experimental.coupled import SolverCoupledProxy

    if os.environ.get("DM_PASSES"):  # every coupler's passes (examples included)
        init = SolverCoupledProxy.Config.__init__

        def _init(self, *a, **kw):
            init(self, *a, **kw)
            object.__setattr__(self, "iterations", int(os.environ["DM_PASSES"]))

        SolverCoupledProxy.Config.__init__ = _init
    if not MODE:
        return

    from . import coupling
    from .solver import SolverDiSMech

    step_coupled = SolverCoupledProxy._step_coupled

    def _step_coupled(self, state_in, state_out, control, contacts, dt):
        _probe(self, state_in, control, contacts, dt)
        step_coupled(self, state_in, state_out, control, contacts, dt)

    SolverCoupledProxy._step_coupled = _step_coupled

    rewind = SolverDiSMech.coupling_rewind_proxy_body

    def _rewind(self, local_to_global, state, coupling_forces, gravity, dt):
        rewind(self, local_to_global, state, coupling_forces, gravity, dt)
        arm = self._proxy_arm
        if arm is not None and getattr(arm, "_schur", None) is not None:
            body, glob, _, world, slot = arm._schur_maps
            Fg = coupling_forces.numpy()
            F = np.zeros((arm._arm.M.shape[0], arm.k))
            for p in range(len(body)):
                F[world[p], 6 * slot[p]:6 * slot[p] + 6] = Fg[glob[p]]
            arm._schur_F = F

    SolverDiSMech.coupling_rewind_proxy_body = _rewind

    update = coupling.ProxyArm.update

    def _update(self, state, h):
        out = update(self, state, h)
        S = getattr(self, "_schur", None)
        if S is None:
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
        if getattr(self, "_schur_step", None) is not None:
            a = self._arm
            W, k = a.M.shape[0], self.k
            self._schur_z = z.numpy()[a.start:a.start + W * k].astype(np.float64).reshape(W, k)

    coupling.ProxyArm.finish = _finish

    harvest = SolverDiSMech.coupling_harvest_proxy_wrenches

    def _harvest(self, local_to_global, out_body_f, **kw):
        arm = self._proxy_arm
        if arm is None or getattr(arm, "_schur_step", None) is None:
            return harvest(self, local_to_global, out_body_f, **kw)
        M, pred, h = arm._schur_step
        Fn = np.einsum("wij,wj->wi", M, arm._schur_z - pred) / (h * h)
        body, glob, _, world, slot = arm._schur_maps
        out = out_body_f.numpy()
        for p in range(len(body)):
            out[glob[p]] = Fn[world[p], 6 * slot[p]:6 * slot[p] + 6]
        out_body_f.assign(out)
        arm._schur_step = None

    SolverDiSMech.coupling_harvest_proxy_wrenches = _harvest
