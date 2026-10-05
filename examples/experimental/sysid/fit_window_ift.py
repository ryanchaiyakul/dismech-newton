"""Windowed (multiple-shooting) IFT: fit the bend stiffness to baseline.npz from short rollouts that restart from the data.

Same parameter (log k), data, drives and loss as `fit_ift.py`, but the 4 s rollout is cut into windows of ``W`` observed
frames (``W * OBS_EVERY`` steps). Every window restarts from the data, so it stays on the branch the data took and each
gradient is an ordinary within-branch IFT gradient: no branch labels, no masks, no sampling.

Window starts (teacher forcing): the window starting at frame ``k0 > 0`` starts from the free node positions observed at
``k0`` (``--start obs``; ``--start clean`` uses the hidden clean positions, a diagnostic upper bound only), the fixed DOFs
from the drive at that step, the free twists of the straight rest pose, and either at rest (``--vel rest``) or with the
central finite difference of the observed positions (``--vel fd``). Frames, reference twist and strains are made
consistent with `gen_baseline.restart` from the straight rest state (as `fit_ift.Fit._start`). The noisy positions are
taken as they are (no relaxation or projection): the first step snaps the stretched edges back. The window at ``k0 = 0``
starts like `fit_ift` (the rest pose, at rest), so ``W = N_OBS`` is `fit_ift` exactly.

Loss: per trial, the squared error of the free nodes against ``x_obs`` summed over every window's frames except its start
frame, scaled as `fit_ift` (mean over 160 frames x free nodes, mm^2); every frame 1..N_OBS is counted once.

All windows of all trials are rods of one batched model, run ``GROUP`` windows per trial at a time (model build costs
~8 ms per rod), the gradient summed over the groups.

    uv run examples/experimental/sysid/fit_window_ift.py time   [--W 4]   # forward + backward timings, FD check
    uv run examples/experimental/sysid/fit_window_ift.py drives [--Ws ..] # loss landscape over K_GRID, window branch match
    uv run examples/experimental/sysid/fit_window_ift.py fit    [--W 4]   # Adam + L-BFGS per drive, replay -> .npz, .png
    uv run examples/experimental/sysid/fit_window_ift.py sweep  [--Ws ..] # Adam fit per drive vs W -> fit_window_sweep.npz
    uv run examples/experimental/sysid/fit_window_ift.py plot             # re-plot fit_window_ift.npz
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).parent))
import fit_ift as fi  # noqa: E402
import gen_baseline as gb  # noqa: E402

from dismech_newton import flatten_state, suspended_tape  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "fit_window_ift.npz"
OUT_SWEEP = HERE / "fit_window_sweep.npz"
W_DEFAULT = 4
W_SWEEP = (2, 4, 8, 16, 40, 160)
GROUP = 10  # at most this many windows per trial in one batch
LAST = 20  # iterates summarised as the fit's tail


def fixed_idx(rods):
    """Fixed DOF indices (rods * 14,): nodes 0, 1, N-2, N-1 and the first and last edge's twist (as `fixed_schedule`)."""
    N, S = gb.NODES, gb.SEGMENTS
    nodes = [0, 1, N - 2, N - 1]
    twist0 = 3 * N * rods
    return np.array([[3 * (r * N + n) + c for n in nodes for c in range(3)] + [twist0 + r * S, twist0 + r * S + S - 1]
                     for r in range(rods)]).reshape(-1)


_MODELS = {}


def model(rods):
    """One model per batch size (building is slow)."""
    if rods not in _MODELS:
        _MODELS[rods] = gb.build(rods, gb.DENSITY_SCALE)
    return _MODELS[rods]


class WinFit:
    def __init__(self, trials, variant, W, start="obs", vel="rest", d=None):
        d = fi.load() if d is None else d
        assert gb.N_OBS % W == 0, f"W must divide {gb.N_OBS}"
        self.trials, self.variant, self.W = trials, variant, W
        self.nw = gb.N_OBS // W  # windows per trial
        self.G = max(g for g in range(1, min(GROUP, self.nw) + 1) if self.nw % g == 0)  # windows per batch
        self.chunks = self.nw // self.G
        self.rods = R = trials * self.G  # rod r = g * trials + b: window g of the batch, trial b
        self.L = L = W * gb.OBS_EVERY  # steps per window
        self.model, self.solver = model(R)
        m, dev = self.model, self.model.device
        N, S = gb.NODES, gb.SEGMENTS

        params = m.dismech.triplet_params
        self.base = wp.clone(params)
        params.requires_grad = True
        self.params = params
        self.log_k = wp.array([np.log(fi.K_GUESS)], dtype=float, requires_grad=True, device=dev)
        self.states = [m.state(requires_grad=True) for _ in range(L + 1)]
        for s in self.states:
            flatten_state(s)
        self.rest = m.state()
        flatten_state(self.rest)
        self.solver.triplets.measure(self.rest, self.rest.dismech.triplet_strain_q)
        q_rest = self.rest.dismech.q.numpy()

        self.fix_idx = wp.array(fixed_idx(R).astype(np.int32), dtype=wp.int32, device=dev)
        _, vals = fi.fixed_schedule(d, variant, trials)
        vals = vals.reshape(gb.STEPS + 1, trials, 14)
        free = np.ones((R, N), np.int32)
        free[:, [0, 1, -2, -1]] = 0
        self.free = wp.array(free.reshape(-1), dtype=wp.int32, device=dev)
        self.scale = 1.0e6 / (gb.N_OBS * (N - 4))
        self.loss = wp.zeros(R, dtype=float, requires_grad=True, device=dev)

        x_src = d["x_obs"] if start == "obs" else d["x"]
        x_obs = d["x_obs"][:trials].astype(np.float64)
        dt_obs = gb.OBS_EVERY * gb.DT
        self.k0 = np.arange(self.nw) * W  # start frame of every window
        self.fix_vals, self.obs, self.q0, self.qd0 = [], [], [], []
        for c in range(self.chunks):
            ws = np.arange(c * self.G, (c + 1) * self.G)
            k0 = self.k0[ws]
            steps = k0[:, None] * gb.OBS_EVERY + np.arange(L + 1)[None]  # (G, L + 1) global steps
            v = vals[steps.T]  # (L + 1, G, trials, 14)
            self.fix_vals.append(wp.array(v.reshape(L + 1, -1).astype(np.float32), dtype=float, device=dev))
            o = x_obs[:, k0[:, None] + np.arange(1, W + 1)[None]]  # (trials, G, W, N, 3)
            self.obs.append(wp.array(o.transpose(2, 1, 0, 3, 4).reshape(W, R * N, 3).astype(np.float32), dtype=wp.vec3,
                                     device=dev))
            # start positions: data at k0 (window 0: the rest pose); fixed DOFs and twists are set in `_start`
            x0 = np.where((k0 > 0)[:, None, None, None], x_src[:trials, k0].transpose(1, 0, 2, 3),
                          d["x_rest"][None, None])  # (G, trials, N, 3)
            q = q_rest.copy()
            q[: 3 * N * R] = x0.reshape(-1)
            self.q0.append(wp.array(q.astype(np.float32), dtype=float, device=dev))
            qd = np.zeros_like(q)
            if vel == "fd":  # central difference of the observed positions, free nodes only
                kp, km = np.minimum(k0 + 1, gb.N_OBS), np.maximum(k0 - 1, 0)
                xd = (x_obs[:, kp] - x_obs[:, km]) / ((kp - km) * dt_obs)[None, :, None, None]  # (trials, G, N, 3)
                xd = xd.transpose(1, 0, 2, 3) * (free.reshape(self.G, trials, N, 1) * (k0 > 0)[:, None, None, None])
                qd[: 3 * N * R] = xd.reshape(-1)
            self.qd0.append(wp.array(qd.astype(np.float32), dtype=float, device=dev) if vel == "fd" else None)
        self.up_obs = d["up_buckle"][:trials]
        self.mid_obs = gb.mid_rel(x_obs)  # (trials, frames)

    def _start(self, c):
        st, dev = self.states[0], self.model.device
        gb.copy_state(st, self.rest)
        st.dismech.q.assign(self.q0[c])
        wp.launch(fi.set_fixed, dim=self.fix_idx.shape[0], inputs=[self.fix_idx, self.fix_vals[c], 0],
                  outputs=[st.dismech.q], device=dev)
        gb.restart(self.solver, st, self.rest)
        if self.qd0[c] is not None:
            st.dismech.qd.assign(self.qd0[c])

    def run(self, c):
        dev, fv = self.model.device, self.fix_vals[c]
        with suspended_tape():
            self.loss.zero_()
            self._start(c)
        wp.launch(fi.set_bend, dim=self.params.shape[0], inputs=[self.log_k, gb.BEND_STRONG_RATIO, self.base],
                  outputs=[self.params], device=dev)
        for i in range(self.L):
            if i:
                with suspended_tape():
                    wp.launch(fi.set_fixed, dim=self.fix_idx.shape[0], inputs=[self.fix_idx, fv, i],
                              outputs=[self.states[i].dismech.q], device=dev)
            self.solver.step(self.states[i], self.states[i + 1], None, None, gb.DT)
            if (i + 1) % gb.OBS_EVERY == 0:
                j = (i + 1) // gb.OBS_EVERY
                with suspended_tape():
                    wp.launch(fi.set_fixed, dim=self.fix_idx.shape[0], inputs=[self.fix_idx, fv, i + 1],
                              outputs=[self.states[i + 1].dismech.q], device=dev)
                wp.launch(fi.frame_loss, dim=self.model.particle_count,
                          inputs=[self.states[i + 1].particle_q, self.obs[c][j - 1], self.free, gb.NODES, self.scale],
                          outputs=[self.loss], device=dev)

    def _per_trial(self):
        return self.loss.numpy().astype(np.float64).reshape(self.G, self.trials).sum(0)

    def _end_mid(self, c):
        X = self.states[self.L].particle_q.numpy().reshape(self.G, self.trials, gb.NODES, 3)
        return gb.mid_rel(X).T  # (trials, G)

    def forward(self, log_k, ends=False):
        """Per-trial losses; with ``ends`` also mid_rel at every window end (trials, windows)."""
        self.log_k.assign(np.array([log_k], dtype=np.float32))
        L, mid = np.zeros(self.trials), []
        for c in range(self.chunks):
            self.run(c)
            L += self._per_trial()
            if ends:
                mid.append(self._end_mid(c))
        return (L, np.concatenate(mid, 1)) if ends else L

    def loss_and_grad(self, log_k, w=None):
        w = np.full(self.trials, 1.0 / self.trials) if w is None else w
        wr = wp.array(np.tile(np.asarray(w, np.float32), self.G), dtype=float, device=self.model.device)
        self.log_k.assign(np.array([log_k], dtype=np.float32))
        L, g = np.zeros(self.trials), 0.0
        for c in range(self.chunks):
            tape = wp.Tape()
            with tape:
                self.run(c)
            tape.backward(grads={self.loss: wr})
            g += float(self.log_k.grad.numpy()[0])
            L += self._per_trial()
            tape.zero()
            tape.reset()
        return L, g

    def window_match(self, mid_end, thresh=5e-3):
        """Fraction of windows that end on the observed side (sign of mid_rel), over all windows and over the windows
        whose observed end |mid_rel| > ``thresh`` (unambiguous)."""
        ke = self.k0 + self.W
        obs = self.mid_obs[:, ke]
        same = np.sign(mid_end) == np.sign(obs)
        clear = np.abs(obs) > thresh
        return same.mean(), same[clear].mean() if clear.any() else np.nan


def time_it(a):
    d = fi.load()
    for variant in fi.VARIANTS:
        f = WinFit(a.rods, variant, a.W, a.start, a.vel, d)
        lk = np.log(fi.K_GUESS)
        f.forward(lk)
        wp.synchronize()
        tic = time.perf_counter()
        L = f.forward(lk)
        wp.synchronize()
        t_fwd = time.perf_counter() - tic
        tic = time.perf_counter()
        L2, g = f.loss_and_grad(lk)
        wp.synchronize()
        t_grad = time.perf_counter() - tic
        h = 1e-2
        fd = (f.forward(lk + h).mean() - f.forward(lk - h).mean()) / (2 * h)
        print(f"W {a.W} {variant}: loss {L.mean():.4f} mm^2 (taped {L2.mean():.4f}); forward {t_fwd:.1f} s, "
              f"forward+backward {t_grad:.1f} s ({f.chunks} batches of {f.rods} rods x {f.L} steps); "
              f"dL/dlogk IFT {g:+.5e}, FD {fd:+.5e}, rel {abs(g - fd) / abs(fd):.2e}", flush=True)


def drives(a):
    """Loss over K_GRID and the window-end branch match, per drive and W."""
    d = fi.load()
    for W in a.Ws:
        for variant in fi.VARIANTS:
            f = WinFit(a.rods, variant, W, a.start, a.vel, d)
            rows = []
            for k in fi.K_GRID:
                L, mid = f.forward(np.log(k), ends=True)
                m_all, m_clear = f.window_match(mid)
                rows.append((k, L.mean()))
                print(f"  W {W:3d} {variant:8s} k {k:7.2f}  loss {L.mean():9.4f}  non-finite trials "
                      f"{np.sum(~np.isfinite(L)):3d}  window branch match {m_all:.3f} "
                      f"(|obs mid| > 5 mm: {m_clear:.3f})", flush=True)
            r = np.array(rows)
            jumps = np.diff(r[:, 1])
            print(f"W {W} {variant}: argmin over grid k {r[np.nanargmin(r[:, 1]), 0]:.2f} (true {fi.K_TRUE}); "
                  f"loss sign changes of slope {np.sum(np.diff(np.sign(jumps[np.isfinite(jumps)])) != 0)}", flush=True)


def fit_one(f, opt, run, evals=fi.MAX_EVALS):
    hist = []

    def fun(x):
        per, g = f.loss_and_grad(float(x[0]))
        hist.append((float(x[0]), per.mean(), g))
        print(f"  {run} eval {len(hist)}: k {np.exp(x[0]):.4f} loss {per.mean():.5f} grad {g:+.3e}", flush=True)
        return per.mean(), np.array([g])

    tic = time.perf_counter()
    if opt == "adam":
        fi.adam(fun, np.log(fi.K_GUESS), iters=evals)
        x_fit, msg = [h[0] for h in hist if np.isfinite(h[1])][-1], "max evaluations"
    else:
        x_fit, msg = fi.lbfgs(fun, np.log(fi.K_GUESS), evals)
    hist = np.array(hist)
    tail = np.exp(hist[-LAST:, 0])
    print(f"{run}: k_fit {np.exp(x_fit):.4f} (true {fi.K_TRUE}); last {len(tail)} iterates mean {tail.mean():.3f} "
          f"range {tail.min():.3f}-{tail.max():.3f}; {len(hist)} evals in {time.perf_counter() - tic:.0f} s ({msg})",
          flush=True)
    return dict(hist=hist, k_fit=np.exp(x_fit))


def replay(res, rods, d):
    """Every fitted k as one full rollout from rest with the oracle drive; the branch match at the end of compression."""
    f = fi.Fit(rods, "true", d)
    up_obs = d["up_buckle"][:rods]
    for name, k in [*((r, res[r]["k_fit"]) for r in list(res)), ("truth", fi.K_TRUE)]:
        f.forward(np.log(k))
        mid = gb.mid_rel(f.trajectories())
        match = np.mean((mid[:, gb.K_BUCKLE] > 0) == up_obs)
        res.setdefault(name, {}).update(replay_mid=mid, replay_match=match)
        print(f"replay {name} (k {k:.3g}) with the true gripper pose: branch match {match:.3f}", flush=True)


def fit(a):
    d = fi.load()
    res = {}
    for variant in fi.VARIANTS:
        f = WinFit(a.rods, variant, a.W, a.start, a.vel, d)
        per_true = f.forward(np.log(fi.K_TRUE))
        for opt in fi.OPTIMIZERS:
            run = f"{variant}_{opt}"
            res[run] = fit_one(f, opt, run)
            res[run]["per_true"] = per_true
    replay(res, a.rods, d)
    np.savez_compressed(OUT, k_true=fi.K_TRUE, k_guess=fi.K_GUESS, W=a.W, start=a.start, vel=a.vel,
                        up_obs=d["up_buckle"][:a.rods], mid_obs=gb.mid_rel(d["x_obs"][:a.rods]),
                        **{f"{r}_{k}": v for r, x in res.items() for k, v in x.items()})
    plot()


def sweep(a):
    """Adam fit per drive for every W: fitted k and the spread of the last LAST iterates."""
    d = fi.load()
    out = {}
    for W in a.Ws:
        for variant in fi.VARIANTS:
            f = WinFit(a.rods, variant, W, a.start, a.vel, d)
            r = fit_one(f, "adam", f"W{W}_{variant}")
            out[f"W{W}_{variant}_hist"] = r["hist"]
            del f
    np.savez_compressed(OUT_SWEEP, Ws=np.array(a.Ws), start=a.start, vel=a.vel, **out)
    print("\nW    drive     k_fit   tail mean  tail range")
    for W in a.Ws:
        for variant in fi.VARIANTS:
            h = out[f"W{W}_{variant}_hist"]
            t = np.exp(h[-LAST:, 0])
            print(f"{W:<4d} {variant:8s} {np.exp(h[-1, 0]):8.3f} {t.mean():8.3f}   {t.min():.3f}-{t.max():.3f}")


def plot():
    import matplotlib.pyplot as plt

    d = np.load(OUT)
    ift = np.load(fi.OUT) if fi.OUT.exists() else None
    t = np.arange(gb.N_OBS + 1) * gb.OBS_EVERY * gb.DT
    colors = {"observed": "#2a6fdb", "command": "#d1453b", "true": "#2a9d55"}
    names = {"observed": "readout", "command": "command", "true": "oracle"}
    runs = [(v, "adam") for v in fi.VARIANTS]
    kt, W = float(d["k_true"]), int(d["W"])

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    for v, o in runs:
        h = d[f"{v}_{o}_hist"]
        kw = dict(color=colors[v], marker="o", ms=2.5, lw=1.2)
        ax[0].semilogy(np.arange(len(h)), h[:, 1], **kw)
        ax[1].semilogy(np.arange(len(h)), np.exp(h[:, 0]), **kw)
        if ift is not None:
            hi = ift[f"{v}_{o}_hist"]
            ax[1].semilogy(np.arange(len(hi)), np.exp(hi[:, 0]), color=colors[v], lw=0.8, alpha=0.35)
    for v in fi.VARIANTS:
        ax[0].axhline(d[f"{v}_adam_per_true"].mean(), color=colors[v], ls=":", lw=1)
    ax[1].axhline(kt, color="k", ls=":", lw=1.2)
    ax[0].set(xlabel="evaluation", ylabel="windowed loss [mm²]", title=f"windowed IFT, W = {W} frames "
              f"({W * gb.OBS_EVERY * gb.DT:.3g} s), start {d['start']}, velocity {d['vel']}")
    ax[1].set(xlabel="evaluation", ylabel="bend stiffness k", title="Adam iterates (faint: single-shooting fit_ift)")
    handles = [plt.Line2D([], [], color=colors[v], lw=2, label=names[v]) for v in fi.VARIANTS]
    ax[0].legend(handles=handles + [plt.Line2D([], [], color="0.3", ls=":", label="at true k")], fontsize=8)
    ax[1].legend(handles=handles + [plt.Line2D([], [], color="k", ls=":", label=f"true k = {kt:g}"),
                                    plt.Line2D([], [], color="0.5", lw=0.8, alpha=0.5, label="fit_ift (W = 160)")],
                 fontsize=8)

    a = ax[2]
    mid_obs = d["mid_obs"] * 1e3
    n = min(64, len(mid_obs))
    for b in range(n):
        a.plot(t, mid_obs[b], color="0.8", lw=0.5)
    for v, o in runs:
        for b in range(n):
            a.plot(t, d[f"{v}_{o}_replay_mid"][b] * 1e3, color=colors[v], lw=0.7, alpha=0.5)
    for b in range(n):
        a.plot(t, d["truth_replay_mid"][b] * 1e3, color="k", lw=0.5, alpha=0.4, ls=":")
    leg = [plt.Line2D([], [], color="0.8", label="observed")]
    leg += [plt.Line2D([], [], color=colors[v], label=f"{names[v]}: k = {float(d[f'{v}_{o}_k_fit']):.3g}, "
                                                      f"{float(d[f'{v}_{o}_replay_match']):.0%}") for v, o in runs]
    leg.append(plt.Line2D([], [], color="k", ls=":", label=f"truth: k = {kt:g}, {float(d['truth_replay_match']):.0%}"))
    a.legend(handles=leg, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    a.set(xlabel="t [s]", ylabel="mid-span z [mm]")
    a.text(0.02, 0.98, "full replay from rest with true gripper motion\n(% = right branch)", transform=a.transAxes,
           va="top", fontsize=7, color="0.3")
    fig.tight_layout()
    fig.savefig(HERE / "fit_window_ift.png", dpi=110)
    print("plot: fit_window_ift.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", default="fit", choices=("time", "drives", "fit", "sweep", "plot"))
    p.add_argument("--rods", type=int, default=256, help="trials")
    p.add_argument("--W", type=int, default=W_DEFAULT, help="frames per window")
    p.add_argument("--Ws", type=int, nargs="+", default=list(W_SWEEP))
    p.add_argument("--start", choices=("obs", "clean"), default="obs", help="window start positions")
    p.add_argument("--vel", choices=("rest", "fd"), default="rest", help="window start velocities")
    a = p.parse_args()
    if a.mode == "plot":
        plot()
    else:
        {"time": time_it, "drives": drives, "fit": fit, "sweep": sweep}[a.mode](a)
