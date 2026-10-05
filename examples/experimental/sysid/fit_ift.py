"""IFT baseline: fit the bend stiffness to baseline.npz by gradients through the solver (the standard step adjoint).

Every trial is one rod of one batched model; a rollout replays all of them at once on a ``wp.Tape`` and one
``tape.backward`` gives d(mean loss)/d(log k) (one implicit-function adjoint solve per step for the whole batch).
The learned parameter is the weak-plane bend stiffness ``k``; the strong plane is ``BEND_STRONG_RATIO k`` (the
ratio, damping, stretch, twist, mass and gravity are known). Loss: mean squared error of the free nodes against the
noisy observations ``x_obs`` over all frames, in mm^2.

The fixed DOFs (clamp nodes 0, 1; gripper nodes N-2, N-1; the gripper edge's twist) are driven by one of

* ``observed``: each trial's gripper pose readout ``grip_obs`` (noisy), linearly interpolated from the 40 Hz frames to
  the 200 Hz steps and applied as one rigid pose of the gripped edge;
* ``command``: the nominal command (x compress, z shear, no rotation), the same for every trial.
* ``true``: the oracle, each trial's hidden actual gripper pose (what the noisy readout measures).

The clamp is a known fixture held at rest. After the fits, every learned k is replayed with the hidden true gripper
pose to compare trajectories and branches with the data.

Each rollout starts from the straight rest pose at rest (the learner knows neither the per-trial initial state nor
the sag; the initial state is forgotten before the fork anyway).

    uv run scratch/sysid/fit_ift.py time          # one forward + backward, timings and an FD check of the gradient
    uv run scratch/sysid/fit_ift.py fit           # Adam and L-BFGS for every drive, then replay -> fit_ift.npz, fit_ift.png
    uv run scratch/sysid/fit_ift.py plot          # re-plot fit_ift.npz
"""

import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).parent))
import gen_baseline as gb  # noqa: E402

from dismech_newton import flatten_state, suspended_tape  # noqa: E402
from dismech_newton.solver import advance_frames_kernel  # noqa: E402
from dismech_newton.strains import vec10f  # noqa: E402
from dismech_newton.triplet import advance_ref_twist_kernel  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "fit_ift.npz"
K_TRUE = gb.BEND
K_GUESS = 2.5
LOG_K_BOUNDS = (np.log(0.5), np.log(300.0))
ADAM_LR = 0.15  # Adam on log k
VARIANTS = ("observed", "command", "true")
OPTIMIZERS = ("adam", "lbfgs")
MAX_EVALS = 40  # gradient evaluations per fit


@wp.kernel
def set_bend(log_k: wp.array[float], ratio: float, base: wp.array[vec10f], params: wp.array[vec10f]):
    t = wp.tid()
    p = base[t]
    k = wp.exp(log_k[0])
    params[t] = vec10f(p[0], p[1], k, ratio * k, p[4], p[5], p[6], p[7], p[8], p[9])


@wp.kernel
def set_fixed(idx: wp.array[wp.int32], vals: wp.array2d[float], step: int, q: wp.array[float]):
    j = wp.tid()
    q[idx[j]] = vals[step, j]


@wp.kernel
def frame_loss(x: wp.array[wp.vec3], obs: wp.array[wp.vec3], free: wp.array[wp.int32], nodes: int, scale: float,
               loss: wp.array[float]):
    i = wp.tid()
    if free[i] != 0:
        wp.atomic_add(loss, i // nodes, scale * wp.length_sq(x[i] - obs[i]))


def load():
    d = np.load(gb.OUT)
    return {k: d[k] for k in d.files}


def pose_vals(x_rest, t_frames, poses, t_step):
    """Fixed DOF values (len(t_step), rods, 14) from per-trial gripper poses (rods, frames, 6) at ``t_frames``: the
    clamp held at rest, the gripped edge moved rigidly (translation, rotation vector about its centre), the gripper
    edge's twist = roll."""
    from scipy.spatial.transform import Rotation

    N = gb.NODES
    B = len(poses)
    P = np.stack([np.stack([np.interp(t_step, t_frames, poses[r, :, j]) for j in range(6)], -1) for r in range(B)])
    a, b = x_rest[N - 2], x_rest[N - 1]
    c = 0.5 * (a + b)
    R = Rotation.from_rotvec(P[..., 3:].reshape(-1, 3))
    vals = np.zeros((len(t_step), B, 14))
    vals[:, :, 0:3], vals[:, :, 3:6] = x_rest[0], x_rest[1]
    vals[:, :, 6:9] = (R.apply(a - c).reshape(B, -1, 3) + c + P[..., :3]).transpose(1, 0, 2)
    vals[:, :, 9:12] = (R.apply(b - c).reshape(B, -1, 3) + c + P[..., :3]).transpose(1, 0, 2)
    vals[:, :, 13] = P[..., 3].T
    return vals


def fixed_schedule(d, variant, rods):
    """Fixed DOF indices (rods * 14,) and their values before every step (STEPS + 1, rods * 14)."""
    N, S, B = gb.NODES, gb.SEGMENTS, rods
    nodes = [0, 1, N - 2, N - 1]
    twist0 = 3 * N * B
    idx = np.array([[3 * (r * N + n) + c for n in nodes for c in range(3)] + [twist0 + r * S, twist0 + r * S + S - 1]
                    for r in range(B)])
    t_step = np.arange(gb.STEPS + 1) * gb.DT
    x_rest = d["x_rest"]
    if variant == "observed":  # the gripper pose readout (noisy), as one rigid pose of the gripped edge
        vals = pose_vals(x_rest, d["t"], d["grip_obs"][:B].astype(np.float64), t_step)
    elif variant == "true":  # the oracle: the hidden actual gripper pose
        vals = pose_vals(x_rest, d["t"], d["grip"][:B], t_step)
    else:  # the nominal command, the same for every trial
        vals = pose_vals(x_rest, t_step, np.repeat(gb.command()[None], B, 0), t_step)
    return idx.reshape(-1), vals.reshape(gb.STEPS + 1, -1)


class Fit:
    def __init__(self, rods, variant, d=None):
        d = load() if d is None else d
        self.rods, self.variant = rods, variant
        self.model, self.solver = gb.build(rods, gb.DENSITY_SCALE)
        m, dev = self.model, self.model.device
        params = m.dismech.triplet_params
        self.base = wp.clone(params)
        params.requires_grad = True
        self.params = params
        self.log_k = wp.array([np.log(K_GUESS)], dtype=float, requires_grad=True, device=dev)
        self.states = [m.state(requires_grad=True) for _ in range(gb.STEPS + 1)]
        for s in self.states:
            flatten_state(s)
        self.rest = m.state()
        flatten_state(self.rest)
        self.solver.triplets.measure(self.rest, self.rest.dismech.triplet_strain_q)
        idx, vals = fixed_schedule(d, variant, rods)
        self.fix_idx = wp.array(idx.astype(np.int32), dtype=wp.int32, device=dev)
        self.fix_vals = wp.array(vals.astype(np.float32), dtype=float, device=dev)
        free = np.ones((rods, gb.NODES), np.int32)
        free[:, [0, 1, -2, -1]] = 0
        self.free = wp.array(free.reshape(-1), dtype=wp.int32, device=dev)
        self.obs = [wp.array(d["x_obs"][:rods, k].reshape(-1, 3), dtype=wp.vec3, device=dev)
                    for k in range(1, gb.N_OBS + 1)]
        self.scale = 1.0e6 / (gb.N_OBS * (gb.NODES - 4))  # per-trial mean squared error over frames and nodes, mm^2
        self.loss = wp.zeros(rods, dtype=float, requires_grad=True, device=dev)
        self.up_obs = d["up_buckle"][:rods]

    def _start(self):
        """states[0] = the rest pose with the step-0 fixed DOFs, frames transported, strains measured."""
        st, src = self.states[0], self.rest
        for k in ("q", "qd", "edge_d1_q", "triplet_ref_twist_q", "triplet_strain_q"):
            getattr(st.dismech, k).assign(getattr(src.dismech, k))
        st.dismech.qd.zero_()
        dev = self.model.device
        wp.launch(set_fixed, dim=self.fix_idx.shape[0], inputs=[self.fix_idx, self.fix_vals, 0],
                  outputs=[st.dismech.q], device=dev)
        d, tr = self.solver.der, self.solver.triplets
        wp.launch(advance_frames_kernel, dim=d.edge_length.shape[0],
                  inputs=[src.particle_q, st.particle_q, d.edge_node0, d.edge_node1, src.dismech.edge_d1_q],
                  outputs=[st.dismech.edge_d1_q], device=dev)
        wp.launch(advance_ref_twist_kernel, dim=tr.count,
                  inputs=[st.dismech.q, st.dismech.edge_d1_q, tr.conn, src.dismech.triplet_ref_twist_q],
                  outputs=[st.dismech.triplet_ref_twist_q], device=dev)
        tr.measure(st, st.dismech.triplet_strain_q)

    def run(self):
        dev = self.model.device
        with suspended_tape():
            self.loss.zero_()
            self._start()
        wp.launch(set_bend, dim=self.params.shape[0], inputs=[self.log_k, gb.BEND_STRONG_RATIO, self.base],
                  outputs=[self.params], device=dev)
        for i in range(gb.STEPS):
            if i:
                with suspended_tape():  # the drive is data: no adjoint through it
                    wp.launch(set_fixed, dim=self.fix_idx.shape[0], inputs=[self.fix_idx, self.fix_vals, i],
                              outputs=[self.states[i].dismech.q], device=dev)
            self.solver.step(self.states[i], self.states[i + 1], None, None, gb.DT)
            if (i + 1) % gb.OBS_EVERY == 0:
                k = (i + 1) // gb.OBS_EVERY
                with suspended_tape():  # the recorded fixed DOFs are the pose at that instant
                    wp.launch(set_fixed, dim=self.fix_idx.shape[0], inputs=[self.fix_idx, self.fix_vals, i + 1],
                              outputs=[self.states[i + 1].dismech.q], device=dev)
                wp.launch(frame_loss, dim=self.model.particle_count,
                          inputs=[self.states[i + 1].particle_q, self.obs[k - 1], self.free, gb.NODES, self.scale],
                          outputs=[self.loss], device=dev)

    def loss_and_grad(self, log_k):
        self.log_k.assign(np.array([log_k], dtype=np.float32))
        tape = wp.Tape()
        with tape:
            self.run()
        tape.backward(grads={self.loss: wp.full(self.rods, 1.0 / self.rods, dtype=float, device=self.model.device)})
        g = float(self.log_k.grad.numpy()[0])
        per_trial = self.loss.numpy().astype(np.float64)
        tape.zero()
        tape.reset()
        return per_trial, g

    def forward(self, log_k):
        self.log_k.assign(np.array([log_k], dtype=np.float32))
        self.run()
        return self.loss.numpy().astype(np.float64)

    def trajectories(self):
        """Simulated node positions at the observed frames (rods, frames, NODES, 3) of the last rollout."""
        return np.stack([self.states[k * gb.OBS_EVERY].particle_q.numpy().reshape(self.rods, gb.NODES, 3)
                         for k in range(gb.N_OBS + 1)], 1)


def time_it(rods=256):
    for variant in VARIANTS:
        f = Fit(rods, variant)
        f.forward(np.log(K_GUESS))  # warm-up (kernel compile, analysis)
        wp.synchronize()
        tic = time.perf_counter()
        L = f.forward(np.log(K_GUESS))
        wp.synchronize()
        t_fwd = time.perf_counter() - tic
        tic = time.perf_counter()
        L2, g = f.loss_and_grad(np.log(K_GUESS))
        wp.synchronize()
        t_grad = time.perf_counter() - tic
        h = 1e-2
        lp, lm = f.forward(np.log(K_GUESS) + h).mean(), f.forward(np.log(K_GUESS) - h).mean()
        fd = (lp - lm) / (2 * h)
        print(f"{variant}: loss {L.mean():.4f} mm^2 (taped {L2.mean():.4f}); forward {t_fwd:.1f} s, "
              f"forward+backward {t_grad:.1f} s; dL/dlogk IFT {g:+.5e}, FD {fd:+.5e}, rel {abs(g - fd) / abs(fd):.2e}",
              flush=True)
        print(f"  peak GPU: see nvidia-smi", flush=True)


def adam(fun, x, iters=MAX_EVALS, lr=ADAM_LR, b1=0.9, b2=0.999):
    """Adam on the scalar log k, clipped to LOG_K_BOUNDS; a non-finite evaluation halves the step back.
    Returns the iterates (every one evaluated, in order)."""
    m = v = 0.0
    xs, prev = [], None
    for i in range(1, iters + 1):
        f, g = fun(np.array([x]))
        if not np.isfinite(f + g[0]):
            x = 0.5 * (x + prev)  # back off towards the last good iterate
            xs.append(x)
            continue
        prev = x
        m, v = b1 * m + (1 - b1) * g[0], b2 * v + (1 - b2) * g[0] ** 2
        xs.append(x)
        x = float(np.clip(x - lr * (m / (1 - b1**i)) / (np.sqrt(v / (1 - b2**i)) + 1e-12), *LOG_K_BOUNDS))
    return xs


def lbfgs(fun, x, evals=MAX_EVALS):
    """scipy L-BFGS-B on the scalar log k within LOG_K_BOUNDS; returns the final iterate and the stop message."""
    from scipy.optimize import minimize

    r = minimize(fun, np.array([x]), jac=True, method="L-BFGS-B", bounds=[LOG_K_BOUNDS],
                 options=dict(maxfun=evals, maxiter=evals))
    return float(r.x[0]), str(r.message)


def fit(rods=256):
    d = load()
    res = {}
    for variant in VARIANTS:
        f = Fit(rods, variant, d)
        per_true = f.forward(np.log(K_TRUE))  # the truth's own loss under this drive
        for opt in OPTIMIZERS:
            run = f"{variant}_{opt}"
            hist = []  # every evaluation: (log k, mean loss, grad)

            def fun(x):
                per, g = f.loss_and_grad(float(x[0]))
                hist.append((float(x[0]), per.mean(), g))
                print(f"  {run} eval {len(hist)}: k {np.exp(x[0]):.4f} loss {per.mean():.5f} grad {g:+.3e}",
                      flush=True)
                return per.mean(), np.array([g])

            tic = time.perf_counter()
            if opt == "adam":
                xs = adam(fun, np.log(K_GUESS))
                x_fit, msg = [h[0] for h in hist if np.isfinite(h[1])][-1], "max evaluations"
            else:
                x_fit, msg = lbfgs(fun, np.log(K_GUESS))
            k_fit = np.exp(x_fit)
            print(f"{run}: k_fit {k_fit:.4f} (true {K_TRUE}), {len(hist)} evals in {time.perf_counter() - tic:.0f} s"
                  f" ({msg})", flush=True)
            res[run] = dict(hist=np.array(hist), k_fit=k_fit, per_true=per_true)
        del f

    f = Fit(rods, "true", d)
    up_obs = d["up_buckle"][:rods]
    for name, k in [*((r, res[r]["k_fit"]) for r in list(res)), ("truth", K_TRUE)]:
        f.forward(np.log(k))
        mid = gb.mid_rel(f.trajectories())
        match = np.mean((mid[:, gb.K_BUCKLE] > 0) == up_obs)
        res.setdefault(name, {}).update(replay_mid=mid, replay_match=match)
        print(f"replay {name} (k {k:.3g}) with the true gripper pose: branch match {match:.3f}", flush=True)
    np.savez_compressed(OUT, k_true=K_TRUE, k_guess=K_GUESS, up_obs=up_obs, mid_obs=gb.mid_rel(d["x_obs"][:rods]),
                        **{f"{r}_{k}": a for r, v in res.items() for k, a in v.items()})
    plot()


def plot():
    import matplotlib.pyplot as plt

    d = np.load(OUT)
    t = np.arange(gb.N_OBS + 1) * gb.OBS_EVERY * gb.DT
    colors = {"observed": "#2a6fdb", "command": "#d1453b", "true": "#2a9d55"}
    names = {"observed": "readout", "command": "command", "true": "oracle"}
    styles = {"adam": "-", "lbfgs": "--"}
    opt_names = {"adam": "Adam", "lbfgs": "L-BFGS"}
    runs = [(v, "adam") for v in VARIANTS]  # the plot shows Adam only (L-BFGS is in fit_ift.npz)
    kt = float(d["k_true"])

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    for v, o in runs:
        h = d[f"{v}_{o}_hist"]
        kw = dict(color=colors[v], ls=styles[o], marker="o", ms=2.5, lw=1.2)
        ax[0].semilogy(np.arange(len(h)), h[:, 1], **kw)
        ax[1].semilogy(np.arange(len(h)), np.exp(h[:, 0]), **kw)
    for v in VARIANTS:
        ax[0].axhline(d[f"{v}_adam_per_true"].mean(), color=colors[v], ls=":", lw=1)
    ax[1].axhline(kt, color="k", ls=":", lw=1.2)
    ax[0].set(xlabel="evaluation", ylabel="loss [mm²]")
    ax[1].set(xlabel="evaluation", ylabel="bend stiffness k")
    handles = [plt.Line2D([], [], color=colors[v], lw=2, label=names[v]) for v in VARIANTS]
    ax[0].legend(handles=handles + [plt.Line2D([], [], color="0.3", ls=":", label="at true k")], fontsize=8)
    ax[1].legend(handles=handles + [plt.Line2D([], [], color="k", ls=":", label=f"true k = {kt:g}")], fontsize=8)

    a = ax[2]
    mid_obs = d["mid_obs"] * 1e3
    n = min(64, len(mid_obs))
    for b in range(n):
        a.plot(t, mid_obs[b], color="0.8", lw=0.5)
    for v, o in runs:
        for b in range(n):
            a.plot(t, d[f"{v}_{o}_replay_mid"][b] * 1e3, color=colors[v], ls=styles[o], lw=0.7, alpha=0.5)
    for b in range(n):
        a.plot(t, d["truth_replay_mid"][b] * 1e3, color="k", lw=0.5, alpha=0.4, ls=":")
    leg = [plt.Line2D([], [], color="0.8", label="observed")]
    leg += [plt.Line2D([], [], color=colors[v], ls=styles[o],
                       label=f"{names[v]}: k = {float(d[f'{v}_{o}_k_fit']):.3g}, "
                             f"{float(d[f'{v}_{o}_replay_match']):.0%}") for v, o in runs]
    leg.append(plt.Line2D([], [], color="k", ls=":", label=f"truth: k = {kt:g}, {float(d['truth_replay_match']):.0%}"))
    a.legend(handles=leg, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    a.set(xlabel="t [s]", ylabel="mid-span z [mm]")
    a.text(0.02, 0.98, "replay with true gripper motion\n(% = right branch)", transform=a.transAxes, va="top",
           fontsize=7, color="0.3")
    fig.tight_layout()
    fig.savefig(HERE / "fit_ift.png", dpi=110)
    print("plot: fit_ift.png")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    rods = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    {"time": time_it, "fit": fit, "plot": plot}[mode](*(() if mode == "plot" else (rods,)))
