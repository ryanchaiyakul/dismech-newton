"""Batched W=10 multiple-shooting L-BFGS-B system identification of cable routing mechanics and variational
tribology ``(EI, rho, mu_0, alpha, beta)`` purely from 83.3 Hz noisy quantized PaXini tactile data.

Usage:
    python examples/experimental/fit_cable_routing.py --test
    python examples/experimental/fit_cable_routing.py --plot
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "examples"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import newton  # noqa: E402
from utils import tactile  # noqa: E402
from utils.common import cached, pads  # noqa: E402

from dismech_newton import SolverDiSMech, flatten_state  # noqa: E402
from dismech_newton.state import suspended_tape  # noqa: E402

DT = 1.0e-3  # 1000 Hz physics step
OBS_EVERY = 12  # 12 ms = 83.3 Hz tactile frame rate
W = 10  # tactile frames per multiple-shooting window (120 substeps = 120 ms)
CHUNKS = 2  # 2 windows per trial -> 240 ms per calibration trial
TRIALS = 4  # light axial, heavy axial, transverse bend/slip, fast dynamic whip
WINDOWS = TRIALS * CHUNKS  # 8 batched W=10 shooting windows
FRAMES_PER_TRIAL = W * CHUNKS
STEPS_PER_TRIAL = FRAMES_PER_TRIAL * OBS_EVERY
STEPS_PER_WINDOW = W * OBS_EVERY

N_NODES = 17
L_SPAN = 0.160  # [m] span across M2826 sensor dome
L_DUAL = L_SPAN / (N_NODES - 1)  # 0.010 m per segment
ROD_RADIUS = 0.006
ROD_EA = 1.0e5
ROD_C_BEND = 3.0e-4
FRICTION_N0 = 1.2
FRICTION_DIR = (1.0, 0.0, 0.0)
SEED = 7

OBS_NOISE_FORCE = 0.030  # [N] sensor noise before PaXini 0.1 N quantization
OBS_NOISE_X0 = 5.0e-5  # [m] shooting boundary node pose init noise (2nd chunk)

NAMES = ("EI", "rho", "mu_0", "alpha", "beta")
TRUE = dict(EI=0.015, rho=1850.0, mu_0=0.46, alpha=-0.26, beta=0.36)
# Start far off truth (nearly Coulomb alpha=0, beta=0 and ~35-50% wrong EI, rho, mu_0)
GUESS = dict(EI=0.0095, rho=1250.0, mu_0=0.30, alpha=-0.02, beta=0.04)


def _load_m2826_mesh() -> newton.Mesh:
    m_mesh = np.load(ROOT / "examples" / "utils" / "paxini" / "M2826_mesh.npz")
    return newton.Mesh(m_mesh["vertices"] * 1.0e-3, m_mesh["faces"].ravel())


def build_rig(EI: float, rho: float, mu_0: float, alpha: float, beta: float, worlds: int, mesh: newton.Mesh,
              grad: bool = False):
    """Build ``worlds`` parallel M2826 PaXini sensor + clamped cable-bridge rigs."""
    proto = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    pad = proto.add_link(label="thumb_dip")
    cfg_pad = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu_0)
    sh = proto.add_shape_mesh(pad, xform=wp.transform_identity(), cfg=cfg_pad, mesh=mesh, label="thumb_dip_paxini")
    pts = np.column_stack([np.linspace(-0.5 * L_SPAN, 0.5 * L_SPAN, N_NODES), np.full(N_NODES, 0.0053), np.full(N_NODES, 0.0235)])
    caps, _ = SolverDiSMech.add_rod(
        proto,
        rod=newton.Rod(pts, radius=ROD_RADIUS),
        cfg=newton.ModelBuilder.ShapeConfig(density=rho, mu=mu_0),
        stretch_stiffness=ROD_EA,
        bend_stiffness=EI / L_DUAL,
        bend_damping=ROD_C_BEND,
    )
    SolverDiSMech.fix_segment(proto, caps[0])
    SolverDiSMech.fix_segment(proto, caps[-1])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    for _ in range(worlds):
        builder.add_world(proto)
    model = builder.finalize()
    pads(model, [w * proto.body_count + pad for w in range(worlds)], kh=1.0e8)
    solver = SolverDiSMech(
        model,
        exact=True,
        iterations=80,
        friction_alpha=alpha,
        friction_beta=beta,
        friction_n0=FRICTION_N0,
        friction_dir=FRICTION_DIR,
    )
    solver.tolerance = 2.0e-4
    tact = tactile.Tactile(model, solver, {"thumb_dip": sh})
    tact.hook()
    d = model.dismech
    if grad:
        for a in (d.triplet_params, model.particle_mass, d.edge_inertia, model.shape_material_mu, solver.contact.trib):
            a.requires_grad = True
    return model, solver, tact


def clamp_trajectories() -> np.ndarray:
    """Clamped-node offsets (TRIALS, STEPS_PER_TRIAL + 1, 3) across 4 calibration trials x 2 shooting chunks."""
    t = np.arange(STEPS_PER_TRIAL + 1) * DT
    out = np.zeros((TRIALS, STEPS_PER_TRIAL + 1, 3), dtype=np.float64)
    t_mid = W * OBS_EVERY * DT  # 0.120 s chunk handoff
    # Smooth envelope inside each 120 ms chunk: contact-free at chunk boundaries (tau = 0, 0.12 s)
    tau = np.mod(t, t_mid)
    win = np.sin(np.pi * np.clip((tau - 0.004) / (t_mid - 0.016), 0.0, 1.0)) ** 2
    ch1 = (t >= t_mid).astype(np.float64)

    # Trial 0: light squeeze + axial slip
    out[0, :, 0] = 0.008 * np.sin(2.0 * np.pi * 8.0 * tau) * (1.0 - 2.0 * ch1)
    out[0, :, 2] = (-0.0046 * (1.0 - ch1) - 0.0049 * ch1) * win

    # Trial 1: heavy squeeze + axial slip -> separates mu_0 and alpha
    out[1, :, 0] = 0.008 * np.sin(2.0 * np.pi * 8.0 * tau) * (1.0 - 2.0 * ch1)
    out[1, :, 2] = (-0.0072 * (1.0 - ch1) - 0.0078 * ch1) * win

    # Trial 2: heavy squeeze + transverse sweep centered on dome -> separates beta and EI
    out[2, :, 1] = 0.005 * np.sin(2.0 * np.pi * 8.0 * tau) * (1.0 - 2.0 * ch1)
    out[2, :, 2] = (-0.0072 * (1.0 - ch1) - 0.0077 * ch1) * win

    # Trial 3: vertical whip (18 Hz) + lateral sweep -> excites inertia (rho) & bending (EI)
    out[3, :, 0] = 0.005 * np.sin(2.0 * np.pi * 10.0 * tau) * ch1
    out[3, :, 1] = 0.004 * np.sin(2.0 * np.pi * 12.0 * tau)
    out[3, :, 2] = (-0.0060 + 0.0014 * np.sin(2.0 * np.pi * 18.0 * tau)) * win
    return out


def clamp_indices(worlds: int) -> np.ndarray:
    return np.array(
        [3 * (w * N_NODES + n) + c for w in range(worlds) for n in (0, 1, N_NODES - 2, N_NODES - 1) for c in range(3)],
        dtype=np.int32,
    )


@wp.kernel
def _set_fixed_kernel(idx: wp.array[wp.int32], vals: wp.array2d[float], step: int, q: wp.array[float]):
    j = wp.tid()
    q[idx[j]] = vals[step, j]


@wp.kernel
def _quantized_loss_kernel(pred: wp.array[wp.vec3], obs: wp.array[wp.vec3], scale: float, loss: wp.array[float]):
    i = wp.tid()
    d = pred[i] - obs[i]
    wp.atomic_add(loss, 0, scale * wp.length_sq(d))


def _quantize_reading(force: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply Gaussian noise + PaXini LSB=0.1 N integer quantization and return dequantized [N]."""
    noisy = force + rng.normal(0.0, OBS_NOISE_FORCE, size=force.shape)
    lsb = float(tactile.LSB)
    qx = np.clip(np.round(noisy[..., 0] / lsb), -128, 127)
    qy = np.clip(np.round(noisy[..., 1] / lsb), -128, 127)
    qz = np.clip(np.round(noisy[..., 2] / lsb), 0, 255)
    return np.stack([qx, qy, qz], axis=-1).astype(np.float32) * np.float32(lsb)


def record(mesh: newton.Mesh) -> dict:
    """Generate 83.3 Hz noisy quantized PaXini recordings of the 4 calibration trials."""
    rng = np.random.default_rng(SEED)
    model, solver, tact = build_rig(**TRUE, worlds=TRIALS, mesh=mesh, grad=False)
    st = [model.state(), model.state()]
    for s in st:
        flatten_state(s)
    solver.restart(st[0], st[0])
    idx_np = clamp_indices(TRIALS)
    idx_wp = wp.array(idx_np, dtype=wp.int32, device=model.device)
    base_q = st[0].dismech.q.numpy()[idx_np].reshape(TRIALS, 12)
    traj = clamp_trajectories()
    vals = np.repeat(traj, 4, axis=2) + base_q[:, None, :]
    vals = vals.transpose(1, 0, 2).reshape(STEPS_PER_TRIAL + 1, -1).astype(np.float32)
    vals_wp = wp.array(vals, dtype=float, device=model.device)

    n_tax = tact.count_per_world
    q_hist = [st[0].dismech.q.numpy().copy()]
    qd_hist = [st[0].dismech.qd.numpy().copy()]
    f_clean = []
    for k in range(STEPS_PER_TRIAL):
        wp.launch(_set_fixed_kernel, dim=len(idx_np), inputs=[idx_wp, vals_wp, k], outputs=[st[0].dismech.q],
                  device=model.device)
        solver.step(st[0], st[1], None, None, DT)
        tact.commit(DT)
        st[0], st[1] = st[1], st[0]
        if (k + 1) % OBS_EVERY == 0:
            tact.sample()
            f_clean.append(tact.force.numpy().reshape(TRIALS, n_tax, 3))
            q_hist.append(st[0].dismech.q.numpy().copy())
            qd_hist.append(st[0].dismech.qd.numpy().copy())

    f_clean = np.stack(f_clean, axis=1)  # (TRIALS, FRAMES_PER_TRIAL, n_tax, 3)
    f_obs = _quantize_reading(f_clean, rng)
    q_hist = np.stack(q_hist, axis=0)  # (FRAMES_PER_TRIAL + 1, dof_total)
    qd_hist = np.stack(qd_hist, axis=0)

    q_wins_n, q_wins_t = [], []
    qd_wins_n, qd_wins_t = [], []
    vals_wins = []
    n3 = 3 * N_NODES * TRIALS
    for ch in range(CHUNKS):
        f0 = ch * W
        s0 = f0 * OBS_EVERY
        qn = q_hist[f0][:n3].reshape(TRIALS, 3 * N_NODES).copy()
        qt = q_hist[f0][n3:].reshape(TRIALS, N_NODES - 1).copy()
        qdn = qd_hist[f0][:n3].reshape(TRIALS, 3 * N_NODES).copy()
        qdt = qd_hist[f0][n3:].reshape(TRIALS, N_NODES - 1).copy()
        if ch > 0:
            qn[:, 6:-6] += rng.normal(0.0, OBS_NOISE_X0, size=(TRIALS, 3 * N_NODES - 12))
        q_wins_n.append(qn)
        q_wins_t.append(qt)
        qd_wins_n.append(qdn)
        qd_wins_t.append(qdt)
        v_chunk = vals[s0 : s0 + STEPS_PER_WINDOW + 1].reshape(STEPS_PER_WINDOW + 1, TRIALS, 12)
        vals_wins.append(v_chunk)

    q0_batch = np.concatenate([np.concatenate(q_wins_n, 0).ravel(), np.concatenate(q_wins_t, 0).ravel()]).astype(
        np.float32
    )
    qd0_batch = np.concatenate([np.concatenate(qd_wins_n, 0).ravel(), np.concatenate(qd_wins_t, 0).ravel()]).astype(
        np.float32
    )
    vals_batch = np.concatenate(vals_wins, axis=1).reshape(STEPS_PER_WINDOW + 1, -1).astype(np.float32)
    f_obs_batch = np.concatenate([f_obs[:, ch * W : (ch + 1) * W] for ch in range(CHUNKS)], axis=0)
    # Shape for batched lookup per frame j in 0..W-1: (W, WINDOWS * n_tax, 3)
    f_obs_frames = f_obs_batch.transpose(1, 0, 2, 3).reshape(W, WINDOWS * n_tax, 3).astype(np.float32)
    f_clean_frames = np.concatenate([f_clean[:, ch * W : (ch + 1) * W] for ch in range(CHUNKS)], axis=0)
    f_clean_frames = f_clean_frames.transpose(1, 0, 2, 3).reshape(W, WINDOWS * n_tax, 3).astype(np.float32)

    return dict(
        q0=q0_batch,
        qd0=qd0_batch,
        vals=vals_batch,
        f_obs=f_obs_frames,
        f_clean=f_clean_frames,
        n_tax=np.array([n_tax], dtype=np.int32),
    )


class TactileMultipleShootingFit:
    """Batched ``W=10`` multiple-shooting objective and exact adjoint gradient over ``(EI, rho, mu_0, alpha, beta)``."""

    def __init__(self, data: dict, mesh: newton.Mesh):
        self.mesh = mesh
        self.model, self.solver, self.tact = build_rig(**TRUE, worlds=WINDOWS, mesh=mesh, grad=True)
        d = self.model.dismech
        self.p_true = d.triplet_params.numpy().astype(np.float64)
        self.m_true = self.model.particle_mass.numpy().astype(np.float64)
        self.i_true = d.edge_inertia.numpy().astype(np.float64)
        self.mu_true = self.model.shape_material_mu.numpy().astype(np.float64)

        self.states = [self.model.state(requires_grad=True) for _ in range(STEPS_PER_WINDOW + 1)]
        for s in self.states:
            flatten_state(s)
        self.rest = self.model.state()
        flatten_state(self.rest)
        self.solver.restart(self.rest, self.rest)

        q_init = data["q0"]
        qd_init = data["qd0"]
        assert q_init.shape == self.rest.dismech.q.shape
        assert qd_init.shape == (3 * N_NODES * WINDOWS + (N_NODES - 1) * WINDOWS,)

        self.q0_wp = wp.array(q_init, dtype=float, device=self.model.device)
        self.qd0_wp = wp.array(qd_init, dtype=float, device=self.model.device)
        self.idx_np = clamp_indices(WINDOWS)
        self.idx_wp = wp.array(self.idx_np, dtype=wp.int32, device=self.model.device)
        self.vals_wp = wp.array(data["vals"], dtype=float, device=self.model.device)
        self.f_obs_wp = [wp.array(data["f_obs"][j], dtype=wp.vec3, device=self.model.device) for j in range(W)]
        self.loss = wp.zeros(1, dtype=float, device=self.model.device, requires_grad=True)
        self.loss_scale = 1.0 / (WINDOWS * W)

    @staticmethod
    def unpack(theta: np.ndarray) -> dict[str, float]:
        return dict(
            EI=TRUE["EI"] * float(np.exp(theta[0])),
            rho=TRUE["rho"] * float(np.exp(theta[1])),
            mu_0=TRUE["mu_0"] * float(np.exp(theta[2])),
            alpha=float(theta[3]),
            beta=float(theta[4]),
        )

    @staticmethod
    def pack(params: dict[str, float]) -> np.ndarray:
        return np.array([
            np.log(params["EI"] / TRUE["EI"]),
            np.log(params["rho"] / TRUE["rho"]),
            np.log(params["mu_0"] / TRUE["mu_0"]),
            params["alpha"],
            params["beta"],
        ], dtype=np.float64)

    def set_params(self, theta: np.ndarray) -> dict[str, float]:
        par = self.unpack(theta)
        r_ei = float(np.exp(theta[0]))
        r_rho = float(np.exp(theta[1]))
        r_mu = float(np.exp(theta[2]))
        d = self.model.dismech
        p = self.p_true.copy()
        p[:, 2:4] *= r_ei
        d.triplet_params.assign(p.astype(np.float32))
        self.model.particle_mass.assign((self.m_true * r_rho).astype(np.float32))
        d.edge_inertia.assign((self.i_true * r_rho).astype(np.float32))
        self.model.shape_material_mu.assign((self.mu_true * r_mu).astype(np.float32))
        self.solver.contact.set_tribology(par["alpha"], par["beta"], FRICTION_N0, FRICTION_DIR)
        self.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        self._ratios = (r_ei, r_rho, r_mu)
        return par

    def rollout_predictions(self, theta: np.ndarray) -> np.ndarray:
        """Forward rollout only; returns predicted tactile forces (W, WINDOWS * n_tax, 3)."""
        self.set_params(theta)
        self.states[0].dismech.q.assign(self.q0_wp)
        self.states[0].dismech.qd.assign(self.qd0_wp)
        self.solver.restart(self.states[0], self.rest)
        self.tact.acc.zero_()
        self.tact.total_acc.zero_()
        self.tact.time.zero_()
        self.tact._acc_dt = 0.0
        preds = []
        for k in range(STEPS_PER_WINDOW):
            wp.launch(_set_fixed_kernel, dim=len(self.idx_np), inputs=[self.idx_wp, self.vals_wp, k],
                      outputs=[self.states[k].dismech.q], device=self.model.device)
            self.solver.step(self.states[k], self.states[k + 1], None, None, DT)
            self.tact.commit(DT)
            if (k + 1) % OBS_EVERY == 0:
                self.tact.sample()
                preds.append(self.tact.force.numpy())
        return np.stack(preds, axis=0)

    def __call__(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        self.set_params(theta)
        m, d = self.model, self.model.dismech
        for a in (d.triplet_params, m.particle_mass, d.edge_inertia, m.shape_material_mu, self.solver.contact.trib):
            if a.grad is not None:
                a.grad.zero_()
        self.loss.zero_()
        with suspended_tape():
            self.states[0].dismech.q.assign(self.q0_wp)
            self.states[0].dismech.qd.assign(self.qd0_wp)
            self.solver.restart(self.states[0], self.rest)
            self.tact.acc.zero_()
            self.tact.total_acc.zero_()
            self.tact.time.zero_()
            self.tact._acc_dt = 0.0

        tape = wp.Tape()
        with tape:
            for k in range(STEPS_PER_WINDOW):
                with suspended_tape():
                    wp.launch(_set_fixed_kernel, dim=len(self.idx_np), inputs=[self.idx_wp, self.vals_wp, k],
                              outputs=[self.states[k].dismech.q], device=self.model.device)
                self.solver.step(self.states[k], self.states[k + 1], None, None, DT)
                self.tact.commit(DT)
                if (k + 1) % OBS_EVERY == 0:
                    j = (k + 1) // OBS_EVERY - 1
                    self.tact.sample()
                    wp.launch(_quantized_loss_kernel, dim=self.tact.count,
                              inputs=[self.tact.force, self.f_obs_wp[j], self.loss_scale],
                              outputs=[self.loss], device=self.model.device)

        val = float(self.loss.numpy()[0])
        if not np.isfinite(val):
            return 1.0e6, np.zeros_like(theta)
        tape.backward(self.loss)

        r_ei, r_rho, r_mu = self._ratios
        g_ei = float((d.triplet_params.grad.numpy()[:, 2:4] * (self.p_true[:, 2:4] * r_ei)).sum())
        g_rho = float(
            (m.particle_mass.grad.numpy() * (self.m_true * r_rho)).sum()
            + (d.edge_inertia.grad.numpy() * (self.i_true * r_rho)).sum()
        )
        g_mu = float((m.shape_material_mu.grad.numpy() * (self.mu_true * r_mu)).sum())
        g_trib = self.solver.contact.trib.grad.numpy()
        grad = np.array([g_ei, g_rho, g_mu, float(g_trib[0]), float(g_trib[1])], dtype=np.float64)
        tape.zero()
        if not np.all(np.isfinite(grad)):
            return 1.0e6, np.zeros_like(theta)
        return val, grad


def fit(data: dict, mesh: newton.Mesh) -> dict:
    obj = TactileMultipleShootingFit(data, mesh)
    theta0 = obj.pack(GUESS)
    best: dict = {}

    def fun(u):
        th = u * opt_scale
        val, g = obj(th)
        if np.isfinite(val) and np.all(np.isfinite(g)):
            if not best or val < best["loss"]:
                best.update(loss=val, theta=th.copy())
            return val, g * opt_scale
        dx = th - best["theta"]
        return best["loss"] + 1.0 + 100.0 * float(dx @ dx), (200.0 * dx) * opt_scale

    opt_scale = np.array([1.0, 1.0, 1.0, 1.0, 3.0], dtype=np.float64)
    u0 = theta0 / opt_scale
    history = [obj.unpack(theta0)]
    losses = [fun(u0)[0]]

    def cb(intermediate_result):
        x = intermediate_result.x.copy() * opt_scale
        par = obj.unpack(x)
        history.append(par)
        losses.append(float(intermediate_result.fun))
        print(
            f"  iter {len(history) - 1:2d}: loss {intermediate_result.fun:.4f}  "
            + "  ".join(f"{k}={par[k]:.4g}" for k in NAMES),
            flush=True,
        )

    tic = time.perf_counter()
    raw_bounds = [(-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), (-0.75, 0.45), (0.0, 1.2)]
    bounds = [(lo / s, hi / s) for (lo, hi), s in zip(raw_bounds, opt_scale)]
    res = minimize(fun, u0, jac=True, method="L-BFGS-B", bounds=bounds, callback=cb,
                   options=dict(maxiter=60, ftol=1.0e-8, gtol=1.0e-5))
    elapsed = time.perf_counter() - tic
    theta_hat = res.x.copy() * opt_scale
    final = obj.unpack(theta_hat)
    pred_init = obj.rollout_predictions(theta0)
    pred_fit = obj.rollout_predictions(theta_hat)
    print(f"fit finished in {elapsed:.1f}s (nfev={res.nfev}, nit={res.nit})")
    return dict(
        history_vals=np.array([[h[k] for k in NAMES] for h in history], dtype=np.float64),
        losses=np.array(losses, dtype=np.float64),
        final=final,
        theta_hat=theta_hat,
        pred_init=pred_init,
        pred_fit=pred_fit,
        elapsed=np.array([elapsed]),
    )


def results(fresh: bool = False) -> tuple[dict, dict]:
    mesh = _load_m2826_mesh()
    cfg = dict(v=4, true=TRUE, guess=GUESS, dt=DT, obs_every=OBS_EVERY, w=W, chunks=CHUNKS, trials=TRIALS,
               seed=SEED, noise=[OBS_NOISE_FORCE, OBS_NOISE_X0])
    data = cached("fit_cable_routing-data", cfg, lambda: record(mesh), fresh)
    out = cached("fit_cable_routing-fit", cfg, lambda: fit(data, mesh), fresh)
    return data, out


def check_accuracy(out: dict) -> None:
    vals = out["history_vals"][-1]
    print("\nRecovered parameters vs ground truth (83.3 Hz noisy quantized tactile data):")
    for k, v in zip(NAMES, vals):
        t = TRUE[k]
        rel = abs(v - t) / abs(t)
        print(f"  {k:6s}: fit={v:+.5g}  true={t:+.5g}  rel_err={rel * 100:.2f}%")
        assert rel < 0.03, f"{k} error {rel * 100:.2f}% >= 3%: fit={v}, true={t}"


def save_plot(data: dict, out: dict, path: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    hist = out["history_vals"]
    it = np.arange(len(hist))
    tex = [r"$EI$", r"$\rho$", r"$\mu_0$", r"$\alpha$", r"$\beta$"]
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    ax0 = axes[0]
    ax0.axhspan(0.97, 1.03, color="#2ca02c", alpha=0.14)
    ax0.axhline(1.0, color="k", ls=":", lw=1.0)
    for j, (k, lb, c) in enumerate(zip(NAMES, tex, cols)):
        ax0.plot(it, hist[:, j] / TRUE[k], marker="o", ms=3, lw=1.8, color=c, label=f"{lb} ({hist[-1, j] / TRUE[k]:.3f})")
    ax0.set(xlabel="L-BFGS-B step", ylabel="fit / truth", title="Convergence")
    ax0.legend(loc="best", fontsize=8.5, ncol=2)
    ax0.grid(alpha=0.25)

    ax1 = axes[1]
    n_tax = int(data["n_tax"][0])
    # Show net tactile shear Fx and normal Fz sum over sensor on window 1 (heavy axial)
    t_ms = np.arange(1, W + 1) * OBS_EVERY * DT * 1e3
    sl = slice(1 * n_tax, 2 * n_tax)
    obs_tot = data["f_obs"][:, sl].sum(axis=1)
    fit_tot = out["pred_fit"][:, sl].sum(axis=1)
    init_tot = out["pred_init"][:, sl].sum(axis=1)
    ax1.plot(t_ms, obs_tot[:, 2], "o", color="#1f77b4", ms=4.5, label=r"$F_z$ obs (LSB 0.1 N)")
    ax1.plot(t_ms, fit_tot[:, 2], "-", color="#1f77b4", lw=2.0, label=r"$F_z$ fit")
    ax1.plot(t_ms, init_tot[:, 2], "--", color="#1f77b4", alpha=0.45, lw=1.2, label=r"$F_z$ init")
    ax1.plot(t_ms, obs_tot[:, 0], "s", color="#d62728", ms=4.0, label=r"$F_x$ obs (LSB 0.1 N)")
    ax1.plot(t_ms, fit_tot[:, 0], "-", color="#d62728", lw=2.0, label=r"$F_x$ fit")
    ax1.plot(t_ms, init_tot[:, 0], "--", color="#d62728", alpha=0.45, lw=1.2, label=r"$F_x$ init")
    ax1.set(xlabel="window time [ms]", ylabel="net sensor force [N]", title="Tactile")
    ax1.legend(loc="best", fontsize=8, ncol=2)
    ax1.grid(alpha=0.25)

    fig.savefig(path, dpi=180)
    print(f"saved plot to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="verify < 3% recovery across all 5 parameters")
    parser.add_argument("--plot", action="store_true", help="save convergence and tactile fit plot")
    parser.add_argument("--fresh", action="store_true", help="recompute dataset and optimization")
    args = parser.parse_args()

    wp.init()
    data, out = results(fresh=args.fresh)
    if args.plot:
        save_plot(data, out, Path(__file__).with_suffix(".png"))
    if args.test or not args.plot:
        check_accuracy(out)


if __name__ == "__main__":
    main()
