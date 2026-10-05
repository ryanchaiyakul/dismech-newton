"""Baseline system-ID dataset: a clamped rod driven past buckling by a noisy gripper, many trials, one command.

A rod in air under gravity (-z), weak bending plane z, clamped at the left edge (nodes 0, 1 and edge 0's twist).
The right edge is the gripper: its 6-DOF pose drives the fixed DOFs (nodes N-2, N-1 and the last edge's twist).
Every trial gets the *same* nominal command (compress along -x, shear along +z, hold), and its own

* actuation noise: the gripper's actual pose = command + smooth tracking error (low-pass filtered Gaussian noise
  on all 6 DOFs, ramped in from zero so t = 0 matches the rest clamp),
* initial-state noise: the free nodes start from the static gravity-sag equilibrium plus random sine modes in y
  and z (zero velocity).

Rest geometry is the straight rod for every trial (constant rest lengths and zero rest curvature). The forward
runs are deterministic (Newton, batched worlds). Observation noise is added afterwards to every node position
(free and fixed) and to the gripper pose readout. The compression buckles the rod; with gravity's sag bias
cancelled only by the noise, trials fork up or down.

    uv run scratch/sysid/gen_baseline.py scan        # P(up) vs density scale (picks the gravity bias)
    uv run scratch/sysid/gen_baseline.py gen [N]     # dataset -> baseline.npz, baseline.png
    uv run scratch/sysid/gen_baseline.py plot        # re-plot baseline.npz
    uv run scratch/sysid/gen_baseline.py check       # sanity checks on baseline.npz
"""

import sys
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp

from dismech_newton import DiSMechSolver, add_rod, flatten_state
from dismech_newton.solver import advance_frames_kernel
from dismech_newton.triplet import advance_ref_twist_kernel

wp.config.log_level = wp.LOG_WARNING
HERE = Path(__file__).parent
OUT = HERE / "baseline.npz"
DEVICE = "cuda:0"

# -- rod (truth parameters) -----------------------------------------------------------------------------------------
LENGTH, SEGMENTS, RADIUS = 0.5, 24, 0.01
NODES = SEGMENTS + 1
MID = NODES // 2
BEND, BEND_STRONG_RATIO = 10.0, 4.0  # weak-plane (z) bend stiffness, strong (y) / weak
DAMPING, STRETCH, TWIST = 0.3, 1.0e6, 10.0
DENSITY_SCALE = 0.1  # x the default density (100 kg/m^3, a foam cord): gravity sag 0.2 mm, so the gripper noise can beat it
GRAVITY = 9.81
SOLVER_THETA = 1.0  # 1 = implicit Euler, 0.5 = trapezoidal (no numerical damping)

# -- time and the nominal command -----------------------------------------------------------------------------------
DT = 5.0e-3
T_COMPRESS, T_SHEAR, T_HOLD = 1.5, 1.5, 1.0
T_END = T_COMPRESS + T_SHEAR + T_HOLD
STEPS = int(round(T_END / DT))
COMPRESS = 4.0e-3  # gripper travel along -x [m]
SHEAR = 1.0e-2  # gripper travel along +z [m]
T_FORK = 0.5  # roughly when the branch is decided (diagnostics only)
OBS_EVERY = 5  # record every 5 steps (40 Hz)
N_OBS = STEPS // OBS_EVERY
K_BUCKLE = int(round(T_COMPRESS / DT)) // OBS_EVERY - 1  # frame at the end of compression

# -- per-trial noise (process) --------------------------------------------------------------------------------------
S_GRIP_T = 2.0e-4  # gripper tracking error, translation [m]
S_GRIP_R = np.radians(0.6)  # gripper tracking error, rotation [rad]; its pitch at the fork decides the branch
TAU_GRIP = 0.15  # tracking-error correlation time (gaussian filter sigma) [s]
T_RAMP = 0.2  # tracking error ramps in over this time
S_Q0 = 3.0e-4  # initial free-node perturbation, first sine mode [m] (mode k: S_Q0 / k)
Q0_MODES = 4

# -- observation noise ----------------------------------------------------------------------------------------------
OBS_NOISE_X = 5.0e-4  # node positions (free and fixed) [m]
OBS_NOISE_GRIP_T, OBS_NOISE_GRIP_R = 5.0e-4, np.radians(0.2)  # gripper pose readout


def _timing():
    """Re-derive the step counts from the phase durations (after overriding them)."""
    global T_END, STEPS, N_OBS, K_BUCKLE
    T_END = T_COMPRESS + T_SHEAR + T_HOLD
    STEPS = int(round(T_END / DT))
    N_OBS = STEPS // OBS_EVERY
    K_BUCKLE = int(round(T_COMPRESS / DT)) // OBS_EVERY - 1


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def command() -> np.ndarray:
    """Nominal gripper pose (STEPS + 1, 6) = (translation, rotation vector) of the right edge from its rest pose;
    row i is applied before step i (row STEPS is the final pose)."""
    t = np.arange(STEPS + 1) * DT
    u = np.zeros((STEPS + 1, 6))
    u[:, 0] = -COMPRESS * smooth(t / T_COMPRESS)
    u[:, 2] = SHEAR * smooth((t - T_COMPRESS) / T_SHEAR)
    return u


def tracking_error(rng, trials: int) -> np.ndarray:
    """Smooth gripper tracking error (trials, STEPS + 1, 6): unit-variance low-pass noise, scaled and ramped in."""
    sig = TAU_GRIP / DT
    half = int(4 * sig)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2)
    k /= np.sqrt((k * k).sum())  # stationary unit variance
    w = rng.normal(size=(trials, STEPS + 1 + 2 * half, 6))
    e = np.stack([np.stack([np.convolve(w[b, :, j], k, mode="valid") for j in range(6)], -1) for b in range(trials)])
    e *= np.r_[[S_GRIP_T] * 3, [S_GRIP_R] * 3]
    return e * smooth(np.arange(STEPS + 1) * DT / T_RAMP)[None, :, None]


def q0_perturbation(rng, trials: int, s_q0: float = None) -> tuple[np.ndarray, np.ndarray]:
    """Initial free-node offsets (trials, NODES, 3) from sine modes in y and z; returns (offsets, coefficients)."""
    s = np.linspace(0.0, 1.0, NODES)
    modes = np.array([np.sin(k * np.pi * s) for k in range(1, Q0_MODES + 1)])
    s_q0 = S_Q0 if s_q0 is None else s_q0
    c = rng.normal(size=(trials, Q0_MODES, 2)) * (s_q0 / np.arange(1, Q0_MODES + 1))[None, :, None]
    d = np.zeros((trials, NODES, 3))
    d[:, :, 1:] = np.einsum("bmc,mn->bnc", c, modes)
    d[:, [0, 1, -2, -1]] = 0.0  # fixed DOFs are not perturbed
    return d, c


# -- model ----------------------------------------------------------------------------------------------------------
def build(rods: int, density_scale: float):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
    for r in range(rods):
        rod = newton.Rod.create_straight((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), LENGTH, segment_count=SEGMENTS,
                                         radius=RADIUS)
        add_rod(builder, rod, stretch_stiffness=STRETCH, bend_stiffness=BEND, twist_stiffness=TWIST,
                bend_damping=DAMPING, proxies=False)
        DiSMechSolver.fix_segment(builder, edge=r * SEGMENTS)
        DiSMechSolver.fix_segment(builder, edge=r * SEGMENTS + SEGMENTS - 1)
    model = builder.finalize(device=DEVICE)
    p = model.dismech.triplet_params.numpy()
    p[:, 3] = BEND * BEND_STRONG_RATIO
    model.dismech.triplet_params.assign(p)
    model.particle_mass.assign(model.particle_mass.numpy() * density_scale)
    model.dismech.edge_inertia.assign(model.dismech.edge_inertia.numpy() * density_scale)
    solver = DiSMechSolver(model, theta=SOLVER_THETA)
    solver.refresh_mass()
    return model, solver


@wp.kernel
def _grip_kernel(p0: wp.array[wp.vec3], pose: wp.array[wp.types.vector(6, float)], nodes: int, segments: int,
                 twist0: int, q: wp.array[float]):
    """Right clamped edge of rod r: rigid pose (translation, rotation vector about the edge centre) of its rest edge."""
    r = wp.tid()
    a = p0[2 * r]
    b = p0[2 * r + 1]
    c = 0.5 * (a + b)
    e = pose[r]
    w = wp.vec3(e[3], e[4], e[5])
    ang = wp.length(w)
    rot = wp.quat_identity()
    if ang > 0.0:
        rot = wp.quat_from_axis_angle(w / ang, ang)
    t = wp.vec3(e[0], e[1], e[2])
    na = c + wp.quat_rotate(rot, a - c) + t
    nb = c + wp.quat_rotate(rot, b - c) + t
    i0 = r * nodes + nodes - 2
    for j in range(3):
        q[3 * i0 + j] = na[j]
        q[3 * (i0 + 1) + j] = nb[j]
    q[twist0 + r * segments + segments - 1] = w[0]  # the edge's twist: roll about +x


class Sim:
    def __init__(self, rods: int, density_scale: float = DENSITY_SCALE):
        self.rods = rods
        self.model, self.solver = build(rods, density_scale)
        m = self.model
        self.x_rest = m.particle_q.numpy().reshape(rods, NODES, 3).copy()
        self.p0 = wp.array(self.x_rest[:, -2:].reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=m.device)
        self.pose = wp.zeros(rods, dtype=wp.types.vector(6, float), device=m.device)
        self.states = [m.state(), m.state()]
        self.rest = m.state()  # the straight rest state: frames, reference twist and strains consistent
        self.base = m.state()  # the gravity-sag equilibrium (set by `sag`)
        for s in (*self.states, self.rest, self.base):
            flatten_state(s)
        self.twist0 = 3 * m.particle_count
        self.solver.triplets.measure(self.rest, self.rest.dismech.triplet_strain_q)
        self._copy(self.base, self.rest)

    def _grip(self, st, pose):
        self.pose.assign(pose.astype(np.float32))
        wp.launch(_grip_kernel, dim=self.rods, inputs=[self.p0, self.pose, NODES, SEGMENTS, self.twist0],
                  outputs=[st.dismech.q], device=self.model.device)

    @staticmethod
    def _copy(dst, src):
        for k in ("q", "qd", "edge_d1_q", "triplet_ref_twist_q", "triplet_strain_q"):
            getattr(dst.dismech, k).assign(getattr(src.dismech, k))

    def _start(self, st, src, x=None, twist=None):
        """``st`` = ``src`` moved to positions ``x`` / twists ``twist`` at rest: the edge frames parallel-transported
        and the reference twist advanced from ``src`` (as at the end of a step), the strains re-measured. Every
        rollout must start like this: the frames and stored strains are state, and stale ones kick the rod."""
        self._copy(st, src)
        q = st.dismech.q.numpy()
        if x is not None:
            q[: self.twist0] = x.reshape(-1)
        if twist is not None:
            q[self.twist0:] = twist.reshape(-1)
        st.dismech.q.assign(q)
        st.dismech.qd.zero_()
        d, tr = self.solver.der, self.solver.triplets
        wp.launch(advance_frames_kernel, dim=d.edge_length.shape[0],
                  inputs=[src.particle_q, st.particle_q, d.edge_node0, d.edge_node1, src.dismech.edge_d1_q],
                  outputs=[st.dismech.edge_d1_q], device=self.model.device)
        wp.launch(advance_ref_twist_kernel, dim=tr.count,
                  inputs=[st.dismech.q, st.dismech.edge_d1_q, tr.conn, src.dismech.triplet_ref_twist_q],
                  outputs=[st.dismech.triplet_ref_twist_q], device=self.model.device)
        tr.measure(st, st.dismech.triplet_strain_q)

    def sag(self, steps=40, dt=0.5) -> tuple[np.ndarray, np.ndarray]:
        """Static gravity-sag equilibrium with both clamps at rest (big implicit steps = a damped static solve);
        stored as the base state every rollout starts from."""
        a, b = self.states
        self._start(a, self.rest)
        for _ in range(steps):
            a.dismech.qd.zero_()
            self.solver.step(a, b, None, None, dt)
            a, b = b, a
        self._copy(self.base, a)
        self.base.dismech.qd.zero_()
        q = a.dismech.q.numpy()
        return q[: self.twist0].reshape(self.rods, NODES, 3), q[self.twist0:].reshape(self.rods, SEGMENTS)

    def run(self, x0, twist0, grip, every=OBS_EVERY, until=STEPS):
        """Roll out from (x0, twist0) with the gripper at grip[:, i] before step i. Returns node positions
        (rods, frames, NODES, 3) and twists (rods, frames, SEGMENTS) at t = 0, every, 2 every, ..."""
        a, b = self.states
        self._start(a, self.base, x0, twist0)
        X, TW = [], []

        def rec(st):
            q = st.dismech.q.numpy()
            X.append(q[: self.twist0].reshape(self.rods, NODES, 3))
            TW.append(q[self.twist0:].reshape(self.rods, SEGMENTS))

        self._grip(a, grip[:, 0])
        rec(a)
        for i in range(until):
            self._grip(a, grip[:, i])
            self.solver.step(a, b, None, None, DT)
            a, b = b, a
            if (i + 1) % every == 0:
                self._grip(a, grip[:, i + 1])  # the recorded fixed DOFs are the pose at that instant
                rec(a)
        return np.stack(X, 1), np.stack(TW, 1)


def mid_rel(X):
    """Mid-span z relative to the end nodes' chord (the buckling read-out; plots and labels only)."""
    return X[..., MID, 2] - 0.5 * (X[..., 0, 2] + X[..., -1, 2])


def sample(rng, trials, sim_sag, s_q0=None):
    x_sag, tw_sag = sim_sag
    dq0, c0 = q0_perturbation(rng, trials, s_q0)
    grip_err = tracking_error(rng, trials)
    return x_sag[:1] + dq0, np.repeat(tw_sag[:1], trials, 0), grip_err, c0


# -- modes ----------------------------------------------------------------------------------------------------------
def scan(trials=128, seed=3, configs=None):
    """What decides the branch: P(up) at the end of compression, and the fraction of trials whose branch flips when
    the initial perturbation (or the gripper error) is removed. ``configs``: (label, {module global: value})."""
    configs = configs or [("as configured", {})]
    for label, over in configs:
        old = {k: globals()[k] for k in (*over, "T_END", "STEPS", "N_OBS", "K_BUCKLE")}
        globals().update(over)
        _timing()
        try:
            sim = Sim(trials, globals()["DENSITY_SCALE"])
            sag = sim.sag()
            x0, tw0, err, _ = sample(np.random.default_rng(seed), trials, sag)
            u = command()[None]
            until = (K_BUCKLE + 1) * OBS_EVERY
            up = lambda x, gr: mid_rel(sim.run(x, tw0, gr, until=until)[0][:, -1]) > 0
            full = up(x0, u + err)
            no_q0 = up(np.repeat(sag[0][:1], trials, 0), u + err)
            no_grip = up(x0, u + 0 * err)
            print(f"{label:40s} sag {mid_rel(sag[0][0]) * 1e3:+.2f} mm  P(up) {full.mean():.2f} | flips w/o q0 "
                  f"{np.mean(full != no_q0):.2f}, w/o grip {np.mean(full != no_grip):.2f} "
                  f"(P(up): no q0 {no_q0.mean():.2f}, no grip {no_grip.mean():.2f})", flush=True)
        finally:
            globals().update(old)
            _timing()


def gen(trials=256, seed=0, chunk=256):
    rng = np.random.default_rng(seed)
    u = command()
    t_obs = np.arange(N_OBS + 1) * OBS_EVERY * DT
    u_obs = u[:: OBS_EVERY]
    X, TW, G, C0 = [], [], [], []
    tic = time.perf_counter()
    for start in range(0, trials, chunk):
        n = min(chunk, trials - start)
        sim = Sim(n)
        sag = sim.sag()
        x0, tw0, err, c0 = sample(rng, n, sag)
        grip = u[None] + err
        x, tw = sim.run(x0, tw0, grip)
        X.append(x), TW.append(tw), G.append(grip[:, ::OBS_EVERY]), C0.append(c0)
        print(f"trials {start}-{start + n}: {time.perf_counter() - tic:.0f} s", flush=True)
    X, TW, G, C0 = (np.concatenate(a) for a in (X, TW, G, C0))
    z = mid_rel(X)
    up_buckle, up_final = z[:, K_BUCKLE] > 0, z[:, -1] > 0

    # observation noise on everything the learner sees, free and fixed DOFs alike
    obs_rng = np.random.default_rng(seed + 1000)
    x_obs = X + obs_rng.normal(0.0, OBS_NOISE_X, X.shape)
    g_obs = G + obs_rng.normal(size=G.shape) * np.r_[[OBS_NOISE_GRIP_T] * 3, [OBS_NOISE_GRIP_R] * 3]

    fixed_nodes = np.zeros(NODES, bool)
    fixed_nodes[[0, 1, -2, -1]] = True
    np.savez_compressed(
        OUT,
        # what the learner gets
        t=t_obs, dt=DT, obs_every=OBS_EVERY,
        command=u_obs,  # (T, 6) nominal gripper pose, identical for every trial
        x_obs=x_obs.astype(np.float32),  # (B, T, NODES, 3) noisy node positions, free and fixed
        grip_obs=g_obs.astype(np.float32),  # (B, T, 6) noisy gripper pose readout
        fixed_nodes=fixed_nodes, gripper_nodes=np.array([NODES - 2, NODES - 1]), clamp_nodes=np.array([0, 1]),
        x_rest=sim.x_rest[0],  # (NODES, 3) straight rest geometry (same for all trials)
        # hidden truth (for evaluation only)
        x=X.astype(np.float32),  # (B, T, NODES, 3) clean node positions
        twist=TW.astype(np.float32),  # (B, T, SEGMENTS) clean edge twists
        grip=G,  # (B, T, 6) actual gripper pose (command + tracking error)
        x_sag=sag[0][0], q0_coef=C0,  # nominal initial state and each trial's (Q0_MODES, [y, z]) perturbation
        up_buckle=up_buckle, up_final=up_final,
        params=np.array([LENGTH, SEGMENTS, RADIUS, BEND, BEND_STRONG_RATIO, DAMPING, STRETCH, TWIST, DENSITY_SCALE,
                         GRAVITY]),
        param_names=np.array(["length", "segments", "radius", "bend", "bend_strong_ratio", "bend_damping",
                              "stretch", "twist", "density_scale", "gravity"]),
        noise=np.array([S_GRIP_T, S_GRIP_R, TAU_GRIP, T_RAMP, S_Q0, Q0_MODES, OBS_NOISE_X, OBS_NOISE_GRIP_T,
                        OBS_NOISE_GRIP_R]),
        noise_names=np.array(["grip_t", "grip_r", "grip_tau", "grip_ramp", "q0", "q0_modes", "obs_x", "obs_grip_t",
                              "obs_grip_r"]),
    )
    print(f"saved {OUT.name}: {trials} trials x {N_OBS + 1} frames; up at buckle {up_buckle.mean():.3f}, "
          f"up at end {up_final.mean():.3f}, switched {np.mean(up_buckle != up_final):.3f}")
    plot()


def plot():
    import matplotlib.pyplot as plt

    d = np.load(OUT)
    t, X, Xo, G, Go, u = d["t"], d["x"], d["x_obs"], d["grip"], d["grip_obs"], d["command"]
    up = d["up_buckle"]
    z, zo = mid_rel(X) * 1e3, mid_rel(Xo) * 1e3
    B = len(X)
    blue, red = "#2a6fdb", "#d1453b"
    col = np.where(up, blue, red)
    tb = t[K_BUCKLE]
    z_clamp = d["x_rest"][0, 2]

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    a = ax[0, 0]
    for b in range(B):
        a.plot(t, z[b], color=col[b], alpha=0.12, lw=0.7)
    for v in (T_COMPRESS, T_COMPRESS + T_SHEAR):
        a.axvline(v, color="0.6", lw=0.8, ls="--")
    a.set(xlabel="t [s]", ylabel="mid-span z − chord [mm]",
          title=f"clean: {B} trials, up {up.mean():.0%} (blue) / down {1 - up.mean():.0%} (red) at t={tb:.2f}s")

    a = ax[0, 1]
    for b in range(min(B, 64)):
        a.plot(t, zo[b], color=col[b], alpha=0.25, lw=0.6)
    a.set(xlabel="t [s]", ylabel="mid-span z − chord [mm]", title="observed (noisy), first 64 trials")

    a = ax[0, 2]
    for k, ls in ((K_BUCKLE, "-"), (len(t) - 1, ":")):
        for b in range(min(B, 48)):
            a.plot(X[b, k, :, 0], (X[b, k, :, 2] - z_clamp) * 1e3, color=col[b], alpha=0.3, lw=0.8, ls=ls)
    a.plot(d["x_sag"][:, 0], (d["x_sag"][:, 2] - z_clamp) * 1e3, "k", lw=1.5, label="nominal initial (gravity sag)")
    a.plot([], [], "0.4", ls="-", label=f"t={tb:.2f}s (end compress)")
    a.plot([], [], "0.4", ls=":", label=f"t={t[-1]:.2f}s (end)")
    a.set(xlabel="x [m]", ylabel="z - clamp height [mm]", title="rod shapes (x–z), 48 trials")
    a.legend(fontsize=7)

    a = ax[1, 0]
    for j, (name, c) in enumerate((("x", "tab:purple"), ("z", "tab:green"))):
        jj = 0 if name == "x" else 2
        for b in range(min(B, 64)):
            a.plot(t, (G[b, :, jj]) * 1e3, color=c, alpha=0.15, lw=0.6)
        a.plot(t, u[:, jj] * 1e3, color="k", lw=1.5, ls="--" if j else "-")
    a.plot([], [], "k", label="command x (same for all)")
    a.plot([], [], "k--", label="command z")
    a.plot([], [], color="tab:purple", label="actual x")
    a.plot([], [], color="tab:green", label="actual z")
    a.set(xlabel="t [s]", ylabel="gripper translation [mm]", title="gripper: one command, noisy actual motion")
    a.legend(fontsize=7)

    a = ax[1, 1]
    names = ["x [mm]", "y [mm]", "z [mm]", "roll [°]", "pitch [°]", "yaw [°]"]
    sc = np.r_[[1e3] * 3, [np.degrees(1.0)] * 3]
    err = (G - u[None]) * sc
    for j, c in zip(range(6), plt.cm.tab10.colors):
        a.plot(t, err[:, :, j].std(0), color=c, label=f"actual − command, std {names[j]}")
        a.plot(t, (Go - G)[:, :, j].std(0) * sc[j], color=c, ls=":", lw=0.8)
    a.plot([], [], "0.4", ls=":", label="readout noise std")
    a.set(xlabel="t [s]", ylabel="std across trials", title="gripper tracking error and readout noise")
    a.legend(fontsize=6)

    a = ax[1, 2]
    kf = np.argmin(np.abs(t - T_FORK))
    pitch = np.degrees(G[:, kf, 4] - u[kf, 4])
    a.scatter(pitch, d["q0_coef"][:, 0, 1] * 1e3, c=col, s=10, alpha=0.7)
    a.set(xlabel=f"gripper pitch error at the fork, t={t[kf]:.2f}s [°]", ylabel="initial z mode-1 coef [mm]",
          title="what picks the branch (colour = branch)")
    fig.tight_layout()
    fig.savefig(HERE / "baseline.png", dpi=110)
    print("plot: baseline.png")


def check():
    """Sanity checks on baseline.npz: shapes, one command, the noise levels, the fixed DOFs, both branches."""
    d = np.load(OUT)
    x, xo, g, go, u, t = d["x"], d["x_obs"], d["grip"], d["grip_obs"], d["command"], d["t"]
    B, T = x.shape[:2]
    print({k: d[k].shape for k in d.files})
    assert u.shape == (T, 6) and g.shape == (B, T, 6) and xo.shape == (B, T, NODES, 3)
    assert np.allclose(u, command()[::OBS_EVERY])
    print(f"B {B}, T {T} frames, dt_obs {t[1] - t[0]:.3f} s, t_end {t[-1]:.2f} s")
    e = g - u[None]
    print(f"tracking error std (after ramp) t [mm] {e[:, T // 4:, :3].std((0, 1)) * 1e3}, "
          f"r [deg] {np.degrees(e[:, T // 4:, 3:].std((0, 1)))}; at t=0: {np.abs(e[:, 0]).max():.1e}")
    print(f"obs noise std: nodes {(xo - x).std() * 1e3:.3f} mm (free {(xo - x)[:, :, ~d['fixed_nodes']].std() * 1e3:.3f}, "
          f"fixed {(xo - x)[:, :, d['fixed_nodes']].std() * 1e3:.3f}); grip t {(go - g)[..., :3].std() * 1e3:.3f} mm, "
          f"r {np.degrees((go - g)[..., 3:].std()):.3f} deg")
    print(f"left clamp drift {np.abs(x[:, :, :2] - d['x_rest'][None, None, :2]).max():.1e} m; "
          f"right clamp node vs commanded+error pose: tip x {np.abs(x[:, :, -1, 0] - d['x_rest'][-1, 0] - g[:, :, 0]).max():.1e} m, "
          f"z {np.abs(x[:, :, -1, 2] - d['x_rest'][-1, 2] - g[:, :, 2]).max():.1e} m (rotation adds ~mm)")
    dq0 = x[:, 0] - d["x_sag"][None]
    print(f"initial-state spread (free nodes, rms over trials) {np.sqrt((dq0 ** 2).mean(0)).max() * 1e3:.3f} mm; "
          f"initial velocity not stored (zero)")
    z = mid_rel(x)
    up = d["up_buckle"]
    print(f"P(up) at buckle {up.mean():.3f}, at end {d['up_final'].mean():.3f}; |mid| at end: up "
          f"{z[up, -1].mean() * 1e3:.1f} mm, down {z[~up, -1].mean() * 1e3:.1f} mm; min |mid| at end "
          f"{np.abs(z[:, -1]).min() * 1e3:.1f} mm")
    kf = np.argmin(np.abs(t - T_FORK))
    sgn = np.where(up, 1.0, -1.0)
    print(f"corr(branch, pitch error at fork) {np.corrcoef(e[:, kf, 4], sgn)[0, 1]:+.2f}; "
          f"corr(branch, initial z mode 1) {np.corrcoef(d['q0_coef'][:, 0, 1], sgn)[0, 1]:+.2f}")
    print(f"finite: {np.isfinite(x).all() and np.isfinite(xo).all()}; size {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if mode == "gen":
        gen(int(sys.argv[2]) if len(sys.argv) > 2 else 256)
    else:
        {"scan": scan, "plot": plot, "check": check}[mode]()
