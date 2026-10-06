"""Identify a rod's bending and stretching stiffness, density and damping from noisy recordings of it buckling.

Shows: system identification by gradients through the solver, on data with process and measurement noise.
A clamped rod is compressed past buckling and sheared by a gripper that tracks its command badly; 256 trials
each get their own gripper error and initial shake, so they buckle up or down. Only noisy node positions and a
noisy gripper readout are recorded. A single rollout from rest cannot follow the data across that fork, so the
fit cuts every recording into short windows that restart from the data (multiple shooting): each window stays
on the branch the data took. All windows of all trials are rods of one batched model; one ``tape.backward`` per
batch gives the gradient of all unknowns, and L-BFGS fits them from a guess off by 3-4x. The viewer replays
every L-BFGS iterate from the first recorded frame (green) over the true motion (grey, without the measurement
noise the fit sees), for one trial on each branch, beside the iterates' parameters.

The dataset, the fit and the replays are cached in ``.cache/examples`` (``--fresh`` recomputes them):

    uv run examples/fit_buckling.py
    uv run examples/fit_buckling.py --unknowns EI,EA     # the others held at their true values
    uv run examples/fit_buckling.py --viewer null --test
"""

import time

import newton
import newton.examples
import numpy as np
import warp as wp
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from utils.common import cached, inset, inset_scale

from dismech_newton import DiSMechSolver, add_rod, flatten_state, suspended_tape
from dismech_newton.solver import advance_frames_kernel
from dismech_newton.triplet import advance_ref_twist_kernel

# -- the rod and the truth -----------------------------------------------------------------
LENGTH, SEGMENTS, RADIUS = 0.5, 24, 0.01
NODES = SEGMENTS + 1
NT = SEGMENTS - 1  # triplets per rod
# the unknowns: weak-plane bend stiffness, stretch stiffness, density (x the default), weak-plane bend damping
TRUE = dict(EI=10.0, EA=1.0e6, rho=0.1, c=0.3)
KINDS = list(TRUE)
STRONG, TWIST, GRAVITY = 4.0, 10.0, 9.81  # known: strong-plane bend = STRONG EI, twist stiffness, gravity
COLUMNS = dict(EI=[2], EA=[0, 1], c=[7])  # triplet_params columns of each unknown (rho scales the masses)
GUESS = dict(EI=0.25, EA=0.3, rho=3.0, c=3.0)  # x the truth

# -- the experiment: compress, shear, hold ----------------------------------------------------
DT, OBS_EVERY = 5.0e-3, 5  # recorded at 40 Hz
T_COMPRESS, T_SHEAR, T_HOLD = 3.0, 3.0, 2.0
STEPS = int(round((T_COMPRESS + T_SHEAR + T_HOLD) / DT))
FRAMES = STEPS // OBS_EVERY  # recorded frames after the first
K_BUCKLE = int(round(T_COMPRESS / DT)) // OBS_EVERY - 1  # the frame at the end of the compression
COMPRESS, SHEAR = 4.0e-2, 3.0e-2  # gripper travel along -x and +z [m]
TRIALS, SEED = 256, 0

# -- noise: the gripper's tracking error and the initial shake (process), the recording (measurement) --
GRIP_T, GRIP_R, GRIP_TAU, GRIP_RAMP = 2.0e-4, np.radians(0.6), 0.15, 0.2  # [m], [rad], [s], [s]
SHAKE, SHAKE_MODES = 3.0e-4, 4  # initial sine modes in y and z [m] (mode k: SHAKE / k)
OBS_X, OBS_GRIP_T, OBS_GRIP_R = 5.0e-4, 5.0e-4, np.radians(0.2)  # node positions, gripper readout

# -- the fit ----------------------------------------------------------------------------------
W = 4  # recorded frames per window (0.1 s)
GROUP = 10  # windows per trial in one batched model
SMOOTH = 15  # the gripper readout is smoothed in time (a cubic fit over this many frames) before it drives the fit
VELOCITY = 7  # a window starts with the recorded positions' velocity, a quadratic fit over this many frames (0: at rest)
SHOW = 1  # trials shown on each branch


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mid_z(x: np.ndarray) -> np.ndarray:
    """Mid-span height above the chord of the end nodes: the buckle's sign tells the branch."""
    return x[..., NODES // 2, 2] - 0.5 * (x[..., 0, 2] + x[..., -1, 2])


# -- the model ------------------------------------------------------------------------------------


def build(rods: int, ratios: np.ndarray | None = None):
    """``rods`` rods in one model, both ends clamped; ``ratios`` (rods, 4): each rod's (EI, EA, rho, c) over the truth."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
    for r in range(rods):
        rod = newton.Rod.create_straight((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), LENGTH, segment_count=SEGMENTS,
                                         radius=RADIUS)
        add_rod(builder, rod, stretch_stiffness=TRUE["EA"], bend_stiffness=TRUE["EI"], twist_stiffness=TWIST,
                bend_damping=TRUE["c"], proxies=False)
        DiSMechSolver.fix_segment(builder, edge=r * SEGMENTS)  # the clamp
        DiSMechSolver.fix_segment(builder, edge=r * SEGMENTS + SEGMENTS - 1)  # the gripper
    model = builder.finalize()
    d = model.dismech
    p = d.triplet_params.numpy().reshape(rods, NT, 10)
    p[..., 3] = STRONG * TRUE["EI"]
    rho = np.full(rods, TRUE["rho"])
    if ratios is not None:
        for kind, r in zip(KINDS, ratios.T):
            if kind == "rho":
                rho *= r
            else:
                p[..., COLUMNS[kind]] *= r[:, None, None]
    d.triplet_params.assign(p.reshape(-1, 10))
    model.particle_mass.assign(model.particle_mass.numpy() * np.repeat(rho, NODES))
    d.edge_inertia.assign(d.edge_inertia.numpy() * np.repeat(rho, SEGMENTS))
    solver = DiSMechSolver(model, theta=1.0)
    solver.refresh_mass()
    return model, solver


STATE = ("q", "qd", "edge_d1_q", "triplet_ref_twist_q", "triplet_strain_q")


def start(solver, st, src, q=None, qd=None):
    """``st`` = ``src`` moved to the DOFs ``q``, with velocities ``qd`` (else at rest). The edge frames, reference
    twists and stored strains are state too: they are carried over from ``src`` as at the end of a step, or the rod
    is kicked."""
    for k in STATE:
        getattr(st.dismech, k).assign(getattr(src.dismech, k))
    if q is not None:
        st.dismech.q.assign(q)
    if qd is None:
        st.dismech.qd.zero_()
    else:
        st.dismech.qd.assign(qd)
    d, tr = solver.der, solver.triplets
    wp.launch(advance_frames_kernel, dim=d.edge_length.shape[0],
              inputs=[src.particle_q, st.particle_q, d.edge_node0, d.edge_node1, src.dismech.edge_d1_q],
              outputs=[st.dismech.edge_d1_q])
    wp.launch(advance_ref_twist_kernel, dim=tr.count,
              inputs=[st.dismech.q, st.dismech.edge_d1_q, tr.conn, src.dismech.triplet_ref_twist_q],
              outputs=[st.dismech.triplet_ref_twist_q])
    tr.measure(st, st.dismech.triplet_strain_q)


def rest_state(model, solver, requires_grad=False):
    st = model.state(requires_grad=requires_grad)
    flatten_state(st)
    solver.triplets.measure(st, st.dismech.triplet_strain_q)
    return st


def fixed_dofs(rods: int) -> np.ndarray:
    """Per rod, the clamped DOFs: nodes 0, 1, N-2, N-1 and the first and last edges' twists (14)."""
    nodes = [0, 1, NODES - 2, NODES - 1]
    twist = 3 * NODES * rods
    return np.array([[3 * (r * NODES + n) + c for n in nodes for c in range(3)]
                     + [twist + r * SEGMENTS, twist + r * SEGMENTS + SEGMENTS - 1] for r in range(rods)]).ravel()


def drive(x_rest: np.ndarray, t: np.ndarray, poses: np.ndarray, t_step: np.ndarray) -> np.ndarray:
    """The clamped DOFs' values before every step (steps, rods * 14) from gripper poses (rods, len(t), 6) =
    (translation, rotation vector about the gripped edge's centre), interpolated to ``t_step``; the clamp at rest."""
    rods = len(poses)
    P = np.stack([np.stack([np.interp(t_step, t, poses[r, :, j]) for j in range(6)], -1) for r in range(rods)])
    a, b = x_rest[-2], x_rest[-1]
    c = 0.5 * (a + b)
    R = Rotation.from_rotvec(P[..., 3:].reshape(-1, 3))
    vals = np.zeros((len(t_step), rods, 14))
    vals[..., 0:3], vals[..., 3:6] = x_rest[0], x_rest[1]
    vals[..., 6:9] = (R.apply(a - c).reshape(rods, -1, 3) + c + P[..., :3]).transpose(1, 0, 2)
    vals[..., 9:12] = (R.apply(b - c).reshape(rods, -1, 3) + c + P[..., :3]).transpose(1, 0, 2)
    vals[..., 13] = P[..., 3].T  # the gripped edge's twist: its roll
    return vals.reshape(len(t_step), -1)


@wp.kernel
def set_fixed(idx: wp.array[wp.int32], vals: wp.array2d[float], step: int, q: wp.array[float]):
    j = wp.tid()
    q[idx[j]] = vals[step, j]


@wp.kernel
def frame_loss(x: wp.array[wp.vec3], obs: wp.array[wp.vec3], free: wp.array[wp.int32], scale: float,
               loss: wp.array[float]):
    i = wp.tid()
    if free[i] != 0:
        wp.atomic_add(loss, i // NODES, scale * wp.length_sq(x[i] - obs[i]))


def rollout(model, solver, st0, fixed, vals):
    """From ``st0`` (its clamped DOFs set here), ``vals[i]`` on the clamped DOFs before step ``i``: the node
    positions (rods, frames, NODES, 3) at every recorded frame."""
    a, b = st0, model.state()
    flatten_state(b)
    idx, vals = wp.array(fixed, dtype=wp.int32), wp.array(vals.astype(np.float32), dtype=float)
    rec = []
    for i in range(STEPS + 1):
        wp.launch(set_fixed, dim=len(idx), inputs=[idx, vals, i], outputs=[a.dismech.q])
        if i % OBS_EVERY == 0:
            rec.append(a.particle_q.numpy().reshape(-1, NODES, 3))
        if i < STEPS:
            solver.step(a, b, None, None, DT)
            a, b = b, a
    return np.stack(rec, 1)


# -- the recordings ---------------------------------------------------------------------------------


def command() -> np.ndarray:
    """The nominal gripper pose before every step (STEPS + 1, 6), the same for every trial."""
    t = np.arange(STEPS + 1) * DT
    u = np.zeros((STEPS + 1, 6))
    u[:, 0] = -COMPRESS * smooth(t / T_COMPRESS)
    u[:, 2] = SHEAR * smooth((t - T_COMPRESS) / T_SHEAR)
    return u


def tracking_error(rng, trials: int) -> np.ndarray:
    """Each trial's smooth gripper error (trials, STEPS + 1, 6): low-pass Gaussian noise, ramped in from zero."""
    sig = GRIP_TAU / DT
    half = int(4 * sig)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2)
    k /= np.sqrt((k * k).sum())  # unit variance
    w = rng.normal(size=(trials, STEPS + 1 + 2 * half, 6))
    e = np.stack([np.stack([np.convolve(w[b, :, j], k, mode="valid") for j in range(6)], -1) for b in range(trials)])
    return e * np.r_[[GRIP_T] * 3, [GRIP_R] * 3] * smooth(np.arange(STEPS + 1) * DT / GRIP_RAMP)[None, :, None]


def shake(rng, trials: int) -> np.ndarray:
    """Each trial's initial offsets of the free nodes (trials, NODES, 3): sine modes in y and z."""
    s = np.linspace(0.0, 1.0, NODES)
    modes = np.array([np.sin(k * np.pi * s) for k in range(1, SHAKE_MODES + 1)])
    c = rng.normal(size=(trials, SHAKE_MODES, 2)) * (SHAKE / np.arange(1, SHAKE_MODES + 1))[None, :, None]
    d = np.zeros((trials, NODES, 3))
    d[..., 1:] = np.einsum("bmc,mn->bnc", c, modes)
    d[:, [0, 1, -2, -1]] = 0.0
    return d


def record() -> dict:
    """Run the experiment: every trial from the gravity sag plus its shake, driven by its actual gripper pose.
    Returns what is recorded (noisy) and, for checks only, the clean positions."""
    trials, seed = TRIALS, SEED
    model, solver = build(trials)
    rng = np.random.default_rng(seed)
    rest = rest_state(model, solver)
    x_rest = rest.particle_q.numpy()[:NODES].astype(np.float64)
    a, b = model.state(), model.state()
    flatten_state(a), flatten_state(b)
    start(solver, a, rest)
    for _ in range(40):  # the static sag: big implicit steps from rest
        a.dismech.qd.zero_()
        solver.step(a, b, None, None, 0.5)
        a, b = b, a
    q = a.dismech.q.numpy()
    q[: 3 * NODES * trials] += shake(rng, trials).ravel()
    start(solver, b, a, q)
    t_step = np.arange(STEPS + 1) * DT
    grip = command()[None] + tracking_error(rng, trials)
    x = rollout(model, solver, b, fixed_dofs(trials), drive(x_rest, t_step, grip, t_step))
    noise = np.random.default_rng(seed + 1000)
    x_obs = x + noise.normal(0.0, OBS_X, x.shape)
    g_obs = grip[:, ::OBS_EVERY] + noise.normal(size=(trials, FRAMES + 1, 6)) * np.r_[[OBS_GRIP_T] * 3,
                                                                                         [OBS_GRIP_R] * 3]
    return dict(x_obs=x_obs.astype(np.float32), grip_obs=g_obs.astype(np.float32), x_rest=x_rest,
                x=x.astype(np.float32))


def drive_readout(data: dict, trials) -> np.ndarray:
    """The clamped DOFs before every step from the recorded gripper readout, smoothed in time."""
    t = np.arange(FRAMES + 1) * OBS_EVERY * DT
    poses = savgol_filter(data["grip_obs"][trials].astype(np.float64), SMOOTH, 3, axis=1)
    return drive(data["x_rest"], t, poses, np.arange(STEPS + 1) * DT)


# -- the fit --------------------------------------------------------------------------------------


class WindowFit:
    """The mean loss over the trials and its gradient in the log of the unknowns over their truth.

    Every trial is cut into windows of ``W`` recorded frames. A window starts from the recorded node positions at
    its first frame, moving with their smoothed velocity (the first window: the straight rest pose, at rest), its
    clamped DOFs follow the smoothed readout, and its loss is the squared error of the free nodes against the recording at its other
    frames (mm^2, the mean over frames and nodes). ``GROUP`` windows of every trial are one batched model.
    """

    def __init__(self, data: dict, unknowns: list[str]):
        trials = len(data["x_obs"])
        self.unknowns, self.trials = unknowns, trials
        windows = FRAMES // W
        self.G = max(g for g in range(1, GROUP + 1) if windows % g == 0)
        self.chunks = windows // self.G
        R = trials * self.G  # rod g * trials + b: window g of the batch, trial b
        self.L = W * OBS_EVERY  # steps per window
        self.model, self.solver = m, solver = build(R)
        d = m.dismech
        self.p_true = d.triplet_params.numpy().reshape(R, NT, 10).astype(np.float64)
        self.m_true = m.particle_mass.numpy().astype(np.float64)
        self.i_true = d.edge_inertia.numpy().astype(np.float64)
        for a in (d.triplet_params, m.particle_mass, d.edge_inertia):
            a.requires_grad = True
        self.states = [m.state(requires_grad=True) for _ in range(self.L + 1)]
        for s in self.states:
            flatten_state(s)
        self.rest = rest_state(m, solver)
        q_rest = self.rest.dismech.q.numpy()
        fixed = fixed_dofs(R)
        self.fixed = wp.array(fixed, dtype=wp.int32)
        free = np.ones((R, NODES), np.int32)
        free[:, [0, 1, -2, -1]] = 0
        self.free = wp.array(free.ravel(), dtype=wp.int32)
        self.scale = 1.0e6 / (FRAMES * (NODES - 4))
        self.loss = wp.zeros(R, dtype=float, requires_grad=True)
        self.weights = wp.array(np.full(R, 1.0 / trials, np.float32), dtype=float)

        vals = drive_readout(data, slice(None)).reshape(STEPS + 1, trials, 14)
        x_obs = data["x_obs"]
        v_obs = np.zeros_like(x_obs)  # free nodes only
        if VELOCITY:
            v_obs[:, :, 2:-2] = savgol_filter(x_obs[:, :, 2:-2], VELOCITY, 2, deriv=1, delta=OBS_EVERY * DT, axis=1)
        self.vals, self.obs, self.q0, self.qd0 = [], [], [], []
        for c in range(self.chunks):
            k0 = np.arange(c * self.G, (c + 1) * self.G) * W  # first frame of every window in the batch
            steps = k0[:, None] * OBS_EVERY + np.arange(self.L + 1)[None]  # (G, L + 1)
            self.vals.append(wp.array(vals[steps.T].reshape(self.L + 1, -1).astype(np.float32), dtype=float))
            o = x_obs[:, k0[:, None] + np.arange(1, W + 1)[None]]  # (trials, G, W, NODES, 3)
            self.obs.append(wp.array(o.transpose(2, 1, 0, 3, 4).reshape(W, R * NODES, 3), dtype=wp.vec3))
            x0 = np.where((k0 > 0)[:, None, None, None], x_obs[:, k0].transpose(1, 0, 2, 3), data["x_rest"])
            q = q_rest.copy()
            q[: 3 * NODES * R] = x0.ravel()
            q[fixed] = vals[steps[:, 0]].ravel()
            self.q0.append(wp.array(q.astype(np.float32), dtype=float))
            qd = np.zeros_like(q)
            qd[: 3 * NODES * R] = (v_obs[:, k0].transpose(1, 0, 2, 3) * (k0 > 0)[:, None, None, None]).ravel()
            self.qd0.append(wp.array(qd.astype(np.float32), dtype=float))

    def set(self, theta: np.ndarray):
        m, d = self.model, self.model.dismech
        p, rho = self.p_true.copy(), 1.0
        for kind, th in zip(self.unknowns, np.exp(theta)):
            if kind == "rho":
                rho = th
            else:
                p[..., COLUMNS[kind]] *= th
        self.p, self.rho = p, rho
        d.triplet_params.assign(p.reshape(-1, 10).astype(np.float32))
        m.particle_mass.assign((self.m_true * rho).astype(np.float32))
        d.edge_inertia.assign((self.i_true * rho).astype(np.float32))
        self.solver.refresh_mass()

    def run(self, c: int):
        with suspended_tape():  # the start and the drive are data: no adjoint through them
            self.loss.zero_()
            start(self.solver, self.states[0], self.rest, self.q0[c], self.qd0[c])
        for i in range(self.L):
            with suspended_tape():
                wp.launch(set_fixed, dim=len(self.fixed), inputs=[self.fixed, self.vals[c], i],
                          outputs=[self.states[i].dismech.q])
            self.solver.step(self.states[i], self.states[i + 1], None, None, DT)
            if (i + 1) % OBS_EVERY == 0:
                j = (i + 1) // OBS_EVERY
                with suspended_tape():  # the recorded clamped DOFs are the pose at that instant
                    wp.launch(set_fixed, dim=len(self.fixed), inputs=[self.fixed, self.vals[c], i + 1],
                              outputs=[self.states[i + 1].dismech.q])
                wp.launch(frame_loss, dim=self.model.particle_count,
                          inputs=[self.states[i + 1].particle_q, self.obs[c][j - 1], self.free, self.scale],
                          outputs=[self.loss])

    def __call__(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Mean loss and its gradient in ``theta`` (one backward pass per batch)."""
        self.set(theta)
        m, d = self.model, self.model.dismech
        gp, gm, gi, loss = 0.0, 0.0, 0.0, 0.0
        for c in range(self.chunks):
            for a in (d.triplet_params, m.particle_mass, d.edge_inertia):
                a.grad.zero_()
            tape = wp.Tape()
            with tape:
                self.run(c)
            tape.backward(grads={self.loss: self.weights})
            gp = gp + d.triplet_params.grad.numpy().reshape(self.p.shape)
            gm = gm + m.particle_mass.grad.numpy()
            gi = gi + d.edge_inertia.grad.numpy()
            loss += self.loss.numpy().sum() / self.trials
            tape.zero()
            tape.reset()
        g = [self.rho * (gm @ self.m_true + gi @ self.i_true) if kind == "rho"
             else np.sum(gp[..., COLUMNS[kind]] * self.p[..., COLUMNS[kind]]) for kind in self.unknowns]
        return float(loss), np.array(g)


def fit(data: dict, unknowns: list[str]) -> dict:
    """L-BFGS on the unknowns' logs over their truth, from the guess; every accepted iterate."""
    f = WindowFit(data, unknowns)
    loss_true = f(np.zeros(len(unknowns)))[0]
    best, evals = {}, [0]

    def fun(theta):
        loss, g = f(theta)
        evals[0] += 1
        if np.isfinite(loss) and np.all(np.isfinite(g)):
            if not best or loss < best["loss"]:
                best.update(loss=loss, theta=theta.copy())
            return loss, g
        # a few windows blew up (low damping and a noisy start): a smooth wall back to the best point so far,
        # so the line search backtracks instead of stopping
        dx = theta - best["theta"]
        return best["loss"] + 1.0 + 100.0 * dx @ dx, 200.0 * dx

    theta0 = np.log([GUESS[k] for k in unknowns])
    iterates, losses = [theta0], [fun(theta0)[0]]

    def accepted(intermediate_result):  # scipy passes the OptimizeResult to a parameter of this name
        r = intermediate_result
        iterates.append(r.x.copy())
        losses.append(r.fun)
        print(f"  iterate {len(iterates) - 1}: loss {r.fun:.4f}  "
              + "  ".join(f"{k} {v:.3f}" for k, v in zip(unknowns, np.exp(r.x))), flush=True)

    tic = time.perf_counter()
    minimize(fun, theta0, jac=True, method="L-BFGS-B", bounds=[(-np.log(30.0), np.log(30.0))] * len(unknowns),
             callback=accepted, options=dict(maxiter=200, ftol=1e-12, gtol=1e-9))
    print(f"fit: {evals[0]} gradient evaluations in {time.perf_counter() - tic:.0f} s; loss {losses[-1]:.4f} "
          f"(at the truth {loss_true:.4f})")
    return dict(theta=np.array(iterates), loss=np.array(losses), loss_true=loss_true)


def replay(data: dict, trials: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    """Every ``ratios`` row (parameters over the truth) on every trial: the whole experiment from the first
    recorded frame, driven by the smoothed readout. Node positions (len(ratios), len(trials), frames, NODES, 3)."""
    n, R = len(ratios), len(ratios) * len(trials)
    model, solver = build(R, np.repeat(ratios, len(trials), 0))
    rest = rest_state(model, solver)
    st = model.state()
    flatten_state(st)
    q = rest.dismech.q.numpy()
    vals = np.tile(drive_readout(data, trials).reshape(STEPS + 1, 1, -1), (1, n, 1)).reshape(STEPS + 1, -1)
    fixed = fixed_dofs(R)
    q[: 3 * NODES * R] = np.tile(data["x_obs"][trials, 0], (n, 1, 1)).ravel()
    q[fixed] = vals[0]
    start(solver, st, rest, q)
    x = rollout(model, solver, st, fixed, vals)
    return x.reshape(n, len(trials), FRAMES + 1, NODES, 3).astype(np.float32)


def shown_trials(data: dict, candidates: int = 32) -> np.ndarray:
    """``SHOW`` trials on each branch (up, down) that the truth, replayed from their first frame, follows."""
    trials = np.arange(candidates)
    x = replay(data, trials, np.ones((1, 4)))[0]
    up = mid_z(data["x"][trials, -1]) > 0
    follows = (np.sign(mid_z(x[:, [K_BUCKLE, -1]])) == np.sign(mid_z(data["x"][trials][:, [K_BUCKLE, -1]]))).all(1)
    return np.concatenate([trials[follows & up][:SHOW], trials[follows & ~up][:SHOW]])


def ratios_of(theta: np.ndarray, unknowns: list[str]) -> np.ndarray:
    """(iterates, 4): (EI, EA, rho, c) over the truth, 1 for the known ones."""
    out = np.ones((len(theta), 4))
    for j, kind in enumerate(unknowns):
        out[:, KINDS.index(kind)] = np.exp(theta[:, j])
    return out


def results(unknowns: list[str], fresh: bool = False) -> dict:
    """The recordings, the fit and the replays, each computed once per configuration (``.cache/examples``)."""
    experiment = dict(version=1, true=TRUE, strong=STRONG, twist=TWIST, length=LENGTH, segments=SEGMENTS,
                      radius=RADIUS, dt=DT, obs_every=OBS_EVERY, phases=[T_COMPRESS, T_SHEAR, T_HOLD],
                      travel=[COMPRESS, SHEAR], trials=TRIALS, seed=SEED,
                      noise=[GRIP_T, GRIP_R, GRIP_TAU, GRIP_RAMP, SHAKE, SHAKE_MODES, OBS_X, OBS_GRIP_T, OBS_GRIP_R])
    data = cached("fit_buckling-data", experiment, record, fresh)
    setup = dict(experiment=experiment, unknowns=unknowns, guess=GUESS, window=W, smooth=SMOOTH, velocity=VELOCITY,
                 version=1)
    out = cached("fit_buckling-fit", setup, lambda: fit(data, unknowns), fresh)

    def replays():
        trials = shown_trials(data)
        return dict(trials=trials, x=replay(data, trials, ratios_of(out["theta"], unknowns)))

    shown = cached("fit_buckling-replay", dict(setup=setup, show=SHOW, version=1), replays, fresh)
    return dict(data=data, **out, **shown)


# -- the picture -------------------------------------------------------------------------------

GREY, GREEN = (0.82, 0.82, 0.82), (0.3, 0.8, 0.5)
COLORS = dict(EI="#5aaaff", EA="#4dcc80", rho="#ff8c3c", c="#c08cff")
TEX = dict(EI=r"$EI$", EA=r"$EA$", rho=r"$\rho$", c=r"$c$")


def convergence(ratios: np.ndarray, unknowns: list[str], k: int, size=(400, 400)) -> np.ndarray:
    """Each unknown over its truth at every iterate, up to iterate ``k``."""

    def draw(ax):
        from matplotlib.ticker import MaxNLocator  # noqa: PLC0415

        s = inset_scale(size)
        it = np.arange(len(ratios))
        ax.axhspan(0.9, 1.1, color="white", alpha=0.08, lw=0)
        ax.axhline(1.0, color="white", lw=1.0 * s, ls=":", alpha=0.6)
        for kind in unknowns:
            r = ratios[:, KINDS.index(kind)]
            ax.plot(it, r, color=COLORS[kind], alpha=0.25)
            ax.plot(it[: k + 1], r[: k + 1], color=COLORS[kind], label=f"{TEX[kind]}  {r[k]:.2f}")
            ax.plot(k, r[k], "o", color=COLORS[kind], ms=7 * s, mec="white", mew=1.2 * s)
        ax.set(yscale="log", ylim=(0.15, 5.0), xlim=(0, len(ratios) - 1))
        ax.minorticks_off()
        ax.set_yticks([0.25, 0.5, 1, 2, 4], labels=["$1/4$", "$1/2$", "$1$", "$2$", "$4$"])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(r"L-BFGS step")
        ax.set_ylabel(r"fit / truth")
        ax.legend(loc="upper right", ncol=2)

    return inset(draw, size)


def body_q(x: np.ndarray) -> np.ndarray:
    """Capsule poses (segments, 7) of a polyline of nodes: at the segments' midpoints, local z along them."""
    d = x[1:] - x[:-1]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    q = np.concatenate([np.cross([0.0, 0.0, 1.0], d), 1.0 + d[:, 2:3]], axis=1)  # (x, y, z, w): z onto d
    return np.concatenate([0.5 * (x[1:] + x[:-1]), q / np.linalg.norm(q, axis=1, keepdims=True)], axis=1)


class Example:
    """The true motion (grey) and the iterate's replays (green) of the shown trials, stacked: one on the upper
    branch above one on the lower. Each iterate plays the whole experiment, then the next iterate."""

    pause = 40  # frames held on the last recorded frame
    gap = 0.09  # [m] between the stacked trials

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        unknowns = (args.unknowns if args is not None else "EI,EA,rho,c").split(",")
        fresh = bool(args is not None and args.fresh)
        r = results(unknowns, fresh)
        self.unknowns, self.trials = unknowns, r["trials"]
        self.ratios = ratios_of(r["theta"], unknowns)
        self.true = r["data"]["x"][self.trials].astype(np.float64)  # (trials, frames, NODES, 3): no measurement noise
        self.replays = r["x"].astype(np.float64)  # (iterates, trials, frames, NODES, 3)
        for k, ratio in enumerate(self.ratios):
            print(f"iterate {k:2d}: loss {r['loss'][k]:.4f}  "
                  + "  ".join(f"{u} {ratio[KINDS.index(u)]:.3f}" for u in unknowns))
        print(f"loss at the truth {float(r['loss_true']):.4f}")
        self.lift = np.zeros((len(self.trials), 1, 3))  # stacked downwards
        self.lift[:, 0, 2] = -self.gap * np.arange(len(self.trials))

        builder = newton.ModelBuilder()
        for color in (GREY, GREEN):
            for _ in self.trials:
                rod = newton.Rod.create_straight((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), LENGTH, segment_count=SEGMENTS,
                                                 radius=RADIUS)
                add_rod(builder, rod, bend_stiffness=1.0, color=color)
        self.model = builder.finalize()
        self.state = self.model.state()
        self.k, self.frame, self.sim_time = 0, 0, 0.0
        viewer.set_model(self.model)
        viewer.set_camera(pos=wp.vec3(0.25, -0.75, 0.87), pitch=0.0, yaw=90.0)
        self.viewer.log_image("fit", self.panel())

    @property
    def t(self) -> int:
        return min(self.frame, FRAMES)

    def panel(self, size=(400, 400)) -> np.ndarray:
        return convergence(self.ratios, self.unknowns, self.k, size)

    def step(self):
        self.frame += 1
        if self.frame > FRAMES + self.pause:
            self.frame = 0
            self.k = (self.k + 1) % len(self.ratios)
            self.viewer.log_image("fit", self.panel())
        self.sim_time += OBS_EVERY * DT

    def render(self):
        t = self.t
        toward = np.array([0.0, -0.02, 0.0])  # the iterate just in front of the truth
        x = np.concatenate([self.true[:, t] + self.lift, self.replays[self.k, :, t] + self.lift + toward])
        self.state.body_q.assign(np.concatenate([body_q(rod) for rod in x]).astype(np.float32))
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def test_final(self):
        ratio = self.ratios[-1]
        for u in self.unknowns:
            r = ratio[KINDS.index(u)]
            assert abs(r - 1.0) < 0.2, f"{u}: fit / truth {r:.3f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--unknowns", default="EI,EA,rho,c", help="fitted, of EI, EA, rho, c (comma-separated)")
    parser.add_argument("--fresh", action="store_true", help="recompute the cached recordings, fit and replays")
    parser.set_defaults(num_frames=2000)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
