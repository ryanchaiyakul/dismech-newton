"""Identify a rod's bending and stretching stiffness, density and damping from noisy recordings of it buckling.

Shows: system identification by gradients through the solver, on data with process and measurement noise.
A clamped rod is compressed past buckling and sheared by a gripper that tracks its command badly; 256 trials
each get their own gripper error and initial shake, so they buckle up or down. A camera records noisy node
positions at 30 fps; the gripper reports its noisy pose and, from a force sensor, the force it holds the rod with.
Positions alone cannot tell the parameters' scale: every stiffness, damping and mass x 2 moves the rod the same way.
The force can.

A single rollout from rest cannot follow the data across the buckling fork, so the fit cuts every recording into
short windows that restart from the data (multiple shooting): each window stays on the branch the data took. All
windows of all trials are rods of one batched model; one ``tape.backward`` per batch gives the gradient of all
unknowns, and L-BFGS fits them from a guess off by 3-4x. It fits twice: the force's error is mostly the gripper
pose readout's, which the force feels much more than the positions do, so the second fit weighs the force by what
the first one left unexplained. The fit simulates with ADMM (the solver that also handles contact; 50 iterations
per step), restarted at every window: its warm start from another window would bias the fit. The recordings are
simulated with Newton.

The viewer replays every L-BFGS step from the first recorded frame (green) over the true motion (grey, without the
measurement noise the fit sees), for one trial on each branch, beside the parameters and the upper trial's gripper
force: the sensor's reading (dots) and the replay's (line).

The dataset, the fit and the replays are cached in ``.cache/examples`` (``--fresh`` recomputes them):

    uv run examples/fit_buckling.py
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
from utils.common import cached, capsule_poses, inset, inset_scale, smoothstep

from dismech_newton import ADMMDiSMechSolver, DiSMechSolver, add_rod, flatten_state, suspended_tape
from dismech_newton.solver import advance_frames_kernel
from dismech_newton.strains import strain_gradient, vec5f, vec10f
from dismech_newton.triplet import advance_ref_twist_kernel, geometry_at, linear_energy, vec5i

# -- the rod and the truth -----------------------------------------------------------------
LENGTH, SEGMENTS, RADIUS = 0.5, 24, 0.01
NODES = SEGMENTS + 1
NT = SEGMENTS - 1  # triplets per rod
# the unknowns: bend stiffness, stretch stiffness, density (x the default), bend damping
TRUE = dict(EI=10.0, EA=1.0e6, rho=0.1, c=0.3)
KINDS = list(TRUE)
STRONG, GRAVITY = 4.0, 9.81  # the strong-plane bend stiffness is STRONG EI, the twist stiffness EI; both damp with c
COLUMNS = dict(EI=[2, 3, 4], EA=[0, 1], c=[7, 8, 9])  # triplet_params columns of each unknown (rho scales the masses)
GUESS = dict(EI=0.25, EA=0.3, rho=3.0, c=3.0)  # x the truth

# -- the experiment: compress, shear, hold ----------------------------------------------------
DT, OBS_EVERY = 1.0 / 240.0, 8  # recorded at 30 fps
T_COMPRESS, T_SHEAR, T_HOLD = 3.0, 3.0, 2.0
STEPS = int(round((T_COMPRESS + T_SHEAR + T_HOLD) / DT))
FRAMES = STEPS // OBS_EVERY  # recorded frames after the first
K_BUCKLE = int(round(T_COMPRESS / DT)) // OBS_EVERY - 1  # the frame at the end of the compression
COMPRESS, SHEAR = 4.0e-2, 3.0e-2  # gripper travel along -x and +z [m]
TRIALS, SEED = 256, 0

# -- noise: the gripper's tracking error and the initial shake (process), the recording (measurement) --
GRIP_T, GRIP_R, GRIP_TAU, GRIP_RAMP = 2.0e-4, np.radians(0.6), 0.15, 0.2  # [m], [rad], [s], [s]
SHAKE, SHAKE_MODES = 3.0e-4, 4  # initial sine modes in y and z [m] (mode k: SHAKE / k)
OBS_X, OBS_GRIP_T, OBS_GRIP_R = 5.0e-4, 5.0e-4, np.radians(0.2)  # node positions, gripper pose readout
OBS_F = 0.25  # the force sensor [N]

# -- the fit ----------------------------------------------------------------------------------
W = 3  # recorded frames per window (0.1 s: much shorter windows are mostly their noisy start)
GROUP = 10  # windows per trial in one batched model (divides FRAMES // W)
SMOOTH = 11  # the gripper readout is smoothed in time (a cubic fit over this many frames) before it drives the fit
VELOCITY = 5  # a window starts with the recorded positions' velocity, a quadratic fit over this many frames


def mid_z(x: np.ndarray) -> np.ndarray:
    """Mid-span height above the chord of the end nodes: the buckle's sign tells the branch."""
    return x[..., NODES // 2, 2] - 0.5 * (x[..., 0, 2] + x[..., -1, 2])


# -- the model ------------------------------------------------------------------------------------


def build(rods: int, ratios: np.ndarray | None = None, admm: bool = False):
    """``rods`` rods in one model, both ends clamped; ``ratios`` (rods, 4): each rod's (EI, EA, rho, c) over the truth.
    ``admm``: the ADMM solver at a fixed 50 iterations per step (else Newton)."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
    for r in range(rods):
        rod = newton.Rod.create_straight((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), LENGTH, segment_count=SEGMENTS,
                                         radius=RADIUS)
        add_rod(builder, rod, stretch_stiffness=TRUE["EA"], bend_stiffness=TRUE["EI"], twist_stiffness=TRUE["EI"],
                bend_damping=TRUE["c"], proxies=False)
        DiSMechSolver.fix_segment(builder, edge=r * SEGMENTS)  # the clamp
        DiSMechSolver.fix_segment(builder, edge=r * SEGMENTS + SEGMENTS - 1)  # the gripper
    model = builder.finalize()
    d = model.dismech
    p = d.triplet_params.numpy().reshape(rods, NT, 10)
    p[..., 3] *= STRONG
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
    solver = ADMMDiSMechSolver(model, theta=1.0, iterations=50, tol=0.0) if admm else DiSMechSolver(model, theta=1.0)
    solver.refresh_mass()
    return model, solver


STATE = ("q", "qd", "edge_d1_q", "triplet_ref_twist_q", "triplet_strain_q")


def start(solver, st, src, q=None, qd=None):
    """``st`` = ``src`` moved to the DOFs ``q``, with velocities ``qd`` (else at rest). The edge frames, reference
    twists and stored strains are state too: they are carried over from ``src`` as at the end of a step, or the rod
    is kicked. The solver starts its next step from ``st`` (no warm start)."""
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
    solver.reset(st)


def rest_state(model, solver):
    st = model.state()
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
def copy(src: wp.array[float], dst: wp.array[float]):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def grip_force(q: wp.array[float], edge_d1: wp.array[wp.vec3], ref_twist: wp.array[float], conn: wp.array[vec5i],
               edge_length: wp.array[float], params: wp.array[vec10f], rest: wp.array[vec5f],
               strain_prev: wp.array[vec5f], dt: float, force: wp.array[wp.vec3]):
    """The force sensor: what the triplet ``t`` adds to the force its rod's gripped nodes take from the rest of the
    rod (the gradient of its elastic and viscous stresses)."""
    t = wp.tid()
    c = conn[t]
    geom = geometry_at(q, q, edge_d1, ref_twist[t], c, edge_length, 3 * NODES * (conn.shape[0] // NT))
    sigma, _ = linear_energy(geom.strain, strain_prev[t], rest[t], params[t], dt)
    g = strain_gradient(geom, sigma)
    a = (t // NT) * NODES + NODES - 2  # the first gripped node
    f = (wp.vec3(g[0], g[1], g[2]) * wp.where(c[2] >= a, 1.0, 0.0)
         + wp.vec3(g[4], g[5], g[6]) * wp.where(c[3] >= a, 1.0, 0.0)
         + wp.vec3(g[8], g[9], g[10]) * wp.where(c[4] >= a, 1.0, 0.0))
    wp.atomic_add(force, t // NT, f)


def sense(solver, q, new, old, force):
    """The gripper's force in the state ``new`` (its DOFs ``q``), just stepped from ``old``, added to ``force``."""
    tr = solver.triplets
    wp.launch(grip_force, dim=tr.count,
              inputs=[q, new.dismech.edge_d1_q, new.dismech.triplet_ref_twist_q, tr.conn, solver.der.edge_length,
                      tr.params, tr.rest, old.dismech.triplet_strain_q, DT],
              outputs=[force])


@wp.kernel
def frame_loss(x: wp.array[wp.vec3], obs: wp.array[wp.vec3], free: wp.array[wp.int32], scale: float,
               loss: wp.array[float]):
    i = wp.tid()
    if free[i] != 0:
        wp.atomic_add(loss, i // NODES, scale * wp.length_sq(x[i] - obs[i]))


@wp.kernel
def force_loss(force: wp.array[wp.vec3], obs: wp.array[wp.vec3], scale: float, loss: wp.array[float]):
    r = wp.tid()
    loss[r] += scale * wp.length_sq(force[r] - obs[r])


def rollout(model, solver, st0, fixed, vals):
    """From ``st0`` (its clamped DOFs set here), ``vals[i]`` on the clamped DOFs before step ``i``: at every recorded
    frame the node positions (rods, frames, NODES, 3) and the gripper's force (rods, frames, 3; 0 at the start)."""
    a, b = st0, model.state()
    flatten_state(b)
    idx, vals = wp.array(fixed, dtype=wp.int32), wp.array(vals.astype(np.float32), dtype=float)
    rods = len(fixed) // 14
    f = wp.zeros(rods, dtype=wp.vec3)
    x, force = [], [np.zeros((rods, 3), np.float32)]
    for i in range(STEPS + 1):
        wp.launch(set_fixed, dim=len(idx), inputs=[idx, vals, i], outputs=[a.dismech.q])
        if i % OBS_EVERY == 0:
            x.append(a.particle_q.numpy().reshape(-1, NODES, 3))
        if i < STEPS:
            solver.step(a, b, None, None, DT)
            if (i + 1) % OBS_EVERY == 0:  # read before the next step's drive moves the gripper
                f.zero_()
                sense(solver, b.dismech.q, b, a, f)
                force.append(f.numpy())
            a, b = b, a
    return np.stack(x, 1), np.stack(force, 1)


# -- the recordings ---------------------------------------------------------------------------------


def command() -> np.ndarray:
    """The nominal gripper pose before every step (STEPS + 1, 6), the same for every trial."""
    t = np.arange(STEPS + 1) * DT
    u = np.zeros((STEPS + 1, 6))
    u[:, 0] = -COMPRESS * smoothstep(t, 0.0, T_COMPRESS)
    u[:, 2] = SHEAR * smoothstep(t, T_COMPRESS, T_COMPRESS + T_SHEAR)
    return u


def tracking_error(rng, trials: int) -> np.ndarray:
    """Each trial's smooth gripper error (trials, STEPS + 1, 6): low-pass Gaussian noise, ramped in from zero."""
    sig = GRIP_TAU / DT
    half = int(4 * sig)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2)
    k /= np.sqrt((k * k).sum())  # unit variance
    w = rng.normal(size=(trials, STEPS + 1 + 2 * half, 6))
    e = np.stack([np.stack([np.convolve(w[b, :, j], k, mode="valid") for j in range(6)], -1) for b in range(trials)])
    return e * np.r_[[GRIP_T] * 3, [GRIP_R] * 3] * smoothstep(np.arange(STEPS + 1) * DT, 0.0, GRIP_RAMP)[None, :, None]


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
    Returns what is recorded (noisy) and, for checks only, the clean positions and force."""
    trials = TRIALS
    model, solver = build(trials)
    rng = np.random.default_rng(SEED)
    rest = rest_state(model, solver)
    x_rest = rest.particle_q.numpy()[:NODES].astype(np.float64)
    a, b = model.state(), model.state()
    flatten_state(a)
    flatten_state(b)
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
    x, force = rollout(model, solver, b, fixed_dofs(trials), drive(x_rest, t_step, grip, t_step))
    noise = np.random.default_rng(SEED + 1000)
    x_obs = x + noise.normal(0.0, OBS_X, x.shape)
    g_obs = grip[:, ::OBS_EVERY] + noise.normal(size=(trials, FRAMES + 1, 6)) * np.r_[[OBS_GRIP_T] * 3,
                                                                                         [OBS_GRIP_R] * 3]
    f_obs = force + noise.normal(0.0, OBS_F, force.shape)
    return dict(x_obs=x_obs.astype(np.float32), grip_obs=g_obs.astype(np.float32), f_obs=f_obs.astype(np.float32),
                x_rest=x_rest, x=x.astype(np.float32), force=force)


def drive_readout(data: dict, trials) -> np.ndarray:
    """The clamped DOFs before every step from the recorded gripper readout, smoothed in time."""
    t = np.arange(FRAMES + 1) * OBS_EVERY * DT
    poses = savgol_filter(data["grip_obs"][trials].astype(np.float64), SMOOTH, 3, axis=1)
    return drive(data["x_rest"], t, poses, np.arange(STEPS + 1) * DT)


# -- the fit --------------------------------------------------------------------------------------


class WindowFit:
    """The loss over the trials and its gradient in the log of the unknowns over their truth.

    Every trial is cut into windows of ``W`` recorded frames. A window starts from the recorded node positions at
    its first frame, moving with their smoothed velocity (the first window: the straight rest pose, at rest), its
    clamped DOFs follow the smoothed readout, and at its other frames it is compared with the recording: the free
    nodes' squared error (mm^2, the mean over frames and nodes) plus the gripper force's, weighed as Gaussian noise
    of ``sigma`` against the positions' ``OBS_X``. ``GROUP`` windows of every trial are one batched model.
    """

    def __init__(self, data: dict):
        trials = len(data["x_obs"])
        self.trials = trials
        self.G, self.chunks = GROUP, FRAMES // W // GROUP
        R = trials * GROUP  # rod g * trials + b: window g of the batch, trial b
        self.L = W * OBS_EVERY  # steps per window
        self.model, self.solver = m, solver = build(R, admm=True)
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
        self.loss_x, self.loss_f = (wp.zeros(R, dtype=float, requires_grad=True) for _ in range(2))
        # the sensor, read at every recorded frame of a window: the DOFs it reads (before the drive moves the clamps)
        self.q_read = [wp.zeros_like(self.states[0].dismech.q, requires_grad=True) for _ in range(W)]
        self.force = [wp.zeros(R, dtype=wp.vec3, requires_grad=True) for _ in range(W)]
        self.sigma = OBS_F

        vals = drive_readout(data, slice(None)).reshape(STEPS + 1, trials, 14)
        x_obs, f_obs = data["x_obs"], data["f_obs"]
        v_obs = np.zeros_like(x_obs)  # free nodes only
        v_obs[:, :, 2:-2] = savgol_filter(x_obs[:, :, 2:-2], VELOCITY, 2, deriv=1, delta=OBS_EVERY * DT, axis=1)
        self.vals, self.obs, self.f_obs, self.q0, self.qd0 = [], [], [], [], []
        for c in range(self.chunks):
            k0 = np.arange(c * self.G, (c + 1) * self.G) * W  # first frame of every window in the batch
            steps = k0[:, None] * OBS_EVERY + np.arange(self.L + 1)[None]  # (G, L + 1)
            self.vals.append(wp.array(vals[steps.T].reshape(self.L + 1, -1).astype(np.float32), dtype=float))
            frames = k0[:, None] + np.arange(1, W + 1)[None]  # (G, W)
            o = x_obs[:, frames]  # (trials, G, W, NODES, 3)
            self.obs.append(wp.array(o.transpose(2, 1, 0, 3, 4).reshape(W, R * NODES, 3), dtype=wp.vec3))
            self.f_obs.append(wp.array(f_obs[:, frames].transpose(2, 1, 0, 3).reshape(W, R, 3), dtype=wp.vec3))
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
        p = self.p_true.copy()
        ratio = dict(zip(KINDS, np.exp(theta)))
        for kind, cols in COLUMNS.items():
            p[..., cols] *= ratio[kind]
        self.p, self.rho = p, ratio["rho"]
        d.triplet_params.assign(p.reshape(-1, 10).astype(np.float32))
        m.particle_mass.assign((self.m_true * self.rho).astype(np.float32))
        d.edge_inertia.assign((self.i_true * self.rho).astype(np.float32))
        self.solver.refresh_mass()

    def run(self, c: int):
        with suspended_tape():  # the start and the drive are data: no adjoint through them
            for a in (self.loss_x, self.loss_f, *self.force):
                a.zero_()
            start(self.solver, self.states[0], self.rest, self.q0[c], self.qd0[c])
        for i in range(self.L):
            with suspended_tape():
                wp.launch(set_fixed, dim=len(self.fixed), inputs=[self.fixed, self.vals[c], i],
                          outputs=[self.states[i].dismech.q])
            self.solver.step(self.states[i], self.states[i + 1], None, None, DT)
            if (i + 1) % OBS_EVERY == 0:
                j = (i + 1) // OBS_EVERY - 1
                new = self.states[i + 1]
                # the tape replays a kernel's adjoint on its arrays as they are then, so the sensor reads a copy
                wp.launch(copy, dim=len(self.q_read[j]), inputs=[new.dismech.q], outputs=[self.q_read[j]])
                sense(self.solver, self.q_read[j], new, self.states[i], self.force[j])
                wp.launch(force_loss, dim=len(self.force[j]), inputs=[self.force[j], self.f_obs[c][j], 1.0 / FRAMES],
                          outputs=[self.loss_f])
                wp.launch(frame_loss, dim=self.model.particle_count,
                          inputs=[new.particle_q, self.obs[c][j], self.free, self.scale], outputs=[self.loss_x])

    def weights(self) -> np.ndarray:
        """The loss's weights of the positions' and the force's squared errors."""
        kappa = 1.0e6 * OBS_X ** 2 / (NODES - 4)  # a node's squared error over OBS_X^2 is worth kappa mm^2
        return np.array([1.0, kappa / self.sigma ** 2])

    def __call__(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """The loss and its gradient in ``theta`` (one backward pass per batch); ``self.sq``: the summed squared
        errors of the positions (mm^2) and the force (N^2) per recorded frame, per trial."""
        self.set(theta)
        m, d = self.model, self.model.dismech
        w = self.weights()
        seeds = [wp.full(len(self.loss_x), float(v) / self.trials, dtype=float) for v in w]
        gp, gm, gi, sq = 0.0, 0.0, 0.0, np.zeros(2)
        for c in range(self.chunks):
            for a in (d.triplet_params, m.particle_mass, d.edge_inertia):
                a.grad.zero_()
            tape = wp.Tape()
            with tape:
                self.run(c)
            tape.backward(grads=dict(zip((self.loss_x, self.loss_f), seeds)))
            gp = gp + d.triplet_params.grad.numpy().reshape(self.p.shape)
            gm = gm + m.particle_mass.grad.numpy()
            gi = gi + d.edge_inertia.grad.numpy()
            sq += [a.numpy().sum() / self.trials for a in (self.loss_x, self.loss_f)]
            tape.zero()
            tape.reset()
        self.sq = sq
        g = [self.rho * (gm @ self.m_true + gi @ self.i_true) if kind == "rho"
             else np.sum(gp[..., COLUMNS[kind]] * self.p[..., COLUMNS[kind]]) for kind in KINDS]
        return float(sq @ w), np.array(g)


def lbfgs(f, theta0: np.ndarray) -> tuple[list, list]:
    """L-BFGS from ``theta0``: its accepted iterates and their losses."""
    best = {}

    def fun(theta):
        loss, g = f(theta)
        if np.isfinite(loss) and np.all(np.isfinite(g)):
            if not best or loss < best["loss"]:
                best.update(loss=loss, theta=theta.copy())
            return loss, g
        # a few windows blew up (low damping and a noisy start): a smooth wall back to the best point so far,
        # so the line search backtracks instead of stopping
        dx = theta - best["theta"]
        return best["loss"] + 1.0 + 100.0 * dx @ dx, 200.0 * dx

    iterates, losses = [theta0], [fun(theta0)[0]]

    def accepted(intermediate_result):  # scipy passes the OptimizeResult to a parameter of this name
        r = intermediate_result
        iterates.append(r.x.copy())
        losses.append(r.fun)
        print(f"  step {len(iterates) - 1}: loss {r.fun:.4f}  "
              + "  ".join(f"{k} {v:.3f}" for k, v in zip(KINDS, np.exp(r.x))), flush=True)

    minimize(fun, theta0, jac=True, method="L-BFGS-B", bounds=[(-np.log(30.0), np.log(30.0))] * len(KINDS),
             callback=accepted, options=dict(maxiter=200, ftol=1e-12, gtol=1e-9))
    return iterates, losses


def fit(data: dict) -> dict:
    """The fit, twice: weighing the force by the sensor's noise, then by its error at that fit. Every accepted
    iterate and the step the second fit starts at."""
    f = WindowFit(data)
    tic = time.perf_counter()
    iterates, losses = lbfgs(f, np.log([GUESS[k] for k in KINDS]))
    first = len(iterates) - 1
    f(iterates[-1])
    f.sigma = np.sqrt(f.sq[1] / 3.0)  # rms per component: mostly the readout drive's pose error
    print(f"the force's error at the fit: {f.sigma:.3g} N; fit again")
    more, more_losses = lbfgs(f, iterates[-1])
    loss_true = f(np.zeros(len(KINDS)))[0]
    print(f"fit: {time.perf_counter() - tic:.0f} s; loss {more_losses[-1]:.4f} (at the truth {loss_true:.4f})")
    return dict(theta=np.array(iterates + more[1:]), loss=np.array(losses + more_losses[1:]), loss_true=loss_true,
                refit=first, sigma=f.sigma)


def replay(data: dict, trials: np.ndarray, ratios: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every ``ratios`` row (parameters over the truth) on every trial: the whole experiment from the first
    recorded frame, driven by the smoothed readout. Node positions (len(ratios), len(trials), frames, NODES, 3)
    and the gripper's force (len(ratios), len(trials), frames, 3)."""
    n, R = len(ratios), len(ratios) * len(trials)
    model, solver = build(R, np.repeat(ratios, len(trials), 0), admm=True)
    rest = rest_state(model, solver)
    st = model.state()
    flatten_state(st)
    q = rest.dismech.q.numpy()
    vals = np.tile(drive_readout(data, trials).reshape(STEPS + 1, 1, -1), (1, n, 1)).reshape(STEPS + 1, -1)
    fixed = fixed_dofs(R)
    q[: 3 * NODES * R] = np.tile(data["x_obs"][trials, 0], (n, 1, 1)).ravel()
    q[fixed] = vals[0]
    start(solver, st, rest, q)
    x, force = rollout(model, solver, st, fixed, vals)
    shape = (n, len(trials), FRAMES + 1)
    return x.reshape(*shape, NODES, 3).astype(np.float32), force.reshape(*shape, 3).astype(np.float32)


def shown_trials(data: dict, candidates: int = 32) -> np.ndarray:
    """A trial on each branch (up, down) that the truth, replayed from its first frame, follows."""
    trials = np.arange(candidates)
    x = replay(data, trials, np.ones((1, 4)))[0][0]
    up = mid_z(data["x"][trials, -1]) > 0
    follows = (np.sign(mid_z(x[:, [K_BUCKLE, -1]])) == np.sign(mid_z(data["x"][trials][:, [K_BUCKLE, -1]]))).all(1)
    return np.array([trials[follows & up][0], trials[follows & ~up][0]])


def results(fresh: bool = False) -> dict:
    """The recordings, the fit and the replays, each computed once per configuration (``.cache/examples``)."""
    experiment = dict(version=3, true=TRUE, strong=STRONG, length=LENGTH, segments=SEGMENTS, radius=RADIUS, dt=DT,
                      obs_every=OBS_EVERY, phases=[T_COMPRESS, T_SHEAR, T_HOLD], travel=[COMPRESS, SHEAR],
                      trials=TRIALS, seed=SEED, noise=[GRIP_T, GRIP_R, GRIP_TAU, GRIP_RAMP, SHAKE, SHAKE_MODES, OBS_X,
                                                       OBS_GRIP_T, OBS_GRIP_R, OBS_F])
    data = cached("fit_buckling-data", experiment, record, fresh)
    setup = dict(experiment=experiment, guess=GUESS, window=W, smooth=SMOOTH, velocity=VELOCITY, version=4)
    out = cached("fit_buckling-fit", setup, lambda: fit(data), fresh)

    def replays():
        trials = shown_trials(data)
        x, force = replay(data, trials, np.exp(out["theta"]))
        return dict(trials=trials, x=x, force=force)

    shown = cached("fit_buckling-replay", dict(setup=setup, version=3), replays, fresh)
    return dict(data=data, **out, **shown)


# -- the picture -------------------------------------------------------------------------------

GREY, GREEN = (0.82, 0.82, 0.82), (0.3, 0.8, 0.5)
COLORS = dict(EI="#5aaaff", EA="#4dcc80", rho="#ff8c3c", c="#c08cff")
TEX = dict(EI=r"$EI$", EA=r"$EA$", rho=r"$\rho$", c=r"$c$")


def convergence(ax, ratios: np.ndarray, k: int, refit: int, scale: float):
    """Each unknown over its truth at every iterate, up to iterate ``k``; the second fit starts at ``refit``."""
    from matplotlib.ticker import MaxNLocator  # noqa: PLC0415

    it = np.arange(len(ratios))
    ax.axhspan(0.9, 1.1, color="white", alpha=0.08, lw=0)
    ax.axhline(1.0, color="white", lw=1.0 * scale, ls=":", alpha=0.6)
    ax.axvline(refit, color="white", lw=1.0 * scale, ls="--", alpha=0.3)
    ax.text(refit, 0.17, " refit", color="white", alpha=0.5, fontsize=14 * scale, va="bottom")
    for j, kind in enumerate(KINDS):
        r = ratios[:, j]
        ax.plot(it, r, color=COLORS[kind], alpha=0.25)
        ax.plot(it[: k + 1], r[: k + 1], color=COLORS[kind], label=f"{TEX[kind]}  {r[k]:.2f}")
        ax.plot(k, r[k], "o", color=COLORS[kind], ms=7 * scale, mec="white", mew=1.2 * scale)
    ax.set(yscale="log", ylim=(0.15, 8.0), xlim=(0, len(ratios) - 1))
    ax.minorticks_off()
    ax.set_yticks([0.25, 0.5, 1, 2, 4], labels=["$1/4$", "$1/2$", "$1$", "$2$", "$4$"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("step")
    ax.set_ylabel("fit / truth")
    ax.legend(loc="upper right", ncol=2)


def force(ax, observed: np.ndarray, predicted: np.ndarray, t: int, ylim: tuple, scale: float):
    """The gripper's vertical force up to frame ``t``: the sensor's reading and the replay's."""
    time_ = np.arange(1, FRAMES + 1) * OBS_EVERY * DT
    ax.plot(time_[:t], observed[1 : t + 1], "o", color="white", alpha=0.35, ms=2.5 * scale, mew=0, label="observed")
    ax.plot(time_[:t], predicted[1 : t + 1], color=GREEN, label="predicted")
    ax.set(xlim=(0, time_[-1]), ylim=ylim)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"gripper force $F_z$ [N]")
    ax.legend(loc="lower left")


class Example:
    """The true motion (grey) and the iterate's replays (green) of the shown trials, stacked: one on the upper
    branch above one on the lower. Each iterate plays the whole experiment, then the next iterate."""

    pause = 40  # frames held on the last recorded frame
    gap = 0.09  # [m] between the stacked trials

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        r = results(bool(args is not None and args.fresh))
        self.trials = r["trials"]
        self.ratios = np.exp(r["theta"])
        self.refit = int(r["refit"])
        self.true = r["data"]["x"][self.trials].astype(np.float64)  # (trials, frames, NODES, 3): no measurement noise
        self.replays = r["x"].astype(np.float64)  # (iterates, trials, frames, NODES, 3)
        self.observed = r["data"]["f_obs"][self.trials[0], :, 2]  # the upper trial's vertical force
        self.predicted = r["force"][:, 0, :, 2]  # (iterates, frames)
        lo, hi = self.observed[1:].min(), self.observed[1:].max()
        self.ylim = (lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
        for k, ratio in enumerate(self.ratios):
            print(f"step {k:2d}: loss {r['loss'][k]:.4f}  " + "  ".join(f"{u} {v:.3f}" for u, v in zip(KINDS, ratio)))
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

    def panel(self, size=(800, 400)) -> np.ndarray:
        """The parameters at every iterate and the gripper's force so far, side by side."""

        def draw(axes):
            s = inset_scale(size, 2)
            convergence(axes[0], self.ratios, self.k, self.refit, s)
            force(axes[1], self.observed, self.predicted[self.k], self.t, self.ylim, s)

        return inset(draw, size, ncols=2)

    def step(self):
        self.frame += 1
        if self.frame > FRAMES + self.pause:
            self.frame = 0
            self.k = (self.k + 1) % len(self.ratios)
        if self.frame <= FRAMES + 1:
            self.viewer.log_image("fit", self.panel())
        self.sim_time += OBS_EVERY * DT

    def render(self):
        t = self.t
        toward = np.array([0.0, -0.02, 0.0])  # the iterate just in front of the truth
        x = np.concatenate([self.true[:, t] + self.lift, self.replays[self.k, :, t] + self.lift + toward])
        self.state.body_q.assign(np.concatenate([capsule_poses(rod) for rod in x]).astype(np.float32))
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def test_final(self):
        for u, r in zip(KINDS, self.ratios[-1]):
            assert abs(r - 1.0) < 0.2, f"{u}: fit / truth {r:.3f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--fresh", action="store_true", help="recompute the cached recordings, fit and replays")
    parser.set_defaults(num_frames=2000)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
