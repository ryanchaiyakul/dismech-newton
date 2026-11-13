"""6-DOF AgileX Piper + 4-finger MIDAS hand with PaXini tactile fingertips routing a slack cable into a C-clip.

Demonstrates:
- Composing Piper 6-DOF arm + MIDAS 4-finger hand (unparked 4-DOF thumb) + PaXini tactile arrays under ``SolverCoupledProxy``.
- Coordinated pinch -> tactile-regulated ``>= 8 cm`` axial slip -> C-clip snap-in -> release across 1..512 worlds.
- Graph-capturable substeps (``FrameGraph``) and masked multi-world ``solver.reset(state, world_mask)``.
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import newton
import numpy as np
import warp as wp
from midas_axial import hand_xml
from newton.solvers import SolverMuJoCo
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy
from utils import tactile
from utils.common import CACHE, PAD_KH, CableExample, pads

import dismech_newton.coupling  # noqa: F401
from dismech_newton import SolverDiSMech, fix_segment, register_custom_attributes
from dismech_newton.state import masked_copy_kernel

PIPER_COMMIT = "ac41fcbcdda598f01b51cf6175ed9a24d0dacadc"

N_NODES = 41
CABLE_R = 0.005
CABLE_RHO = 1400.0
CABLE_EI = 0.015
CABLE_EA = 120.0
CABLE_X = 0.361
CABLE_Y0 = -0.20
CABLE_Y1 = 0.26
CABLE_Z0 = 0.272
CABLE_Z1 = 0.222
CLIP_POS = (0.361, 0.175, 0.170)

# Piper joint waypoints from Cartesian IK: (start_pregrasp, mid_slide, end_axial_slide, down_into_clip, lift_clear)
ARM_START = np.array([0.0295, 1.4479, -0.9646, -0.1880, 0.2270, -1.3739], dtype=np.float32)
ARM_MID = np.array([0.3659, 1.4776, -1.1251, -0.7056, 0.4954, -0.7113], dtype=np.float32)
ARM_SLIDE = np.array([0.6349, 1.6042, -1.4297, -0.7620, 0.8050, -0.5822], dtype=np.float32)
ARM_PUSH = np.array([0.6350, 1.6118, -1.3068, -0.8619, 0.7151, -0.4446], dtype=np.float32)
ARM_LIFT = np.array([0.6348, 1.6212, -1.6413, -0.6566, 0.9530, -0.7468], dtype=np.float32)

# Unparked 4-DOF thumb opposing index/middle/ring: [roll, side, mcp, dip], finger [pitch, pip]
THUMB_OPEN = np.array([2.15, 0.50, 0.05, 1.35], dtype=np.float32)
THUMB_PINCH = np.array([2.15, 0.3321, -0.3006, 1.3315], dtype=np.float32)
FINGER_OPEN = np.array([-0.55, -0.65], dtype=np.float32)
FINGER_PINCH = np.array([-0.845, -0.910], dtype=np.float32)

# Actuator indices in composed Piper (0..5) + MIDAS (6..18)
ACT_THUMB_MCP = 8
ACT_THUMB_DIP = 9
ACT_INDEX_PITCH = 11
ACT_INDEX_PIP = 12
ACT_MID_PITCH = 14
ACT_MID_PIP = 15
ACT_RING_PITCH = 17
ACT_RING_PIP = 18

CLIP_XML = f"""<mujoco model="c_clip">
  <worldbody>
    <body name="clip_base" pos="{CLIP_POS[0]} {CLIP_POS[1]} {CLIP_POS[2]}">
      <geom name="clip_floor" type="box" size="0.015 0.016 0.002" pos="0 0 0" contype="2" conaffinity="2"/>
      <geom name="clip_wall_l" type="box" size="0.002 0.016 0.012" pos="-0.013 0 0.012" contype="2" conaffinity="2"/>
      <geom name="clip_wall_r" type="box" size="0.002 0.016 0.012" pos="0.013 0 0.012" contype="2" conaffinity="2"/>
      <body name="clip_jaw_l" pos="-0.011 0 0.024">
        <inertial pos="0.004 0 0" mass="0.008" diaginertia="1e-5 1e-5 1e-5"/>
        <joint name="clip_jaw_l_hinge" type="hinge" axis="0 1 0" range="-0.02 0.90" stiffness="0.02" damping="0.003"/>
        <geom name="clip_jaw_l_geom" type="box" size="0.0045 0.015 0.0016" pos="0.0040 0 -0.0018" euler="0 0.30 0"
              contype="2" conaffinity="2"/>
      </body>
      <body name="clip_jaw_r" pos="0.011 0 0.024">
        <inertial pos="-0.004 0 0" mass="0.008" diaginertia="1e-5 1e-5 1e-5"/>
        <joint name="clip_jaw_r_hinge" type="hinge" axis="0 1 0" range="-0.90 0.02" stiffness="0.02" damping="0.003"/>
        <geom name="clip_jaw_r_geom" type="box" size="0.0045 0.015 0.0016" pos="-0.0040 0 -0.0018" euler="0 -0.30 0"
              contype="2" conaffinity="2"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def piper_xml() -> Path:
    """Download Piper ROS description if absent and return the arm MJCF path."""
    root = CACHE / f"piper_ros-{PIPER_COMMIT}"
    xml = root / "src" / "piper_description" / "mujoco_model" / "piper_no_gripper_description.xml"
    if xml.exists():
        return xml
    url = f"https://github.com/agilexrobotics/piper_ros/archive/{PIPER_COMMIT}.zip"
    CACHE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=60).read())) as zf:
        zf.extractall(CACHE)
    return xml


@wp.kernel
def _tactile_squeeze_kernel(
    world_count: int,
    nu: int,
    q_stride: int,
    sensors_per_world: int,
    act_q: wp.array[int],
    step_idx: wp.array[int],
    schedule: wp.array2d[float],
    cur_total: wp.array[wp.vec3],
    target_force: float,
    gain: float,
    out_targets: wp.array[float],
):
    w = wp.tid()
    if w >= world_count:
        return
    idx = wp.min(step_idx[w], schedule.shape[0] - 1)
    active = schedule[idx, nu]
    base_q = w * q_stride
    for j in range(nu):
        out_targets[base_q + act_q[j]] = schedule[idx, j]
    if active <= 0.0:
        return
    fz_sum = float(0.0)
    base_s = w * sensors_per_world
    for s in range(sensors_per_world):
        fz_sum += wp.abs(cur_total[base_s + s][2])
    delta = wp.clamp(gain * (target_force - fz_sum), -0.06, 0.24)
    # Increase opposing squeeze cleanly across thumb flex (- moves +X) and finger flexion (- moves -X)
    out_targets[base_q + act_q[ACT_THUMB_MCP]] = wp.clamp(schedule[idx, ACT_THUMB_MCP] - 0.8 * delta, -1.57, 1.57)
    out_targets[base_q + act_q[ACT_THUMB_DIP]] = wp.clamp(schedule[idx, ACT_THUMB_DIP] - 0.6 * delta, -1.57, 1.57)
    for k in range(3):
        pitch = 11 + 3 * k
        pip = 12 + 3 * k
        out_targets[base_q + act_q[pitch]] = wp.clamp(schedule[idx, pitch] - 0.5 * delta, -1.8, 0.0)
        out_targets[base_q + act_q[pip]] = wp.clamp(schedule[idx, pip] - 0.6 * delta, -1.45, 0.0)


@wp.kernel
def _inc_step_kernel(step_idx: wp.array[int]):
    w = wp.tid()
    step_idx[w] = step_idx[w] + 1


@wp.kernel
def _reset_step_kernel(mask: wp.array[wp.bool], step_idx: wp.array[int]):
    w = wp.tid()
    if mask[w]:
        step_idx[w] = 0


def _initial_targets() -> np.ndarray:
    u = np.zeros(19, dtype=np.float32)
    u[:6] = ARM_START
    u[6:10] = THUMB_OPEN
    for base in (10, 13, 16):
        u[base] = 0.0
        u[base + 1 : base + 3] = FINGER_OPEN
    return u


def _smooth(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


def _schedule_targets(t: float) -> tuple[np.ndarray, float]:
    """Return nominal (19,) actuator targets and tactile-squeeze gate at simulation time t [s]."""
    u = _initial_targets()
    if t < 0.30:
        a = _smooth(t / 0.30)
        arm = ARM_START
        thumb = (1.0 - a) * THUMB_OPEN + a * THUMB_PINCH
        finger = (1.0 - a) * FINGER_OPEN + a * FINGER_PINCH
        gate = 1.0 if t > 0.15 else 0.0
    elif t < 0.725:
        a = _smooth((t - 0.30) / 0.425)
        arm = (1.0 - a) * ARM_START + a * ARM_MID
        thumb, finger = THUMB_PINCH, FINGER_PINCH
        gate = 1.0
    elif t < 1.15:
        a = _smooth((t - 0.725) / 0.425)
        arm = (1.0 - a) * ARM_MID + a * ARM_SLIDE
        thumb, finger = THUMB_PINCH, FINGER_PINCH
        gate = 1.0
    elif t < 1.70:
        a = _smooth((t - 1.15) / 0.55)
        arm = (1.0 - a) * ARM_SLIDE + a * ARM_PUSH
        thumb, finger = THUMB_PINCH, FINGER_PINCH
        gate = 1.0
    elif t < 1.95:
        a = _smooth((t - 1.70) / 0.25)
        arm = ARM_PUSH
        thumb = (1.0 - a) * THUMB_PINCH + a * THUMB_OPEN
        finger = (1.0 - a) * FINGER_PINCH + a * FINGER_OPEN
        gate = 0.0
    else:
        a = _smooth((t - 1.95) / 0.45)
        arm = (1.0 - a) * ARM_PUSH + a * ARM_LIFT
        thumb, finger = THUMB_OPEN, FINGER_OPEN
        gate = 0.0
    u[:6] = arm
    u[6:10] = thumb
    for base in (10, 13, 16):
        u[base + 1 : base + 3] = finger
    return u, gate


def _cable_points(seed: int = 0) -> np.ndarray:
    """Slack anchored cable curve aligned cleanly between the pre-grasp pads and passing above the C-clip."""
    s = np.linspace(0.0, 1.0, N_NODES)
    y = (1.0 - s) * CABLE_Y0 + s * CABLE_Y1
    # Natural resting sag matching gravity equilibrium at grasp (z~0.237) and passing above C-clip (z~0.224)
    z = (1.0 - s) * CABLE_Z0 + s * CABLE_Z1 - 0.012 * np.sin(np.pi * s)
    jitter = np.random.default_rng(seed).normal(0.0, 0.0004) if seed else 0.0
    x = np.full(N_NODES, CABLE_X + jitter)
    return np.column_stack([x, y, z]).astype(np.float32)


def _set_initial_joints(builder: newton.ModelBuilder, clip_start_q: int, finger_start_q: int) -> None:
    """Place Piper + MIDAS at pre-grasp open pose in prototype coordinates."""
    builder.joint_q[:6] = ARM_START.tolist()
    builder.joint_q[6:10] = THUMB_OPEN.tolist()
    # index, middle, ring: each has [abad, pitch, pip, dip, linkage]
    for k in range(3):
        b = finger_start_q + 5 * k
        builder.joint_q[b : b + 3] = [0.0, float(FINGER_OPEN[0]), float(FINGER_OPEN[1])]
    builder.joint_q[clip_start_q : clip_start_q + 2] = [0.0, 0.0]


class Example(CableExample):
    """6-DOF Piper arm + 4-finger MIDAS hand + PaXini tactile C-clip cable routing."""

    fps = 30
    substeps = 20
    default_frames = 72
    plot_every = 20

    def __init__(self, viewer, args=None):
        self.world_count = max(1, int(getattr(args, "worlds", getattr(args, "world_count", 1)) if args else 1))
        self.seed = int(getattr(args, "seed", 0) if args is not None else 0)
        self.mu = float(getattr(args, "mu", 0.35) if args is not None else 0.35)

        proto = newton.ModelBuilder()
        register_custom_attributes(proto)
        proto.add_mjcf(str(piper_xml()), collapse_fixed_joints=False, parse_mujoco_options=False,
                       enable_self_collisions=False)
        link6 = next(i for i, n in enumerate(proto.body_label) if n.endswith("/link6"))
        q_hand = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.pi / 2))
        proto.add_mjcf(str(hand_xml()), parent_body=link6, xform=wp.transform(wp.vec3(0.0, 0.0, 0.01), q_hand),
                       parse_mujoco_options=False, enable_self_collisions=False)
        sensors = tactile.mount(proto)
        robot_bodies = proto.body_count
        proto.custom_attributes["mujoco:gravcomp"].values = dict.fromkeys(range(robot_bodies), 1.0)

        clip_q = len(proto.joint_q)
        proto.add_mjcf(CLIP_XML, parse_mujoco_options=False, enable_self_collisions=False)
        _set_initial_joints(proto, clip_q, finger_start_q=10)

        pts = _cable_points(self.seed)
        cfg = newton.ModelBuilder.ShapeConfig(density=CABLE_RHO, mu=self.mu)
        bodies, _ = SolverDiSMech.add_rod(
            proto,
            rod=newton.Rod(pts, radius=CABLE_R),
            cfg=cfg,
            stretch_stiffness=CABLE_EA,
            stretch_damping=0.05,
            bend_stiffness=CABLE_EI,
            bend_damping=1.0e-4,
            twist_stiffness=CABLE_EI,
            twist_damping=1.0e-4,
            collide=True,
        )
        # Anchor both ends with slight slack in the span
        for b_idx in (bodies[0], bodies[-1]):
            fix_segment(proto, body=b_idx)
        world = newton.ModelBuilder()
        register_custom_attributes(world)
        for _ in range(self.world_count):
            world.add_world(proto)
        model = world.finalize()
        model.shape_material_mu.fill_(self.mu)
        pads(model, [b for b, n in enumerate(model.body_label) if "_dip" in n], kh=PAD_KH)

        # Rigid MuJoCo domain: Piper + MIDAS + C-clip; Soft domain: DER rod + proxy links
        capsules = sorted({int(b) for b in model.dismech.edge_body.numpy() if b >= 0})
        rigid = [b for b in range(model.body_count) if b not in capsules]
        child = model.joint_child.numpy()
        rigid_joints = [j for j in range(model.joint_count) if child[j] in rigid]
        proxy_bodies = [b for b in rigid if any(k in model.body_label[b] for k in ("_dip", "clip_jaw"))]

        def mujoco(view):
            return SolverMuJoCo(
                view,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                iterations=100,
                ls_iterations=20,
                use_mujoco_contacts=False,
                njmax=256,
                nconmax=64,
            )

        solver = SolverCoupledProxy(
            model,
            entries=[
                SolverCoupled.Entry(name="mjc", solver=mujoco, bodies=rigid, joints=rigid_joints),
                SolverCoupled.Entry(
                    name="soft",
                    solver=SolverDiSMech,
                    particles=list(range(model.particle_count)),
                    bodies=capsules,
                    joints=[j for j in range(model.joint_count) if j not in rigid_joints],
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[SolverCoupledProxy.Proxy(source="mjc", destination="soft", bodies=proxy_bodies)]
            ),
        )
        self.soft = solver.solver("soft")
        self.mjc = solver.solver("mjc")
        self.tact = tactile.Tactile(model, self.soft, sensors)
        self.tact.hook()

        self.control = model.control()
        self.contacts = newton.Contacts(0, 0, device=model.device)
        self.nu = self.mjc.mj_model.nu
        self.q_stride = len(self.control.joint_target_q) // self.world_count
        act_names = (
            [f"joint{i}" for i in range(1, 7)]
            + ["thumb_cmc_roll_joint", "thumb_cmc_side_joint", "thumb_mcp_joint", "thumb_dip_joint"]
            + [
                f"{f}{j}"
                for f in ("index", "middle", "ring")
                for j in ("_mcp_abad_joint", "_mcp_pitch_joint", "_pip_joint")
            ]
        )
        p_labels = [n.split("/")[-1] for n in proto.joint_label]
        self._act_q = wp.array(
            [int(proto.joint_q_start[p_labels.index(name)]) for name in act_names],
            dtype=int,
            device=model.device,
        )
        frames = getattr(args, "frames", getattr(args, "num_frames", self.default_frames)) if args else self.default_frames
        total_steps = max(self.default_frames, int(frames)) * self.substeps
        sim_dt = 1.0 / (self.fps * self.substeps)
        sched = np.zeros((total_steps, self.nu + 1), dtype=np.float32)
        for k in range(total_steps):
            u, g = _schedule_targets((k + 1) * sim_dt)
            sched[k, : self.nu] = u
            sched[k, self.nu] = g
        self._schedule_wp = wp.array(sched, dtype=float, device=model.device)
        self._step_idx = wp.zeros(self.world_count, dtype=int, device=model.device)
        self.drives = [self._drive_step]

        self.start(viewer, model, solver)
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_1)
        dev = model.device
        self._init_state = {
            name: wp.clone(getattr(self.state_0, name))
            for name in ("particle_q", "particle_qd", "particle_f", "body_q", "body_qd", "body_f", "joint_q", "joint_qd")
            if getattr(self.state_0, name) is not None
        }
        self._jq_world = wp.array(
            np.repeat(np.arange(self.world_count, dtype=np.int32), self.state_0.joint_q.shape[0] // self.world_count),
            dtype=wp.int32,
            device=dev,
        )
        self._jqd_world = wp.array(
            np.repeat(np.arange(self.world_count, dtype=np.int32), self.state_0.joint_qd.shape[0] // self.world_count),
            dtype=wp.int32,
            device=dev,
        )
        self._tact_world = wp.array(
            np.repeat(np.arange(self.world_count, dtype=np.int32), self.tact.sensors_per_world),
            dtype=wp.int32,
            device=dev,
        )
        self._tact_zero = wp.zeros_like(self.tact.cur_total)

        # Indices on world 0 (and per-world strides) for diagnostics and axial slip tracking
        self.nodes_per_world = N_NODES
        self.grasp_node = int(np.argmin(np.abs(pts[:, 1] - 0.01)))
        self.clip_node = int(np.argmin(np.abs(pts[:, 1] - CLIP_POS[1])))
        self.mid_pad_body = next(i for i, n in enumerate(model.body_label) if n.endswith("/middle_dip_link"))
        self.clip_jaw_q_offset = clip_q

        self.balance = 0.0
        self.history: list[tuple[float, float, float, float, float]] = []
        self._slip_anchor_y: np.ndarray | None = None
        self.max_slip = np.zeros(self.world_count, dtype=np.float64)

    def _drive_step(self, _state, _frac: float) -> None:
        wp.launch(
            _tactile_squeeze_kernel,
            dim=self.world_count,
            inputs=[
                self.world_count,
                self.nu,
                self.q_stride,
                self.tact.sensors_per_world,
                self._act_q,
                self._step_idx,
                self._schedule_wp,
                self.tact.cur_total,
                2.5,
                0.035,
            ],
            outputs=[self.control.joint_target_q],
            device=self.model.device,
        )

    def simulate(self) -> None:
        for k in range(self.substeps):
            for drive in self.drives:
                drive(self.state_0, (k + 1) / self.substeps)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.post_substep()

    def post_substep(self) -> None:
        self.tact.commit(self.sim_dt)
        wp.launch(_inc_step_kernel, dim=self.world_count, inputs=[self._step_idx], device=self.model.device)

    def step(self) -> None:
        self.graph()
        self.sim_time += self.frame_dt
        self.frame += 1
        self.tact.sample()
        self._record_metrics()

    def _record_metrics(self) -> None:
        raw = self.tact.force.numpy().reshape(self.world_count, -1, 3)
        frames = self.tact.frame.numpy()
        back = np.einsum("tji,wtj->wti", frames, raw * [1.0, 1.0, -1.0])
        starts = self.tact.start
        sums = np.array(
            [
                back[w, int(a) : int(b)].sum(axis=0)
                for w in range(self.world_count)
                for a, b in zip(starts[:-1], starts[1:])
            ]
        )
        total = self.tact.total.numpy()
        self.balance = max(self.balance, float(np.abs(sums - total).max()))

        # Relative axial slip between middle fingertip pad and grabbed cable node
        pq = self.state_0.particle_q.numpy()
        bq = self.state_0.body_q.numpy()
        b_stride = self.model.body_count // self.world_count
        for w in range(self.world_count):
            pad_y = float(bq[w * b_stride + self.mid_pad_body, 1])
            node_y = float(pq[w * self.nodes_per_world + self.grasp_node, 1])
            rel_y = pad_y - node_y
            if self._slip_anchor_y is None and self.sim_time >= 0.28:
                self._slip_anchor_y = np.array(
                    [
                        float(bq[k * b_stride + self.mid_pad_body, 1])
                        - float(pq[k * self.nodes_per_world + self.grasp_node, 1])
                        for k in range(self.world_count)
                    ]
                )
            if self._slip_anchor_y is not None:
                self.max_slip[w] = max(self.max_slip[w], abs(rel_y - float(self._slip_anchor_y[w])))

        clip_z = float(pq[self.clip_node, 2])
        fz = float(np.abs(total[: self.tact.sensors_per_world, 2]).sum())
        ft = float(np.hypot(total[: self.tact.sensors_per_world, 0], total[: self.tact.sensors_per_world, 1]).sum())
        self.history.append((self.sim_time, float(self.max_slip.mean()), fz, ft, clip_z))

    def reset_worlds(self, mask: list[bool] | np.ndarray) -> None:
        """Graph-friendly masked reset preserving untouched worlds."""
        dev = self.model.device
        m_arr = wp.array(list(mask) + [False], dtype=wp.bool, device=dev)
        init = self._init_state
        pairs = [
            (self.model.particle_world, init["particle_q"], self.state_0.particle_q),
            (self.model.particle_world, init["particle_qd"], self.state_0.particle_qd),
            (self.model.body_world, init["body_q"], self.state_0.body_q),
            (self.model.body_world, init["body_qd"], self.state_0.body_qd),
            (self._jq_world, init["joint_q"], self.state_0.joint_q),
            (self._jqd_world, init["joint_qd"], self.state_0.joint_qd),
            (self._tact_world, self._tact_zero, self.tact.cur_total),
        ]
        if self.state_0.particle_f is not None:
            pairs.append((self.model.particle_world, init["particle_f"], self.state_0.particle_f))
        if self.state_0.body_f is not None:
            pairs.append((self.model.body_world, init["body_f"], self.state_0.body_f))
        for world_ids, src, dst in pairs:
            wp.launch(masked_copy_kernel, dim=src.shape[0], inputs=[0, world_ids, m_arr, src, dst], device=dev)
        self.solver.reset(self.state_0, world_mask=m_arr)
        self.state_1.assign(self.state_0)
        wp.launch(_reset_step_kernel, dim=self.world_count, inputs=[m_arr, self._step_idx], device=dev)

    def test_final(self) -> None:
        assert self.balance < 1.0e-4, f"|sum(F_taxel) - F_sensor| = {self.balance:.2e} >= 1e-4 N"
        pq = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(pq)), "non-finite cable positions"
        min_slip = float(self.max_slip.min())
        assert min_slip >= 0.08, f"axial slip {min_slip * 100:.2f} cm < 8.0 cm"
        # Verify cable captured inside C-clip pocket across every world (nearest node to clip y)
        for w in range(self.world_count):
            pw = pq[w * self.nodes_per_world : (w + 1) * self.nodes_per_world]
            c = pw[int(np.argmin(np.abs(pw[:, 1] - CLIP_POS[1])))]
            dx = abs(float(c[0]) - CLIP_POS[0])
            dz = float(c[2]) - CLIP_POS[2]
            assert dx < 0.011, f"world {w} missed clip x-pocket: |dx|={dx * 1000:.1f} mm"
            assert 0.0 <= dz < 0.023, f"world {w} not captured in clip z-pocket: dz={dz * 1000:.1f} mm (z={c[2]:.4f})"

    def log_plot(self, path: str | Path) -> None:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        arr = np.array(self.history)
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
        axes[0].plot(arr[:, 0], arr[:, 1] * 100.0, lw=2.0)
        axes[0].axhline(8.0, color="k", ls="--", lw=1.0)
        axes[0].set(xlabel="time [s]", ylabel="axial slip [cm]", title="Slip")
        axes[0].grid(alpha=0.25)

        axes[1].plot(arr[:, 0], arr[:, 2], lw=2.0, label=r"$F_n$")
        axes[1].plot(arr[:, 0], arr[:, 3], lw=1.8, label=r"$|F_t|$")
        axes[1].set(xlabel="time [s]", ylabel="force [N]", title="Tactile")
        axes[1].legend(loc="best", fontsize=8.5)
        axes[1].grid(alpha=0.25)

        axes[2].plot(arr[:, 0], (arr[:, 4] - CLIP_POS[2]) * 1000.0, lw=2.0)
        axes[2].axhline(23.0, color="r", ls=":", lw=1.2, label="jaw")
        axes[2].set(xlabel="time [s]", ylabel="height [mm]", title="Capture")
        axes[2].legend(loc="best", fontsize=8.5)
        axes[2].grid(alpha=0.25)

        fig.savefig(path, dpi=180)
        print(f"saved plot to {path}")


def _run_seeds(seeds: int, frames: int) -> None:
    viewer = newton.viewer.ViewerNull(num_frames=frames)
    slips, c_zs = [], []
    for seed in range(seeds):
        ex = Example(viewer, argparse.Namespace(worlds=1, seed=seed))
        for _ in range(frames):
            ex.step()
        pq = ex.state_0.particle_q.numpy()
        slips.append(float(ex.max_slip[0]) * 100.0)
        c_zs.append((float(pq[ex.clip_node, 2]) - CLIP_POS[2]) * 1000.0)
        ex.test_final()
    print(f"Seeds={seeds} passed: slip={np.median(slips):.2f} cm [{min(slips):.2f}, {max(slips):.2f}], "
          f"clip_dz={np.median(c_zs):.2f} mm [{min(c_zs):.2f}, {max(c_zs):.2f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--frames", type=int, default=Example.default_frames)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    wp.init()
    if args.seeds > 1:
        _run_seeds(args.seeds, args.frames)
    else:
        viewer = newton.viewer.ViewerNull(num_frames=args.frames)
        ex = Example(viewer, args)
        for _ in range(args.frames):
            ex.step()
        if args.plot:
            ex.log_plot(Path(__file__).with_suffix(".png"))
        if args.test or (not args.plot and args.frames >= Example.default_frames):
            ex.test_final()
            print(f"A3 single/batched check passed (worlds={args.worlds}, min_slip={ex.max_slip.min()*100:.2f} cm, "
                  f"balance={ex.balance:.2e} N)")
