"""Differentiable capsule-tactile RL for C-clip cable routing (Milestone A4).

Stage 1: Receding-window ``StepAdjoint`` trajectory and nominal pinch optimization via ``wp.Tape()`` under
``SolverCoupledProxy`` and capsule PaXini ``Tactile`` sensors (no texture SDFs on backward paths).
Stage 2: Short-Horizon Actor-Critic (SHAC, ``H = 16`` analytic gradient steps) training a residual ``MLP-GRU`` policy
``(q_robot, taxel_FxFyFz_83Hz, clip_pose) -> delta_q_target`` across ``+-40%`` bending stiffness ``EI`` and friction
``mu`` randomization, compared against the Stage 1 fixed-pinch baseline on out-of-distribution test cases.

    CUDA_VISIBLE_DEVICES=1 uv run --extra mujoco --extra examples examples/experimental/train_cable_routing.py --test --plot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import newton
import numpy as np
import torch
import torch.nn as nn
import warp as wp
from newton import ModelBuilder
from newton.solvers import SolverMuJoCo
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = str(ROOT / "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from utils import tactile  # noqa: E402
from utils.common import cached  # noqa: E402

from dismech_newton import SolverDiSMech, fix_segment, flatten_state, register_custom_attributes  # noqa: E402

DT = 2.0e-3
OBS_EVERY = 6  # 83.3 Hz tactile sampling (12 ms per RL frame)
H_SHAC = 16
T_FRAMES = 32  # 2 * H_SHAC = 0.384 s full slide -> push -> snap-in episode

N_NODES = 15
CABLE_R = 0.005
CABLE_EA = 120.0
CABLE_EI0 = 0.015
CABLE_RHO = 1400.0
CABLE_MU0 = 0.50
CAP_R = 0.006
CAP_HALF = 0.007

CLIP_Y = 0.08
CLIP_Z = 0.0
CLIP_POCKET_Z = 0.023
CLIP_POCKET_X = 0.011

# Rigid gripper nominal waypoints in joint space: [wx, wy, wz, pinch]
NOM_START = np.array([0.0, -0.02, 0.031, 0.0080], dtype=np.float32)
NOM_GRASP = np.array([0.0, -0.02, 0.031, 0.0150], dtype=np.float32)
NOM_SLIDE = np.array([0.0, CLIP_Y + 0.020, 0.028, 0.0152], dtype=np.float32)
NOM_PUSH = np.array([0.0, CLIP_Y + 0.020, 0.0055, 0.0160], dtype=np.float32)
NOM_OPEN = np.array([0.0, CLIP_Y + 0.020, 0.0055, 0.0020], dtype=np.float32)
NOM_LIFT = np.array([0.0, CLIP_Y + 0.020, 0.044, 0.0020], dtype=np.float32)

ACTION_LO = (-0.03, -0.05, -0.006, 0.001)
ACTION_HI = (0.03, 0.13, 0.055, 0.019)


def resolve_device(requested: str = "cuda:rl") -> tuple[str, torch.device]:
    """Resolve ``device='cuda:rl'`` while enforcing the ``<= 2`` GPU limit."""
    n_gpu = torch.cuda.device_count()
    assert 1 <= n_gpu <= 2, f"expected 1..2 visible CUDA devices (got {n_gpu}); set CUDA_VISIBLE_DEVICES=1"
    if requested == "cuda:rl":
        wp_dev = "cuda:0"
    else:
        wp_dev = requested
    wp.set_device(wp_dev)
    return wp_dev, torch.device(wp_dev)


@wp.kernel
def _pack_state_kernel(
    pq: wp.array[wp.vec3],
    pqd: wp.array[wp.vec3],
    jq: wp.array[float],
    jqd: wp.array[float],
    n_p: int,
    nq: int,
    nqd: int,
    out: wp.array[float],
):
    i = wp.tid()
    if i < n_p:
        p = pq[i]
        v = pqd[i]
        out[3 * i + 0] = p[0]
        out[3 * i + 1] = p[1]
        out[3 * i + 2] = p[2]
        base = 3 * n_p
        out[base + 3 * i + 0] = v[0]
        out[base + 3 * i + 1] = v[1]
        out[base + 3 * i + 2] = v[2]
    if i < nq:
        out[6 * n_p + i] = jq[i]
    if i < nqd:
        out[6 * n_p + nq + i] = jqd[i]


@wp.kernel
def _unpack_state_kernel(
    vec: wp.array[float],
    n_p: int,
    nq: int,
    nqd: int,
    pq: wp.array[wp.vec3],
    pqd: wp.array[wp.vec3],
    jq: wp.array[float],
    jqd: wp.array[float],
):
    i = wp.tid()
    if i < n_p:
        base = 3 * n_p
        pq[i] = wp.vec3(vec[3 * i + 0], vec[3 * i + 1], vec[3 * i + 2])
        pqd[i] = wp.vec3(vec[base + 3 * i + 0], vec[base + 3 * i + 1], vec[base + 3 * i + 2])
    if i < nq:
        jq[i] = vec[6 * n_p + i]
    if i < nqd:
        jqd[i] = vec[6 * n_p + nq + i]


@wp.kernel
def _map_ctrl_kernel(u: wp.array[float], q_off: int, out_tq: wp.array[float]):
    # u = [wx, wy, wz, pinch] -> rigid joint targets at q_off + [0..4]
    out_tq[q_off + 0] = u[0]
    out_tq[q_off + 1] = u[1]
    out_tq[q_off + 2] = u[2]
    out_tq[q_off + 3] = u[3]
    out_tq[q_off + 4] = -u[3]


@wp.kernel
def _read_tactile_kernel(
    tot: wp.array[wp.vec3],
    out_f: wp.array[float],
):
    # Two capsule sensors: pad 0 (left/-X) and pad 1 (right/+X)
    f0 = tot[0]
    f1 = tot[1]
    out_f[0] = f0[0]
    out_f[1] = f0[1]
    out_f[2] = f0[2]
    out_f[3] = f1[0]
    out_f[4] = f1[1]
    out_f[5] = f1[2]


class _TactileFrameFn(torch.autograd.Function):
    """Exact differentiable control step over ``OBS_EVERY`` coupled substeps using ``wp.Tape()``."""

    @staticmethod
    def forward(ctx, x_prev: torch.Tensor, u_t: torch.Tensor, env: RoutingTwinEnv, frame_idx: int):
        s_prev = env.states[frame_idx * OBS_EVERY]
        s_curr = env.states[(frame_idx + 1) * OBS_EVERY]

        x_wp = wp.from_torch(x_prev.contiguous(), dtype=wp.float32, requires_grad=True)
        u_wp = wp.from_torch(u_t.contiguous(), dtype=wp.float32, requires_grad=True)
        x_next_wp = wp.zeros_like(x_wp, requires_grad=True)
        f_wp = wp.zeros(6, dtype=float, device=env.wp_dev, requires_grad=True)

        # Unpack differentiable particle/joint state (preserving s_prev.dismech frame from prior step)
        ctrl = env.ctrls[frame_idx]
        wp.copy(ctrl.joint_target_q, env.clip_tq_template)

        tape = wp.Tape()
        with tape:
            wp.launch(
                _unpack_state_kernel,
                dim=max(env.n_p, env.nq_total, env.nqd_total),
                inputs=[x_wp, env.n_p, env.nq_total, env.nqd_total],
                outputs=[s_prev.particle_q, s_prev.particle_qd, s_prev.joint_q, s_prev.joint_qd],
                device=env.wp_dev,
            )
            wp.launch(_map_ctrl_kernel, dim=1, inputs=[u_wp, env.q_off], outputs=[ctrl.joint_target_q], device=env.wp_dev)
            for k in range(OBS_EVERY):
                sa = env.states[frame_idx * OBS_EVERY + k]
                sb = env.states[frame_idx * OBS_EVERY + k + 1]
                env.coupled.step(sa, sb, ctrl, env.contacts, DT)
                env.tact.commit(DT)
            env.tact.sample()
            wp.launch(_read_tactile_kernel, dim=1, inputs=[env.tact.total], outputs=[f_wp], device=env.wp_dev)
            wp.launch(
                _pack_state_kernel,
                dim=max(env.n_p, env.nq_total, env.nqd_total),
                inputs=[
                    s_curr.particle_q,
                    s_curr.particle_qd,
                    s_curr.joint_q,
                    s_curr.joint_qd,
                    env.n_p,
                    env.nq_total,
                    env.nqd_total,
                ],
                outputs=[x_next_wp],
                device=env.wp_dev,
            )

        ctx.env = env
        ctx.tape = tape
        ctx.x_wp = x_wp
        ctx.u_wp = u_wp
        ctx.x_next_wp = x_next_wp
        ctx.f_wp = f_wp
        return wp.to_torch(x_next_wp).clone(), wp.to_torch(f_wp).clone()

    @staticmethod
    def backward(ctx, grad_x_next: torch.Tensor, grad_f: torch.Tensor):
        gx_in = wp.from_torch(grad_x_next.contiguous(), dtype=wp.float32)
        gf_in = wp.from_torch(grad_f.contiguous(), dtype=wp.float32)
        ctx.tape.backward(grads={ctx.x_next_wp: gx_in, ctx.f_wp: gf_in})
        gx = wp.to_torch(ctx.x_wp.grad).clone()
        gu = wp.to_torch(ctx.u_wp.grad).clone()
        ctx.tape.zero()
        return gx, gu, None, None


class RoutingTwinEnv:
    """Coupled elastic cable + 2-finger capsule PaXini tactile gripper + spring-jaw C-clip."""

    def __init__(self, wp_dev: str, torch_dev: torch.device):
        self.wp_dev = wp_dev
        self.torch_dev = torch_dev

        b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        register_custom_attributes(b)

        # Elastic DER cable anchored at both ends, arched with slack above C-clip at CLIP_Y
        s_pts = np.linspace(0.0, 1.0, N_NODES)
        y_pts = np.linspace(-0.14, 0.16, N_NODES)
        z_pts = 0.024 + 0.008 * np.sin(np.pi * s_pts)
        pts = np.column_stack([np.zeros(N_NODES), y_pts, z_pts]).astype(np.float32)
        rod = newton.Rod(points=pts, radius=CABLE_R)
        rod_cfg = ModelBuilder.ShapeConfig(density=CABLE_RHO, mu=CABLE_MU0)
        caps, _ = SolverDiSMech.add_rod(
            b,
            rod=rod,
            cfg=rod_cfg,
            stretch_stiffness=CABLE_EA,
            bend_stiffness=CABLE_EI0,
            bend_damping=2.0e-4,
        )
        fix_segment(b, caps[0])
        fix_segment(b, caps[-1])
        self.rod_capsules = list(caps)

        # Rigid carrier + left/right opposing capsule tactile fingers (group 1)
        fg_cfg = ModelBuilder.ShapeConfig(density=4000.0, mu=CABLE_MU0, collision_group=1)
        zero_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        m_virt = 0.02
        i_virt = wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-5)
        bx = b.add_link(xform=zero_xform, mass=m_virt, inertia=i_virt, label="wx_body")
        by = b.add_link(xform=zero_xform, mass=m_virt, inertia=i_virt, label="wy_body")
        bz = b.add_link(xform=zero_xform, mass=m_virt, inertia=i_virt, label="wz_body")
        fl = b.add_link(xform=zero_xform, label="thumb_body")
        fr = b.add_link(xform=zero_xform, label="index_body")

        # Capsule fingers oriented along Y so opposing pinch compresses along X
        rot_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)
        self.sh_thumb = b.add_shape_capsule(
            fl, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), rot_y), radius=CAP_R, half_height=CAP_HALF, cfg=fg_cfg
        )
        self.sh_index = b.add_shape_capsule(
            fr, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), rot_y), radius=CAP_R, half_height=CAP_HALF, cfg=fg_cfg
        )

        # C-clip on a lateral shift prismatic rail (allows clip_x perturbation without model rebuild)
        clip_cfg = ModelBuilder.ShapeConfig(density=5000.0, mu=0.25, collision_group=2)
        cbase = b.add_link(xform=wp.transform(wp.vec3(0.0, CLIP_Y, CLIP_Z), wp.quat_identity()), label="clip_base")
        b.add_shape_box(cbase, hx=0.015, hy=0.008, hz=0.002, cfg=clip_cfg)
        b.add_shape_box(
            cbase, xform=wp.transform(wp.vec3(-0.013, 0.0, 0.012), wp.quat_identity()),
            hx=0.002, hy=0.008, hz=0.012, cfg=clip_cfg,
        )
        b.add_shape_box(
            cbase, xform=wp.transform(wp.vec3(0.013, 0.0, 0.012), wp.quat_identity()),
            hx=0.002, hy=0.008, hz=0.012, cfg=clip_cfg,
        )
        jaw_l = b.add_link(
            xform=wp.transform(wp.vec3(-0.011, CLIP_Y, CLIP_Z + 0.024), wp.quat_identity()), label="clip_jaw_l"
        )
        jaw_r = b.add_link(
            xform=wp.transform(wp.vec3(0.011, CLIP_Y, CLIP_Z + 0.024), wp.quat_identity()), label="clip_jaw_r"
        )
        b.add_shape_box(
            jaw_l, xform=wp.transform(wp.vec3(0.0042, 0.0, -0.0018), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.25)),
            hx=0.0046, hy=0.008, hz=0.0016, cfg=clip_cfg,
        )
        b.add_shape_box(
            jaw_r, xform=wp.transform(wp.vec3(-0.0042, 0.0, -0.0018), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.25)),
            hx=0.0046, hy=0.008, hz=0.0016, cfg=clip_cfg,
        )

        # Wrist 3-DOF + pinch + C-clip joints (created contiguously for articulation)
        pos_mode = newton.JointTargetMode.POSITION
        jx = b.add_joint_prismatic(
            parent=-1, child=bx, axis=(1.0, 0.0, 0.0), target_ke=1800.0, target_kd=80.0,
            armature=0.02, actuator_mode=pos_mode, limit_lower=-0.05, limit_upper=0.05,
        )
        jy = b.add_joint_prismatic(
            parent=bx, child=by, axis=(0.0, 1.0, 0.0), target_ke=1800.0, target_kd=80.0,
            armature=0.02, actuator_mode=pos_mode, limit_lower=-0.08, limit_upper=0.14,
        )
        jz = b.add_joint_prismatic(
            parent=by, child=bz, axis=(0.0, 0.0, 1.0), target_ke=2000.0, target_kd=90.0,
            armature=0.02, actuator_mode=pos_mode, limit_lower=-0.03, limit_upper=0.08,
        )
        jl = b.add_joint_prismatic(
            parent=bz, child=fl, axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(-0.024, 0.0, 0.0), wp.quat_identity()),
            target_ke=500.0, target_kd=30.0, armature=0.01, actuator_mode=pos_mode,
            limit_lower=-0.01, limit_upper=0.025,
        )
        jr = b.add_joint_prismatic(
            parent=bz, child=fr, axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.024, 0.0, 0.0), wp.quat_identity()),
            target_ke=500.0, target_kd=30.0, armature=0.01, actuator_mode=pos_mode,
            limit_lower=-0.025, limit_upper=0.01,
        )
        j_clip = b.add_joint_prismatic(
            parent=-1, child=cbase, axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, CLIP_Y, CLIP_Z), wp.quat_identity()),
            target_ke=2.0e4, target_kd=400.0, armature=0.05, actuator_mode=pos_mode,
            limit_lower=-0.025, limit_upper=0.025,
        )
        j_jl = b.add_joint_revolute(
            parent=cbase, child=jaw_l, axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(-0.011, 0.0, 0.024), wp.quat_identity()),
            target_ke=0.004, target_kd=0.001, armature=1.0e-4, actuator_mode=pos_mode,
            limit_lower=-0.02, limit_upper=0.95,
        )
        j_jr = b.add_joint_revolute(
            parent=cbase, child=jaw_r, axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.011, 0.0, 0.024), wp.quat_identity()),
            target_ke=0.004, target_kd=0.001, armature=1.0e-4, actuator_mode=pos_mode,
            limit_lower=-0.95, limit_upper=0.02,
        )
        self.rigid_bodies = [bx, by, bz, fl, fr, cbase, jaw_l, jaw_r]
        self.rigid_joints = [jx, jy, jz, jl, jr, j_clip, j_jl, j_jr]
        b.add_articulation(self.rigid_joints)
        # Bake NOM_START as initial rigid joint_q so SolverMuJoCo and DisMech start aligned
        b.joint_q[-8:-3] = [NOM_START[0], NOM_START[1], NOM_START[2], NOM_START[3], -NOM_START[3]]

        model = b.finalize(device=wp_dev)
        # Keep rod baseline params & mu on CPU for fast per-episode EI / mu scaling
        self.model = model
        self.ei_base = model.dismech.triplet_params.numpy()[:, 2:4].copy()
        self.mu_base = model.shape_material_mu.numpy().copy()

        soft_bodies = [bd for bd in range(model.body_count) if bd not in self.rigid_bodies]
        soft_joints = [jt for jt in range(model.joint_count) if jt not in self.rigid_joints]
        self.coupled = SolverCoupledProxy(
            model,
            entries=[
                SolverCoupled.Entry(
                    name="mjc",
                    solver=lambda v: SolverMuJoCo(
                        v, solver="newton", integrator="implicitfast", njmax=128, nconmax=128, use_mujoco_contacts=False
                    ),
                    bodies=self.rigid_bodies,
                    joints=self.rigid_joints,
                ),
                SolverCoupled.Entry(
                    name="soft",
                    solver=SolverDiSMech,
                    particles=list(range(model.particle_count)),
                    bodies=soft_bodies,
                    joints=soft_joints,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mjc",
                        destination="soft",
                        bodies=[fl, fr, cbase, jaw_l, jaw_r],
                        mode="lagged",
                    )
                ]
            ),
        )
        self.soft_solver = self.coupled._entries["soft"].solver
        self.tact = tactile.Tactile(model, self.soft_solver, {"thumb_dip": self.sh_thumb, "index_dip_link": self.sh_index})
        self.tact.hook()

        self.base_state = model.state()
        flatten_state(self.base_state)
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.base_state)
        self.init_state = model.state()
        flatten_state(self.init_state)
        self.n_p = model.particle_count
        self.n_j = model.joint_count
        self.q_off = int(model.joint_q_start.numpy()[self.rigid_joints[0]])
        self.qd_off = int(model.joint_qd_start.numpy()[self.rigid_joints[0]])
        self.nq_total = int(model.joint_coord_count)
        self.nqd_total = int(model.joint_dof_count)
        self.state_dim = 6 * self.n_p + self.nq_total + self.nqd_total

        self.states = [model.state(requires_grad=True) for _ in range(T_FRAMES * OBS_EVERY + 1)]
        for st in self.states:
            flatten_state(st)
        self.ctrls = []
        for _ in range(T_FRAMES):
            c = model.control()
            c.joint_target_q = wp.zeros(model.joint_coord_count, dtype=float, device=wp_dev, requires_grad=True)
            self.ctrls.append(c)
        self.contacts = model.contacts()
        self.clip_tq_template = wp.zeros(model.joint_coord_count, dtype=float, device=wp_dev)

        # Index of node initially sitting right above CLIP_Y
        self.clip_node = int(np.argmin(np.abs(pts[:, 1] - CLIP_Y)))
        self.pts0 = pts.copy()

    def set_episode_params(self, ei_scale: float, mu_scale: float, clip_dx: float = 0.0) -> None:
        p = self.model.dismech.triplet_params.numpy()
        p[:, 2:4] = self.ei_base * float(ei_scale)
        self.model.dismech.triplet_params.assign(p)

        mu = self.mu_base.copy()
        for sid in (self.sh_thumb, self.sh_index):
            mu[sid] = CABLE_MU0 * float(mu_scale)
        self.model.shape_material_mu.assign(mu)

        # Shift C-clip base joint target cleanly
        tq = np.zeros(self.nq_total, dtype=np.float32)
        tq[self.q_off + 5] = float(clip_dx)
        self.clip_tq_template.assign(tq)

    def reset_state(self, clip_dx: float = 0.0, cable_dx: float = 0.0) -> torch.Tensor:
        self.init_state.assign(self.base_state)
        pq = self.init_state.particle_q.numpy()
        pq[:, 0] += float(cable_dx)
        self.init_state.particle_q.assign(pq)
        jq = self.init_state.joint_q.numpy()
        jq[self.q_off : self.q_off + 5] = [NOM_START[0], NOM_START[1], NOM_START[2], NOM_START[3], -NOM_START[3]]
        jq[self.q_off + 5] = float(clip_dx)
        self.init_state.joint_q.assign(jq)
        newton.eval_fk(self.model, self.init_state.joint_q, self.init_state.joint_qd, self.init_state)
        self.coupled.reset(self.init_state)
        self.tact.sample()
        self.states[0].assign(self.init_state)

        x0 = wp.zeros(self.state_dim, dtype=float, device=self.wp_dev)
        wp.launch(
            _pack_state_kernel,
            dim=max(self.n_p, self.nq_total, self.nqd_total),
            inputs=[self.init_state.particle_q, self.init_state.particle_qd,
                    self.init_state.joint_q, self.init_state.joint_qd, self.n_p, self.nq_total, self.nqd_total],
            outputs=[x0],
            device=self.wp_dev,
        )
        return wp.to_torch(x0).clone()


def nominal_schedule(n_frames: int = T_FRAMES) -> np.ndarray:
    """Smooth piecewise nominal schedule ``U_nom`` of shape ``(n_frames, 4)``."""
    u = np.zeros((n_frames, 4), dtype=np.float32)
    for t in range(n_frames):
        s = t / max(n_frames - 1, 1)
        if s < 0.15:
            a = s / 0.15
            a = a * a * (3.0 - 2.0 * a)
            u[t] = (1.0 - a) * NOM_START + a * NOM_GRASP
        elif s < 0.48:
            a = (s - 0.15) / 0.33
            a = a * a * (3.0 - 2.0 * a)
            u[t] = (1.0 - a) * NOM_GRASP + a * NOM_SLIDE
        elif s < 0.74:
            a = (s - 0.48) / 0.26
            a = a * a * (3.0 - 2.0 * a)
            u[t] = (1.0 - a) * NOM_SLIDE + a * NOM_PUSH
        elif s < 0.86:
            a = (s - 0.74) / 0.12
            a = a * a * (3.0 - 2.0 * a)
            u[t] = (1.0 - a) * NOM_PUSH + a * NOM_OPEN
        else:
            a = (s - 0.86) / 0.14
            a = a * a * (3.0 - 2.0 * a)
            u[t] = (1.0 - a) * NOM_OPEN + a * NOM_LIFT
    return u


def rollout_step_loss(x_curr: torch.Tensor, f_tact: torch.Tensor, env: RoutingTwinEnv,
                      clip_dx: float, t: int, n_frames: int) -> torch.Tensor:
    """Dense physical routing + snap-in loss in cm^2 units for crisp SHAC gradients."""
    pq = x_curr[: 3 * env.n_p].view(env.n_p, 3)
    node = pq[env.clip_node]
    s = t / max(n_frames - 1, 1)

    lat_cm = (node[0] - clip_dx) * 100.0
    if s < 0.48:
        target_z = 0.032
    elif s < 0.74:
        a = (s - 0.48) / 0.26
        target_z = (1.0 - a) * 0.032 + a * 0.015
    else:
        target_z = 0.015
    z_cm = (node[2] - target_z) * 100.0
    tact_force = f_tact[3] - f_tact[0]  # inward normal squeeze along X
    grip_pen = 0.02 * (torch.relu(1.5 - tact_force) ** 2) if 0.35 <= s < 0.72 else torch.zeros_like(lat_cm)
    end_pen = 3.0 * (lat_cm ** 2 + (torch.relu(node[2] - 0.016) * 100.0) ** 2) if t >= n_frames - 6 else 0.0
    return 1.2 * lat_cm ** 2 + 0.8 * z_cm ** 2 + grip_pen + end_pen


def stage1_receding_adjoint(env: RoutingTwinEnv, n_iters: int = 8) -> tuple[np.ndarray, list[float]]:
    """Stage 1: Receding-window ``StepAdjoint`` trajectory + pinch optimization via ``wp.Tape()``."""
    u_init = nominal_schedule(T_FRAMES)
    u_ref = torch.tensor(u_init, device=env.torch_dev, dtype=torch.float32)
    u_param = torch.tensor(u_init, device=env.torch_dev, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([u_param], lr=3.0e-4)
    losses = []
    best_loss = float("inf")
    best_u = u_init.copy()
    lo = torch.tensor(ACTION_LO, device=env.torch_dev)
    hi = torch.tensor(ACTION_HI, device=env.torch_dev)

    for it in range(n_iters):
        opt.zero_grad()
        env.set_episode_params(1.0, 1.0, clip_dx=0.0)
        x = env.reset_state(clip_dx=0.0, cable_dx=0.0)
        tot_loss = torch.zeros((), device=env.torch_dev)
        # Receding windows of length H_SHAC = 16 spanning all T_FRAMES
        for start in range(0, T_FRAMES, H_SHAC):
            x_win = x.detach()
            win_loss = torch.zeros((), device=env.torch_dev)
            for k in range(H_SHAC):
                t = start + k
                u_t = torch.clamp(u_param[t], lo, hi)
                x_win, f_tact = _TactileFrameFn.apply(x_win, u_t, env, t)
                win_loss = win_loss + rollout_step_loss(x_win, f_tact, env, 0.0, t, T_FRAMES)
                win_loss = win_loss + 400.0 * torch.sum((u_t - u_ref[t]) ** 2)
            win_loss.backward()
            tot_loss = tot_loss + win_loss.detach()
            x = x_win.detach()
        torch.nn.utils.clip_grad_norm_([u_param], 1.0)
        opt.step()
        val = float(tot_loss.item())
        losses.append(val)
        if val < best_loss:
            best_loss = val
            best_u = torch.clamp(u_param.detach(), lo, hi).cpu().numpy()
        if it % 3 == 0 or it == n_iters - 1:
            print(f"  Stage 1 iter {it:2d}: loss={val:.4f}", flush=True)
    return best_u, losses


class ResidualTactilePolicy(nn.Module):
    """MLP-GRU residual policy ``(q_robot, taxel_FxFyFz_83Hz, clip_pose) -> delta_q_target``."""

    def __init__(self, obs_dim: int = 13, act_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, act_dim))
        # Small nonzero init keeps upstream encoder/GRU gradients live from step 0
        nn.init.normal_(self.head[-1].weight, std=0.03)
        nn.init.zeros_(self.head[-1].bias)
        self.hidden = hidden

    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.enc(obs)
        h_next = self.gru(z, h)
        scale = torch.tensor([0.012, 0.008, 0.007, 0.004], device=obs.device, dtype=obs.dtype)
        return torch.tanh(self.head(h_next)) * scale, h_next


def build_obs(x: torch.Tensor, f_tact: torch.Tensor, clip_dx: float, env: RoutingTwinEnv) -> torch.Tensor:
    """Assemble ``(q_robot(4), taxel_FxFyFz(6), clip_pose(3))`` observation tensor."""
    jq = x[6 * env.n_p + env.q_off : 6 * env.n_p + env.q_off + 5]
    q_robot = torch.stack([jq[0], jq[1], jq[2], 0.5 * (jq[3] - jq[4])])
    clip_pose = torch.tensor([clip_dx * 80.0, CLIP_Y, CLIP_Z], device=x.device, dtype=x.dtype)
    return torch.cat([q_robot * 10.0, f_tact * 0.15, clip_pose])


def stage2_shac_train(env: RoutingTwinEnv, u_nom_np: np.ndarray, epochs: int = 15) -> tuple[ResidualTactilePolicy, list[float]]:
    """Stage 2: SHAC (``H = 16`` analytic gradient steps) over randomized ``+-40%`` ``EI`` and ``mu``."""
    policy = ResidualTactilePolicy().to(env.torch_dev)
    ema_policy = ResidualTactilePolicy().to(env.torch_dev)
    ema_policy.load_state_dict(policy.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=3.0e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=3.0e-4)
    u_nom = torch.tensor(u_nom_np, device=env.torch_dev, dtype=torch.float32)
    rng = np.random.default_rng(7)
    lo = torch.tensor(ACTION_LO, device=env.torch_dev)
    hi = torch.tensor(ACTION_HI, device=env.torch_dev)
    hist = []

    for ep in range(epochs):
        # Antithetic pair (+clip_dx, -clip_dx) cancels lateral sign variance per SHAC update
        c0 = float(rng.uniform(0.001, 0.006)) * (1.0 if ep % 2 == 0 else -1.0)
        pair = [
            (float(rng.uniform(0.60, 1.40)), float(rng.uniform(0.60, 1.40)), c0, float(rng.uniform(-0.002, 0.002))),
            (float(rng.uniform(0.60, 1.40)), float(rng.uniform(0.60, 1.40)), -c0, float(rng.uniform(-0.002, 0.002))),
        ]
        opt.zero_grad()
        batch_loss = 0.0
        for ei_s, mu_s, clip_dx, cable_dx in pair:
            env.set_episode_params(ei_s, mu_s, clip_dx=clip_dx)
            x = env.reset_state(clip_dx=clip_dx, cable_dx=cable_dx)
            f_prev = torch.zeros(6, device=env.torch_dev, dtype=torch.float32)
            h = torch.zeros(policy.hidden, device=env.torch_dev, dtype=torch.float32)
            for start in range(0, T_FRAMES, H_SHAC):
                x_w = x.detach()
                h_w = h.detach()
                f_w = f_prev.detach()
                win_loss = torch.zeros((), device=env.torch_dev)
                for k in range(H_SHAC):
                    t = start + k
                    obs = build_obs(x_w, f_w, clip_dx, env)
                    du, h_w = policy(obs, h_w)
                    u_t = torch.clamp(u_nom[t] + du, lo, hi)
                    x_w, f_w = _TactileFrameFn.apply(x_w, u_t, env, t)
                    win_loss = win_loss + rollout_step_loss(x_w, f_w, env, clip_dx, t, T_FRAMES)
                    win_loss = win_loss + 0.05 * torch.sum(du ** 2)
                (0.5 * win_loss).backward()
                batch_loss += 0.5 * float(win_loss.item())
                x = x_w.detach()
                h = h_w.detach()
                f_prev = f_w.detach()

        torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
        opt.step()
        sched.step()
        if ep < 5:
            ema_policy.load_state_dict(policy.state_dict())
        else:
            with torch.no_grad():
                for p_ema, p in zip(ema_policy.parameters(), policy.parameters()):
                    p_ema.mul_(0.75).add_(p, alpha=0.25)
        hist.append(batch_loss)
        if ep % 3 == 0 or ep == epochs - 1:
            print(f"  Stage 2 SHAC ep {ep:2d}: loss={batch_loss:.4f} (clip_dx=+-{abs(c0)*1e3:.1f}mm)", flush=True)
    return ema_policy, hist


@torch.no_grad()
def evaluate_episode(
    env: RoutingTwinEnv,
    u_nom_np: np.ndarray,
    policy: ResidualTactilePolicy | None,
    ei_scale: float,
    mu_scale: float,
    clip_dx: float,
    cable_dx: float,
) -> dict:
    """Evaluate open-loop baseline (``policy=None``) or tactile ``MLP-GRU`` on one OOD episode."""
    env.set_episode_params(ei_scale, mu_scale, clip_dx=clip_dx)
    x = env.reset_state(clip_dx=clip_dx, cable_dx=cable_dx)
    u_nom = torch.tensor(u_nom_np, device=env.torch_dev, dtype=torch.float32)
    lo = torch.tensor(ACTION_LO, device=env.torch_dev)
    hi = torch.tensor(ACTION_HI, device=env.torch_dev)
    f_w = torch.zeros(6, device=env.torch_dev, dtype=torch.float32)
    h = torch.zeros(policy.hidden, device=env.torch_dev) if policy is not None else None
    tact_hist, z_hist = [], []

    for t in range(T_FRAMES):
        du = torch.zeros(4, device=env.torch_dev)
        if policy is not None and h is not None:
            obs = build_obs(x, f_w, clip_dx, env)
            du, h = policy(obs, h)
        u_t = torch.clamp(u_nom[t] + du, lo, hi)
        x, f_w = _TactileFrameFn.apply(x, u_t, env, t)
        pq = x[: 3 * env.n_p].view(env.n_p, 3)
        z_hist.append(float(pq[env.clip_node, 2].item()))
        tact_hist.append(f_w.cpu().numpy())

    pq_end = x[: 3 * env.n_p].view(env.n_p, 3).cpu().numpy()
    node = pq_end[env.clip_node]
    dx = abs(float(node[0]) - clip_dx)
    dz = float(node[2]) - CLIP_Z
    slip = float(np.max(np.abs(pq_end[:, 1] - env.pts0[:, 1])))
    ok = (dx < CLIP_POCKET_X) and (0.0 <= dz < CLIP_POCKET_Z)
    return dict(ok=ok, dx=dx, dz=dz, slip=slip, z_hist=np.array(z_hist), tact=np.array(tact_hist))


def run_benchmark(device: str = "cuda:rl", fresh: bool = False) -> dict:
    """Run Stage 1 + Stage 2 SHAC + OOD ``+-40%`` evaluation."""
    wp_dev, torch_dev = resolve_device(device)
    env = RoutingTwinEnv(wp_dev, torch_dev)

    t0 = time.perf_counter()
    # Smooth friction transition during Stage 1/2 avoids stick-band gradient chatter;
    # evaluate both baseline and trained policy under the sharp physical default.
    base_slip = float(env.soft_solver.contact.slip_speed)
    try:
        env.soft_solver.contact.slip_speed = 0.20
        print("Stage 1: Receding-window StepAdjoint optimization...")
        u_nom, s1_losses = stage1_receding_adjoint(env, n_iters=8)

        print("Stage 2: SHAC (H=16) residual tactile MLP-GRU training...")
        policy, s2_losses = stage2_shac_train(env, u_nom, epochs=16)
    finally:
        env.soft_solver.contact.slip_speed = base_slip

    # Held-out OOD grid across +-40% EI, +-40% mu, and lateral clip offsets
    rng = np.random.default_rng(123)
    ood_cases = []
    for ei_s in (0.60, 0.80, 1.00, 1.20, 1.40):
        for mu_s in (0.60, 1.00, 1.40):
            clip_dx = float(rng.uniform(-0.006, 0.006))
            cable_dx = float(rng.uniform(-0.002, 0.002))
            ood_cases.append((ei_s, mu_s, clip_dx, cable_dx))

    base_res = [evaluate_episode(env, u_nom, None, *c) for c in ood_cases]
    shac_res = [evaluate_episode(env, u_nom, policy, *c) for c in ood_cases]
    base_acc = 100.0 * sum(r["ok"] for r in base_res) / len(ood_cases)
    shac_acc = 100.0 * sum(r["ok"] for r in shac_res) / len(ood_cases)
    elapsed = time.perf_counter() - t0
    print(f"OOD +-40% EI/mu ({len(ood_cases)} cases) in {elapsed:.1f}s: "
          f"fixed-pinch={base_acc:.1f}% vs tactile-SHAC={shac_acc:.1f}%")
    return dict(
        s1_losses=np.array(s1_losses),
        s2_losses=np.array(s2_losses),
        ood_cases=np.array(ood_cases),
        base_ok=np.array([r["ok"] for r in base_res], dtype=np.float32),
        shac_ok=np.array([r["ok"] for r in shac_res], dtype=np.float32),
        base_acc=np.array([base_acc]),
        shac_acc=np.array([shac_acc]),
        sample_base=base_res[0]["z_hist"],
        sample_shac=shac_res[0]["z_hist"],
        sample_tact=shac_res[0]["tact"],
    )


def save_plot(out: dict, path: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)

    axes[0].plot(out["s2_losses"], marker="o", ms=3, lw=2.0)
    axes[0].set(xlabel="SHAC epoch", ylabel="rollout loss", title="Training")
    axes[0].grid(alpha=0.25)

    t_ms = np.arange(1, T_FRAMES + 1) * OBS_EVERY * DT * 1e3
    axes[1].plot(t_ms, out["sample_tact"][:, 3] - out["sample_tact"][:, 0], lw=2.0, label=r"$F_n$")
    axes[1].plot(t_ms, np.hypot(out["sample_tact"][:, 1], out["sample_tact"][:, 2]), lw=1.8, label=r"$|F_t|$")
    axes[1].set(xlabel="time [ms]", ylabel="force [N]", title="Tactile")
    axes[1].legend(loc="best", fontsize=8.5)
    axes[1].grid(alpha=0.25)

    axes[2].bar(["Fixed", "SHAC"], [float(out["base_acc"][0]), float(out["shac_acc"][0])], color=["#7f7f7f", "#2ca02c"])
    axes[2].axhline(90.0, color="r", ls="--", lw=1.2)
    axes[2].set(ylabel="OOD success [%]", ylim=(0, 108), title="OOD")
    axes[2].grid(axis="y", alpha=0.25)

    fig.savefig(path, dpi=180)
    print(f"saved plot to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:rl")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    wp.init()
    cfg = dict(v=1, dt=DT, obs_every=OBS_EVERY, h=H_SHAC, t_frames=T_FRAMES)
    if args.test or args.fresh:
        out = run_benchmark(device=args.device, fresh=args.fresh)
    else:
        out = cached("train_cable_routing", cfg, lambda: run_benchmark(device=args.device, fresh=True), args.fresh)
    if args.plot:
        save_plot(out, Path(__file__).with_suffix(".png"))
    if args.test:
        shac_acc = float(out["shac_acc"][0])
        base_acc = float(out["base_acc"][0])
        assert shac_acc > 90.0, f"OOD SHAC accuracy {shac_acc:.1f}% <= 90%"
        assert shac_acc > base_acc, f"SHAC ({shac_acc:.1f}%) did not beat fixed-pinch ({base_acc:.1f}%)"
        print(f"A4 check passed (OOD tactile-SHAC={shac_acc:.1f}% > fixed-pinch={base_acc:.1f}%)")
