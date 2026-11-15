"""Tier 1 paired ``FrameGraph`` operator benchmarks at W=1 (latency) and W=256 (throughput)."""

import newton
import numpy as np
import pytest
import warp as wp
from conftest import bench_framegraph
from newton.solvers import SolverSemiImplicit
from utils.common import FrameGraph, grid

from dismech_newton import (
    ContactResidual,
    JointResidual,
    RodResidual,
    ShellResidual,
    SolverDiSMech,
    add_cloth_mesh,
    add_rod,
    fix_segment,
    register_custom_attributes,
)
from dismech_newton.contact import Slots
from dismech_newton.coupling import SolverCoupledProxy

RADIUS = 0.01


def _tile_worlds(sub: newton.ModelBuilder, worlds: int, device, gravity=(0.0, 0.0, -9.81)) -> newton.Model:
    if worlds == 1:
        m = sub.finalize(device=device)
        m.gravity.assign(gravity)
        return m
    mb = newton.ModelBuilder(gravity=gravity)
    register_custom_attributes(mb)
    for _ in range(worlds):
        mb.add_world(sub)
    m = mb.finalize(device=device)
    m.gravity.assign(gravity)
    return m


@pytest.mark.parametrize("worlds", [1, 256], ids=["W1", "W256"])
@pytest.mark.parametrize("variant", ["base", "dahl", "rff"])
def test_bench_rod(benchmark, device, needs_cuda, worlds: int, variant: str):
    """25-segment cantilever rod: baseline vs Dahl hysteresis vs 64-feature RFF residual."""
    sub = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(sub)
    n_pts = 25
    pts = np.column_stack([np.linspace(0.0, 1.0, n_pts), np.zeros(n_pts), np.zeros(n_pts)])
    add_rod(
        sub,
        rod=newton.Rod(pts, radius=RADIUS),
        stretch_stiffness=50.0,
        bend_stiffness=0.05,
        twist_stiffness=0.05,
        stretch_damping=1.0e-2,
        bend_damping=1.0e-3,
        twist_damping=1.0e-3,
        dahl_eps_max=0.02 if variant == "dahl" else 0.0,
        dahl_tau=0.05 if variant == "dahl" else 0.0,
    )
    fix_segment(sub, edge=0)
    m = _tile_worlds(sub, worlds, device)

    res = None
    if variant == "rff":
        rng = np.random.default_rng(5)
        w = rng.normal(0.0, 0.05, size=(64, 5)).astype(np.float32)
        res = RodResidual(rff_dim=64, w_rod=w, seed=5)
    sim = SolverDiSMech(m, self_contact=False, residual=res)
    s0, s1 = m.state(), m.state()
    dt = 0.01

    def work():
        for _ in range(2):
            sim.step(s0, s1, None, None, dt)
            sim.step(s1, s0, None, None, dt)

    bench_framegraph(benchmark, FrameGraph(work, sim, enabled=True), substeps=4)


@pytest.mark.parametrize("worlds", [1, 256], ids=["W1", "W256"])
@pytest.mark.parametrize("variant", ["base", "rff"])
def test_bench_shell(benchmark, device, needs_cuda, worlds: int, variant: str):
    """5x5 grid (32 triangles) cantilever shell: baseline vs 64-feature RFF residual."""
    sub = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(sub)
    v, f = grid(5, 5, 0.4, 0.4)
    ids = add_cloth_mesh(
        sub,
        vertices=v.tolist(),
        indices=f.ravel().tolist(),
        density=1.0,
        tri_ke=2.0e3,
        tri_ka=2.0e3,
        tri_kd=0.05,
        bend_stiffness=0.08,
        damping=1.0e-3,
    )
    for k in np.flatnonzero(v[:, 0] <= 0.1 + 1.0e-6):
        sub.particle_flags[int(ids[k])] &= ~newton.ParticleFlags.ACTIVE
    m = _tile_worlds(sub, worlds, device)

    res = None
    if variant == "rff":
        rng = np.random.default_rng(5)
        scale = np.array([0.25, 0.25, 0.25, 0.002, 0.002, 0.002], dtype=np.float32)
        w = (rng.normal(0.0, 1.0, size=(64, 6)) * scale).astype(np.float32)
        res = ShellResidual(rff_dim=64, w_shell=w, seed=6)
    sim = SolverDiSMech(m, self_contact=False, residual=res)
    s0, s1 = m.state(), m.state()
    dt = 0.01

    def work():
        for _ in range(2):
            sim.step(s0, s1, None, None, dt)
            sim.step(s1, s0, None, None, dt)

    bench_framegraph(benchmark, FrameGraph(work, sim, enabled=True), substeps=4)


@wp.kernel(enable_backward=False)
def _set_aniso(val: wp.vec4, s: Slots):
    i = wp.tid()
    if s.active[i] != 0:
        s.aniso[i] = val


@pytest.mark.parametrize("worlds", [1, 256], ids=["W1", "W256"])
@pytest.mark.parametrize("variant", ["base", "aniso", "rff"])
def test_bench_contact(benchmark, device, needs_cuda, worlds: int, variant: str):
    """4-segment rod sliding on ground: baseline vs elliptic anisotropy vs ContactResidual(R=32, M=4)."""

    sub = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    rod = newton.Rod.create_straight((-0.08, 0.0, 1.015 * RADIUS), (1.0, 0.0, 0.0), 0.16, segment_count=4,
                                     radius=RADIUS)
    add_rod(sub, rod=rod, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.3),
            stretch_stiffness=1.0e4, bend_stiffness=2.0, twist_stiffness=1.0)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.3))
    for _ in range(worlds):
        b.add_world(sub)
    m = b.finalize(device=device)
    m.dismech.surface_id.assign(np.zeros(m.particle_count, dtype=np.int32))
    ss = m.dismech.shape_surface_id.numpy().copy()
    ss[0] = 1
    m.dismech.shape_surface_id.assign(ss)
    su = m.dismech.shape_surface_u.numpy().copy()
    su[0] = np.array([0.8, 0.6, 0.0], dtype=np.float32)
    m.dismech.shape_surface_u.assign(su)

    res = None
    if variant == "rff":
        rng = np.random.default_rng(9)
        R, M = 32, 4
        emb = rng.normal(0.0, 0.5, (1, M, 8)).astype(np.float32)
        emb[:, :, 7] = np.array([[0.5, -0.3, 0.2, -0.4]], dtype=np.float32)
        wc = rng.normal(0.0, 0.4, (1, R, 2)).astype(np.float32)
        res = ContactResidual(rff_dim=R, num_materials=M, embeddings=emb, w_contact=wc, seed=9)
    sim = SolverDiSMech(m, self_contact=False, residual=res)
    if variant == "aniso":
        orig_fill = sim.contact.source.fill
        val = wp.vec4(0.8, 0.6, 0.0, 0.4)

        def _fill_aniso(state_in, dt_in, q, s):
            orig_fill(state_in, dt_in, q, s)
            wp.launch(_set_aniso, dim=s.active.shape[0], inputs=[val, s], device=m.device)

        sim.contact.source.fill = _fill_aniso

    s0, s1 = m.state(), m.state()
    s0.particle_qd.assign(np.tile([0.4, 0.2, 0.0], (m.particle_count, 1)).astype(np.float32))
    f_drv = np.tile([0.06, 0.02, 0.0], (m.particle_count, 1)).astype(np.float32)
    s0.particle_f.assign(f_drv)
    s1.particle_f.assign(f_drv)
    dt = 0.005

    def work():
        for _ in range(2):
            sim.step(s0, s1, None, None, dt)
            sim.step(s1, s0, None, None, dt)

    bench_framegraph(benchmark, FrameGraph(work, sim, enabled=True), substeps=4)


@pytest.mark.parametrize("worlds", [1, 256], ids=["W1", "W256"])
@pytest.mark.parametrize("variant", ["base", "joint_rff"])
def test_bench_coupled(benchmark, device, needs_cuda, worlds: int, variant: str):
    """Coupled 1-DOF arm + 16-node rod via ``SolverCoupledProxy``: baseline vs JointResidual(R=32)."""
    rng = np.random.default_rng(31)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    register_custom_attributes(b)
    rigid_b, rigid_j, rigid_sh, soft_p, soft_b, soft_sh, arm_b, prox_b = ([] for _ in range(8))
    pts = np.column_stack([np.linspace(-0.1, 0.15, 16), np.zeros(16), np.zeros(16)]).tolist()
    for _ in range(worlds):
        b.begin_world()
        link0 = b.add_link(mass=0.25, inertia=wp.mat33(np.diag([0.02, 0.03, 0.04])))
        sh0 = b.add_shape_sphere(link0, radius=0.02)
        j0 = b.add_joint_revolute(
            parent=-1, child=link0, parent_xform=wp.transform((0.0, 0.0, 0.0299), wp.quat_identity()),
            child_xform=wp.transform((-0.1, 0.0, 0.0), wp.quat_identity()), axis=(0, 1, 0),
        )
        b.add_articulation([j0])
        rigid_b.append(link0)
        rigid_j.append(j0)
        rigid_sh.append(sh0)
        ap = b.add_link(is_kinematic=True, mass=0.0, inertia=wp.mat33(np.zeros((3, 3))))
        apsh = b.add_shape_sphere(ap, radius=0.02)
        p0, s0_i = b.particle_count, b.shape_count
        rb = newton.Rod(points=pts, closed=False, radius=RADIUS)
        r_b, _ = add_rod(b, rod=rb, bend_stiffness=1.0e-2, stretch_stiffness=1.0e3, collide=True, proxies=True)
        fix_segment(b, body=r_b[0])
        b.end_world()
        soft_p.extend(range(p0, b.particle_count))
        soft_b.extend(r_b)
        soft_sh.append(apsh)
        soft_sh.extend(range(s0_i, b.shape_count))
        arm_b.append(link0)
        prox_b.append(ap)

    m = b.finalize(device=device)
    D = m.joint_dof_count // worlds
    spec = None
    if variant == "joint_rff":
        spec = JointResidual(
            rff_dim=32,
            coulomb=0.2,
            stribeck=0.1,
            w_joint=rng.normal(0.0, 0.2, (1, D, 32)).astype(np.float32),
        )
    entries = [
        SolverCoupledProxy.Entry(name="rigid", solver=lambda mv: SolverSemiImplicit(mv), bodies=rigid_b,
                                 joints=rigid_j, shapes=rigid_sh),
        SolverCoupledProxy.Entry(name="soft", solver=lambda mv: SolverDiSMech(mv, residual=spec),
                                 particles=soft_p, bodies=soft_b, shapes=soft_sh),
    ]
    proxies = [SolverCoupledProxy.Proxy(source="rigid", destination="soft", bodies=arm_b, proxy_bodies=prox_b)]
    coupled = SolverCoupledProxy(m, entries=entries, coupling=SolverCoupledProxy.Config(proxies=proxies, iterations=1))

    s0, s1, ctrl = m.state(), m.state(), m.control()
    newton.eval_fk(m, s0.joint_q, s0.joint_qd, s0)
    ctrl.joint_f = wp.zeros(m.joint_dof_count, dtype=float, device=device)
    fg = FrameGraph(lambda: coupled.step(s0, s1, ctrl, None, 2.0e-3), coupled._entries["soft"].solver)
    bench_framegraph(benchmark, fg, substeps=1, batch=20)
