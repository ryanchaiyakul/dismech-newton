"""Live implicit coupled ``material(eps, p, rest, eps_prev, inv_dt) -> (energy, sigma, H_mat)`` hook: exact full
``J^T H_mat J`` assembly, nonlinear hyperelastic equilibrium, ``StepAdjoint`` verification, and preconditioner-only PSD
projection in ``assemble()``."""

import newton
import numpy as np
import warp as wp
from fd import assert_derivative, jacobian
from newton.solvers import SolverSemiImplicit
from rollouts import ClampedRollout, ShellRollout, simulate
from scipy.sparse import csr_matrix

from dismech_newton import SolverDiSMech, flatten_state
from dismech_newton.constants import vec5f, vec6f, vec10f, vec12f
from dismech_newton.coupling import SolverCoupledProxy
from dismech_newton.dofs import bind_dofs

mat55f = wp.types.matrix((5, 5), float)
mat66f = wp.types.matrix((6, 6), float)


@wp.func
def _coupled_rod_material(eps: vec5f, p: vec10f, rest: vec5f, eps_prev: vec5f, inv_dt: float):
    """Conservative quartic bend stiffening + off-diagonal bend-twist coupling + Kelvin-Voigt damping."""
    alpha = float(2.0)
    beta = float(0.3)
    de = eps - rest
    dp = eps - eps_prev
    c0, c1, c2, c3, c4 = p[5] * inv_dt, p[6] * inv_dt, p[7] * inv_dt, p[8] * inv_dt, p[9] * inv_dt
    d2_2, d3_2 = de[2] * de[2], de[3] * de[3]
    e = (
        0.5
        * (
            p[0] * de[0] * de[0]
            + p[1] * de[1] * de[1]
            + p[2] * d2_2
            + p[3] * d3_2
            + p[4] * de[4] * de[4]
        )
        + 0.5
        * (
            c0 * dp[0] * dp[0]
            + c1 * dp[1] * dp[1]
            + c2 * dp[2] * dp[2]
            + c3 * dp[3] * dp[3]
            + c4 * dp[4] * dp[4]
        )
        + 0.25 * alpha * (d2_2 * d2_2 + d3_2 * d3_2)
        + beta * de[2] * de[4]
    )
    sigma = vec5f(
        p[0] * de[0] + c0 * dp[0],
        p[1] * de[1] + c1 * dp[1],
        p[2] * de[2] + c2 * dp[2] + alpha * de[2] * d2_2 + beta * de[4],
        p[3] * de[3] + c3 * dp[3] + alpha * de[3] * d3_2,
        p[4] * de[4] + c4 * dp[4] + beta * de[2],
    )
    H = mat55f(
        p[0] + c0, 0.0, 0.0, 0.0, 0.0,
        0.0, p[1] + c1, 0.0, 0.0, 0.0,
        0.0, 0.0, p[2] + c2 + 3.0 * alpha * d2_2, 0.0, beta,
        0.0, 0.0, 0.0, p[3] + c3 + 3.0 * alpha * d3_2, 0.0,
        0.0, 0.0, beta, 0.0, p[4] + c4,
    )
    return e, sigma, H


@wp.func
def _coupled_shell_material(eps: vec6f, p: vec12f, rest: vec6f, eps_prev: vec6f, inv_dt: float):
    """Conservative quartic membrane stiffening + membrane-bend off-diagonal coupling + Kelvin-Voigt damping."""
    alpha = float(5.0)
    beta = float(1.0e-3)
    de = eps - rest
    dp = eps - eps_prev
    c0, c1, c2 = p[6] * inv_dt, p[7] * inv_dt, p[8] * inv_dt
    c3, c4, c5 = p[9] * inv_dt, p[10] * inv_dt, p[11] * inv_dt
    d0_2, d1_2 = de[0] * de[0], de[1] * de[1]
    e = (
        0.5
        * (
            p[0] * d0_2
            + p[1] * d1_2
            + p[2] * de[2] * de[2]
            + p[3] * de[3] * de[3]
            + p[4] * de[4] * de[4]
            + p[5] * de[5] * de[5]
        )
        + 0.5
        * (
            c0 * dp[0] * dp[0]
            + c1 * dp[1] * dp[1]
            + c2 * dp[2] * dp[2]
            + c3 * dp[3] * dp[3]
            + c4 * dp[4] * dp[4]
            + c5 * dp[5] * dp[5]
        )
        + 0.25 * alpha * (d0_2 * d0_2 + d1_2 * d1_2)
        + beta * de[0] * de[3]
    )
    sigma = vec6f(
        p[0] * de[0] + c0 * dp[0] + alpha * de[0] * d0_2 + beta * de[3],
        p[1] * de[1] + c1 * dp[1] + alpha * de[1] * d1_2,
        p[2] * de[2] + c2 * dp[2],
        p[3] * de[3] + c3 * dp[3] + beta * de[0],
        p[4] * de[4] + c4 * dp[4],
        p[5] * de[5] + c5 * dp[5],
    )
    H = mat66f(
        p[0] + c0 + 3.0 * alpha * d0_2, 0.0, 0.0, beta, 0.0, 0.0,
        0.0, p[1] + c1 + 3.0 * alpha * d1_2, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, p[2] + c2, 0.0, 0.0, 0.0,
        beta, 0.0, 0.0, p[3] + c3, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, p[4] + c4, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, p[5] + c5,
    )
    return e, sigma, H


@wp.func
def _indefinite_rod_material(eps: vec5f, p: vec10f, rest: vec5f, eps_prev: vec5f, inv_dt: float):
    """Non-convex bending law with negative curvature ``c_neg`` and off-diagonal coupling so ``H_mat`` is indefinite."""
    c_neg = float(-5.0)
    beta = float(2.0)
    de = eps - rest
    dp = eps - eps_prev
    c0, c1, c2, c3, c4 = p[5] * inv_dt, p[6] * inv_dt, p[7] * inv_dt, p[8] * inv_dt, p[9] * inv_dt
    e = (
        0.5
        * (
            p[0] * de[0] * de[0]
            + p[1] * de[1] * de[1]
            + p[2] * de[2] * de[2]
            + p[3] * de[3] * de[3]
            + p[4] * de[4] * de[4]
        )
        + 0.5
        * (
            c0 * dp[0] * dp[0]
            + c1 * dp[1] * dp[1]
            + c2 * dp[2] * dp[2]
            + c3 * dp[3] * dp[3]
            + c4 * dp[4] * dp[4]
        )
        + 0.5 * c_neg * de[2] * de[2]
        + beta * de[2] * de[3]
    )
    sigma = vec5f(
        p[0] * de[0] + c0 * dp[0],
        p[1] * de[1] + c1 * dp[1],
        (p[2] + c_neg) * de[2] + c2 * dp[2] + beta * de[3],
        p[3] * de[3] + c3 * dp[3] + beta * de[2],
        p[4] * de[4] + c4 * dp[4],
    )
    H = mat55f(
        p[0] + c0, 0.0, 0.0, 0.0, 0.0,
        0.0, p[1] + c1, 0.0, 0.0, 0.0,
        0.0, 0.0, p[2] + c2 + c_neg, beta, 0.0,
        0.0, 0.0, beta, p[3] + c3, 0.0,
        0.0, 0.0, 0.0, 0.0, p[4] + c4,
    )
    return e, sigma, H


def test_coupled_rod_material_equilibrium_and_tangent():
    """Live ``rod_material`` assembles full ``J^T H_mat J`` (including off-diagonal coupling), stiffens nonlinearly,
    and induces twist under planar vertical loading via bend-twist coupling."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    rod = newton.Rod.create_straight((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.4, segment_count=6, radius=0.006)
    SolverDiSMech.add_rod(b, rod=rod, stretch_stiffness=5.0e3, bend_stiffness=0.5, twist_stiffness=0.5,
                          bend_damping=2.0e-2, twist_damping=2.0e-2, collide=False, proxies=False)
    SolverDiSMech.fix_segment(b, edge=0)
    m = b.finalize()
    f = np.zeros((7, 3), dtype=np.float32)
    f[-1, 2] = -1.0

    sol_lin = SolverDiSMech(m, exact=True, iterations=40, self_contact=False)
    sol_mat = SolverDiSMech(m, exact=True, iterations=40, self_contact=False, rod_material=_coupled_rod_material)
    s_lin, _ = simulate(m, sol_lin, 25, 0.02, force=f)
    s_mat, _ = simulate(m, sol_mat, 25, 0.02, force=f)
    assert sol_mat.counts.numpy().tolist() == [0, 0, 0]
    nd = 3 * m.particle_count
    twists_lin = np.abs(s_lin.dismech.q.numpy()[nd:]).max()
    twists_mat = np.abs(s_mat.dismech.q.numpy()[nd:]).max()
    assert twists_lin < 1.0e-6
    assert twists_mat > 1.0e-2
    assert -s_mat.particle_q.numpy()[-1, 2] < -s_lin.particle_q.numpy()[-1, 2]

    # Verify assembled A includes full off-diagonal J^T H_mat J matching FD of element grad() around rest
    bind_dofs(sol_mat.dofs, m, sol_mat.mass, sol_mat.fixed, sol_mat._no_force, s_lin, 0.02)
    s0 = m.state()
    flatten_state(s0)
    blk = sol_mat.elements[0]
    blk.begin_step(s0, 0.02, True)
    sol_mat.A.vals.zero_()
    sol_mat._anchor.zero_()
    blk.assemble(s0.dismech.q, sol_mat.A)
    A = csr_matrix((sol_mat.A.vals.numpy(), sol_mat.A.indices.numpy(), sol_mat.A.indptr.numpy()),
                   shape=(sol_mat.A.n, sol_mat.A.n)).toarray()
    A = A + np.triu(A, 1).T

    q0 = s0.dismech.q.numpy()
    free = sol_mat.fixed.numpy() == 0
    b_p, b_m = wp.zeros(sol_mat.num_dofs, dtype=wp.float64), wp.zeros(sol_mat.num_dofs, dtype=wp.float64)
    v_test = np.zeros_like(q0)
    v_test[nd:][free[nd:]] = 1.0  # pure twist perturbation -> exercises J^T H_{4,2} J bend-twist off-diagonal coupling
    h = 1.0e-4
    blk.grad(wp.array((q0 + h * v_test).astype(np.float32), dtype=float), b_p)
    blk.grad(wp.array((q0 - h * v_test).astype(np.float32), dtype=float), b_m)
    fd_Av = (b_p.numpy() - b_m.numpy()) / (2.0 * h)
    # Must have nonzero off-diagonal force response in node z DOFs from pure twist perturbation
    assert np.linalg.norm(fd_Av[:nd][free[:nd]]) > 1.0e-2
    np.testing.assert_allclose(A[np.ix_(free, free)] @ v_test[free], fd_Av[free], rtol=1.0e-4, atol=1.0e-4)


def test_coupled_rod_material_adjoint(rng):
    """``StepAdjoint`` differentiates through a live coupled nonlinear ``rod_material`` matching FD across params,
    rest strains, states, and external forces."""
    opts = {"exact": True, "iterations": 30, "self_contact": False, "rod_material": _coupled_rod_material}
    ClampedRollout(rng).check(opts, fd_options=opts)


def test_coupled_shell_material_adjoint(rng):
    """``StepAdjoint`` differentiates through a live coupled nonlinear ``shell_material`` matching FD across
    membrane/bending stiffness, rest strains, masses, states, and forces."""
    opts = {"exact": True, "iterations": 30, "self_contact": False, "shell_material": _coupled_shell_material}
    ShellRollout(rng).check(opts, fd_options=opts)


def test_indefinite_material_projects_psd_only_in_assemble(rng):
    """Negative-curvature indefinite ``H_mat`` is projected PSD strictly inside ``assemble()``, leaving ``grad()`` and
    ``energy()`` exact and unclipped."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    rod = newton.Rod.create_straight((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.4, segment_count=6, radius=0.006)
    SolverDiSMech.add_rod(b, rod=rod, stretch_stiffness=1.0e2, bend_stiffness=0.05, twist_stiffness=0.05,
                          bend_damping=1.0e-3, twist_damping=1.0e-3, collide=False, proxies=False)
    SolverDiSMech.fix_segment(b, edge=0)
    m = b.finalize()
    sol = SolverDiSMech(m, exact=True, iterations=5, self_contact=False, rod_geometric=False,
                        rod_material=_indefinite_rod_material)
    s = m.state()
    flatten_state(s)
    q = s.dismech.q.numpy()
    q[3 * m.particle_count - 1] -= 0.05
    s.dismech.q.assign(q)
    bind_dofs(sol.dofs, m, sol.mass, sol.fixed, sol._no_force, s, 0.02)
    blk = sol.elements[0]
    blk.begin_step(s, 0.02, True)
    sol.A.vals.zero_()
    sol._anchor.zero_()
    blk.assemble(s.dismech.q, sol.A)
    M = csr_matrix((sol.A.vals.numpy(), sol.A.indices.numpy(), sol.A.indptr.numpy()), shape=(sol.A.n, sol.A.n)).toarray()
    M = M + np.triu(M, 1).T
    assert np.linalg.eigvalsh(M).min() >= -1.0e-3

    b_grad = wp.zeros(sol.num_dofs, dtype=wp.float64)
    blk.grad(s.dismech.q, b_grad)
    g_vec = b_grad.numpy()
    d = np.zeros_like(q)
    free = sol.fixed.numpy() == 0
    d[free] = rng.normal(size=int(free.sum())).astype(np.float32)
    h_fd = 1.0e-4
    qc = wp.array(np.stack([(q + h_fd * d).astype(np.float32), (q - h_fd * d).astype(np.float32)]), dtype=float)
    en = wp.zeros(sol.num_candidates, dtype=wp.float64)
    blk.energy(qc, en)
    fd_e = float(en.numpy()[0] - en.numpy()[1]) / (2.0 * h_fd)
    dot_g = float(np.dot(g_vec, d))
    assert abs(fd_e - dot_g) < 2.0e-3 * max(abs(fd_e), 1.0e-12)


@wp.kernel
def _hook_rod_eval_kernel(
    pb: wp.array[vec10f],
    rb: wp.array[vec5f],
    scale: wp.array[float],
    po: wp.array[vec10f],
    ro: wp.array[vec5f],
):
    i = wp.tid()
    p, r, s = pb[i], rb[i], scale[0]
    po[i] = vec10f(p[0], p[1], p[2] * s, p[3] * s, p[4] * s, p[5], p[6], p[7] * s, p[8] * s, p[9] * s)
    ro[i] = vec5f(r[0], r[1], r[2] + 0.02 * (s - 1.0), r[3], r[4])


@wp.kernel
def _hook_loss_kernel(q: wp.array[float], w: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(out, 0, q[i] * w[i])


@wp.kernel(enable_backward=False)
def _hook_scale_mu_kernel(factor: float, active: wp.array[wp.int32], mu: wp.array[float]):
    i = wp.tid()
    if active[i] != 0:
        mu[i] = mu[i] * factor


def test_prior_duck_typed_residual_hooks(rng):
    """Duck-typed ``SolverDiSMech(..., residual=...)`` hooks compose seamlessly without public residual classes:
    1) ``eval_rod`` + ``grad_inputs()`` propagate ``StepAdjoint`` gradients matching FD; 2) contact ``apply_contact`` /
    ``replay_contact`` modulate friction and replay on backward; 3) ``joint_residual`` runs via ``SolverCoupledProxy``."""

    class _DummyContactHook:
        def __init__(self, factor: float):
            self.factor = factor
            self.params = ()
            self.replayed = 0

        def bind(self, model, device=None):
            return self

        def apply_contact(self, s, state_in, source, arm, dt):
            wp.launch(_hook_scale_mu_kernel, dim=s.active.shape[0], inputs=[self.factor, s.active],
                      outputs=[s.mu], device=s.mu.device)

        def replay_contact(self, c, q_in, qd_in, poses, body_qd, source, dt, q_in0=None):
            self.replayed += 1
            return wp.clone(c.mu), c.pad, c.aniso

    class _DummyJointHook:
        def __init__(self):
            self.params = ()
            self.applied = 0

        def bind(self, model, device=None):
            return self

        def apply(self, state_in, control):
            self.applied += 1
            return control

    class _DummyResidual:
        has_rod = True
        has_shell = False
        has_rod_memory = False
        has_shell_memory = False

        def __init__(self, factor: float):
            self.scale = wp.array([1.25], dtype=float, requires_grad=True)
            self.contact = _DummyContactHook(factor)
            self.joint = _DummyJointHook()

        def bind(self, model, triplets, shells, device=None):
            return self

        def grad_inputs(self):
            return [self.scale]

        def eval_rod(self, rods, s, dt, pb, rb, po, ro):
            wp.launch(_hook_rod_eval_kernel, dim=rods.count, inputs=[pb, rb, self.scale], outputs=[po, ro],
                      device=pb.device)

    # 1. Rod hook forward + IFT backward wrt hook parameter
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    rod = newton.Rod.create_straight((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.4, segment_count=4, radius=0.008)
    SolverDiSMech.add_rod(b, rod=rod, stretch_stiffness=5.0e3, bend_stiffness=0.5, twist_stiffness=0.5,
                          collide=False, proxies=False)
    SolverDiSMech.fix_segment(b, edge=0)
    m = b.finalize()
    hook = _DummyResidual(0.4)
    sol = SolverDiSMech(m, exact=True, iterations=30, self_contact=False, residual=hook, tolerance=1.0e-5)
    assert sol.joint_residual is hook.joint

    w_np = rng.normal(size=sol.num_dofs).astype(np.float32)
    w_wp = wp.array(w_np, dtype=float)
    states = [m.state(requires_grad=True) for _ in range(3)]
    for s in states:
        flatten_state(s)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
        sol.step(states[0], states[1], None, None, 0.01)
        sol.step(states[1], states[2], None, None, 0.01)
        wp.launch(_hook_loss_kernel, dim=w_np.size, inputs=[states[2].dismech.q, w_wp], outputs=[loss])
    tape.backward(loss)
    adj_grad = float(hook.scale.grad.numpy()[0])

    def eval_scale(X):
        out = []
        for (ds,) in X:
            hook.scale.assign([float(1.25 + ds)])
            a, c, d = m.state(), m.state(), m.state()
            for s in (a, c, d):
                flatten_state(s)
            sol.step(a, c, None, None, 0.01)
            sol.step(c, d, None, None, 0.01)
            out.append([float(d.dismech.q.numpy().astype(np.float64) @ w_np.astype(np.float64))])
        return np.array(out)

    fd_j, est_j = jacobian(eval_scale, np.zeros(1), np.array([1.0e-4]), error=True)
    assert_derivative(adj_grad, float(fd_j[0, 0]), float(est_j[0, 0]), 5.0e-3, name="residual.scale")

    # 2. Contact hook scales effective friction (0.5 -> 0.2) and replays inside StepAdjoint
    g, v0, dt = 9.81, 1.0, 0.002
    bc = newton.ModelBuilder(gravity=(0.0, 0.0, -g))
    pts = np.column_stack([np.linspace(-0.1, 0.1, 9), np.zeros(9), np.full(9, 0.005)])
    SolverDiSMech.add_rod(bc, rod=newton.Rod(pts, radius=0.005), stretch_stiffness=1.0e3, bend_stiffness=1.0e-2,
                          twist_stiffness=1.0e-2, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.5))
    bc.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    mc = bc.finalize()
    hook_c = _DummyContactHook(0.4)
    sol_c = SolverDiSMech(mc, residual=hook_c)
    s0c, s1c = mc.state(requires_grad=True), mc.state(requires_grad=True)
    flatten_state(s0c)
    flatten_state(s1c)
    for i in range(15):
        sol_c.step(s0c, s1c, None, None, dt)
        s0c, s1c = s1c, s0c
    qd_c = np.zeros((mc.particle_count, 3), dtype=np.float32)
    qd_c[:, 1] = v0
    s0c.particle_qd.assign(qd_c)
    y0 = s0c.particle_q.numpy()[:, 1].mean()
    with wp.Tape() as tape_c:
        sol_c.step(s0c, s1c, None, None, dt)
    s1c.dismech.q.grad.fill_(1.0)
    tape_c.backward()
    assert hook_c.replayed == 1
    s0c, s1c = s1c, s0c
    for _ in range(int(0.4 / dt)):
        sol_c.step(s0c, s1c, None, None, dt)
        s0c, s1c = s1c, s0c
    slid = s0c.particle_q.numpy()[:, 1].mean() - y0
    exact_slide = v0**2 / (2.0 * 0.5 * 0.4 * g)
    assert abs(slid / exact_slide - 1.0) < 0.05, (slid, exact_slide)

    # 3. Standalone joint apply() hook wired cleanly through SolverCoupledProxy
    bj = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverDiSMech.register_custom_attributes(bj)
    link = bj.add_link(mass=0.25, inertia=wp.mat33(np.diag([0.02, 0.03, 0.04])))
    rsh = bj.add_shape_sphere(link, radius=0.02)
    j0 = bj.add_joint_revolute(parent=-1, child=link, parent_xform=wp.transform((0.0, 0.0, 0.03), wp.quat_identity()),
                               child_xform=wp.transform((-0.08, 0.0, 0.0), wp.quat_identity()), axis=(0, 1, 0))
    bj.add_articulation([j0])
    proxy = bj.add_link(is_kinematic=True, mass=0.0, inertia=wp.mat33(np.zeros((3, 3))))
    psh = bj.add_shape_sphere(proxy, radius=0.02)
    p0, sh0 = bj.particle_count, bj.shape_count
    rb, _ = SolverDiSMech.add_rod(
        bj, rod=newton.Rod.create_straight((-0.1, 0.0, 0.0), (1.0, 0.0, 0.0), 0.2, segment_count=4, radius=0.01),
        collide=True, proxies=True,
    )
    SolverDiSMech.fix_segment(bj, body=rb[0])
    mj = bj.finalize()
    jhook = _DummyJointHook()
    coupled = SolverCoupledProxy(
        mj,
        entries=[
            SolverCoupledProxy.Entry(name="rigid", solver=lambda mv: SolverSemiImplicit(mv),
                                     bodies=[link], joints=[j0], shapes=[rsh]),
            SolverCoupledProxy.Entry(name="soft", solver=lambda mv: SolverDiSMech(mv, residual=jhook),
                                     particles=list(range(p0, bj.particle_count)), bodies=rb,
                                     shapes=[psh, *range(sh0, bj.shape_count)]),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[SolverCoupledProxy.Proxy(source="rigid", destination="soft", bodies=[link], proxy_bodies=[proxy])],
            iterations=1,
        ),
    )
    sa, sb, ctrl = mj.state(), mj.state(), mj.control()
    newton.eval_fk(mj, sa.joint_q, sa.joint_qd, sa)
    coupled.step(sa, sb, ctrl, None, 2.0e-3)
    assert jhook.applied == 1
