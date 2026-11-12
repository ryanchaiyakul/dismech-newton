"""The direct solves against dense ones (upper CSR); SciPy on the CPU, cuDSS (float64 and float32) on CUDA."""

import numpy as np
import warp as wp
from fd import assert_close

from dismech_newton.linear import CudssSolver, ScipySolver, SymmetricCSR, cudss


def _system(rng, device, n: int = 40):
    """A random SPD matrix on the pattern of overlapping 4-DOF stencils, as an upper CSR, and the full matrix."""
    stencils = np.array([np.arange(k, k + 4) for k in range(0, n - 3, 2)])
    A = SymmetricCSR(n, (stencils,), device)
    full = n * np.eye(n)
    for d in stencils:
        block = rng.normal(size=(4, 4))
        full[np.ix_(d, d)] += block @ block.T
    indptr, indices = A.pattern
    rows = np.repeat(np.arange(n), np.diff(indptr))
    A.vals.assign(full[rows, indices])
    return A, full


def _solvers(A, device):
    """Each solver and its tolerance (the float32 solution's at best)."""
    if not device.is_cuda or cudss is None:  # (SciPy on a CUDA matrix warns: only without nvmath)
        return {"scipy": (ScipySolver(A), 1.0e-6)}
    return {"cudss": (CudssSolver(A), 1.0e-6), "cudss float32": (CudssSolver(A, single=True), 1.0e-5)}


def test_solvers_match_dense(device, rng):
    A, full = _system(rng, device)
    b = rng.normal(size=full.shape[0])
    expected = np.linalg.solve(full, b)
    for name, (solver, rtol) in _solvers(A, device).items():
        x = wp.zeros(len(b), dtype=float, device=device)
        solver.solve(wp.array(b, dtype=wp.float64, device=device), x)
        assert_close(x.numpy(), expected, rtol, name)


def test_invalidate_refactorises(device, rng):
    """New values take effect only after ``invalidate``."""
    A, full = _system(rng, device)
    b = rng.normal(size=full.shape[0])
    for name, (solver, _) in _solvers(A, device).items():
        x = wp.zeros(len(b), dtype=float, device=device)
        solver.solve(wp.array(b, dtype=wp.float64, device=device), x)
        A.vals.assign(2.0 * A.vals.numpy())
        solver.invalidate()
        solver.solve(wp.array(b, dtype=wp.float64, device=device), x)
        A.vals.assign(0.5 * A.vals.numpy())
        np.testing.assert_allclose(x.numpy(), np.linalg.solve(2.0 * full, b), rtol=1.0e-4, err_msg=name)


def test_two_sided_pcg_recovers_common_mode(device, rng):
    """A two-sided barrier pair in ``A`` charges ``2 k`` per side to ground; ``two_sided_mul`` restores the exact
    ``(x_0 - x_1)^T H (x_0 - x_1)`` operator and SPD PCG recovers the exact rigid-body translation."""
    import newton  # noqa: PLC0415

    from dismech_newton import SolverDiSMech, flatten_state  # noqa: PLC0415
    from dismech_newton.state import bind_dofs  # noqa: PLC0415

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    th = 0.01
    for z in (th, 0.0):
        rod = newton.Rod(np.array([[0.0, 0.0, z], [0.1, 0.0, z]]), radius=0.5 * th)
        SolverDiSMech.add_rod(builder, rod=rod, stretch_stiffness=0.0, bend_stiffness=0.0, twist_stiffness=0.0)
    model = builder.finalize(device=device)
    model.particle_mass.fill_(1.0)
    solver = SolverDiSMech(model)
    solver.refresh_mass()
    state = model.state()
    flatten_state(state)
    q = state.dismech.q

    # Activate one two-sided pair between node 0 (rod 0) and node 2 (rod 1) at d = th (curvature k = 1e5 along z).
    k_stiff = 1.0e5
    s = solver.contact.s
    s.active.assign([1] + [0] * (solver.contact.count - 1))
    s.world.zero_()
    pair = np.full((solver.contact.count, 4), -1, dtype=np.int32)
    pair[0, :2] = [0, 2]
    s.pair.assign(pair)
    w = np.zeros((solver.contact.count, 4), dtype=np.float32)
    w[0, :2] = [1.0, -1.0]
    s.w.assign(w)
    normal = np.zeros((solver.contact.count, 3), dtype=np.float32)
    normal[0] = [0.0, 0.0, 1.0]
    s.normal.assign(normal)
    s.anchor.zero_()
    c0 = np.zeros((solver.contact.count, 3), dtype=np.float32)
    c0[0] = [0.0, 0.0, th]
    s.c0.assign(c0)
    s.thickness.fill_(th)
    s.stiffness.fill_(k_stiff)
    s.lam.zero_()
    s.mu.zero_()
    s.rho.zero_()
    solver.contact.slip = 1.0e-2

    # 1. Quadratic form check: v^T (A_pair v + delta_A_2side v) == (v_0z - v_2z)^2 * k_stiff
    solver.A.vals.zero_()
    solver.contact.assemble(q, solver.A)
    A_pair = solver.A.to_scipy()
    v = rng.normal(size=solver.num_dofs)
    v_wp = wp.array(v, dtype=wp.float64, device=device)
    dy_wp = wp.zeros(solver.num_dofs, dtype=wp.float64, device=device)
    solver.contact.two_sided_mul(q, v_wp, dy_wp)
    quad_full = float(v @ (A_pair @ v + dy_wp.numpy()))
    quad_exact = float((v[2] - v[8]) ** 2 * k_stiff)
    np.testing.assert_allclose(quad_full, quad_exact, rtol=1.0e-5)

    # 2. Common-mode external force f = 1.0 on both nodes 0 and 2 (m / h^2 = 1.0, k = 1e5):
    dt = 1.0
    for e in solver.elements:
        e.begin_step(state, dt)
    f_dofs = bind_dofs(solver.dofs, model, solver.mass, solver.fixed, solver._no_force, state, dt)
    solver._assemble(q, f_dofs)
    solver._linear.invalidate()
    b = np.zeros(solver.num_dofs, dtype=np.float64)
    b[[2, 8]] = 1.0
    solver._b.assign(b)
    solver._linear.solve(solver._b, solver._dq)
    dq_majorized = solver._dq.numpy().copy()
    # Majorized step is crushed by 1 / (1 + 2 k) ~ 5e-6:
    np.testing.assert_allclose(dq_majorized[[2, 8]], 1.0 / (1.0 + 2.0 * k_stiff), rtol=1.0e-4)

    # SPD PCG recovers exact rigid-body translation f / (m / h^2) = 1.0 to < 1e-5:
    solver._pcg(q)
    dq_pcg = solver._dq.numpy()
    np.testing.assert_allclose(dq_pcg[[2, 8]], [1.0, 1.0], rtol=1.0e-5)

    # 3. Asymmetric force f = [1.0, 0.0] exciting both common mode and differential mode (2 PCG sweeps):
    b[[2, 8]] = [1.0, 0.0]
    solver._b.assign(b)
    solver._linear.solve(solver._b, solver._dq)
    solver._pcg(q)
    dq_asym = solver._dq.numpy()
    diff = 0.5 / (1.0 + 2.0 * k_stiff)
    np.testing.assert_allclose(dq_asym[[2, 8]], [0.5 + diff, 0.5 - diff], rtol=1.0e-5)

