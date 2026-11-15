"""Physical, reset, multi-step adjoint FD, and FrameGraph tests for Phase 3 carried constitutive memory."""

from __future__ import annotations

import gc

import newton
import numpy as np
import pytest
import warp as wp
from fd import assert_close, jacobian
from rollouts import cloth, simulate
from utils.common import FrameGraph, capturable, grid

from dismech_newton import SolverDiSMech, flatten_state
from dismech_newton import solver as dn_solver
from dismech_newton.builder import register_custom_attributes, vec5f, vec6f
from dismech_newton.residual import ConstitutiveResidual, MemorySpec, ResidualSpec


def _build_rod(
    n: int = 8,
    length: float = 0.5,
    bend_k: float = 2.0e-2,
    stretch_k: float = 5.0e2,
    twist_k: float = 2.0e-2,
    bend_d: float = 2.0e-4,
    stretch_d: float = 1.0e-2,
    twist_d: float = 2.0e-4,
    dahl_eps_max: float = 0.0,
    dahl_tau: float = 0.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> newton.ModelBuilder:
    b = newton.ModelBuilder(gravity=gravity)
    register_custom_attributes(b)
    pts = np.column_stack([np.linspace(0.0, length, n), np.zeros(n), np.zeros(n)])
    SolverDiSMech.add_rod(
        b,
        rod=newton.Rod(pts, radius=0.006),
        stretch_stiffness=stretch_k,
        stretch_damping=stretch_d,
        bend_stiffness=bend_k,
        bend_damping=bend_d,
        twist_stiffness=twist_k,
        twist_damping=twist_d,
        dahl_eps_max=dahl_eps_max,
        dahl_tau=dahl_tau,
        collide=False,
        proxies=False,
    )
    SolverDiSMech.fix_segment(b, edge=0)
    return b


def _build_shell(
    nx: int = 4,
    ny: int = 3,
    lx: float = 0.3,
    ly: float = 0.2,
    bend_k: float = 2.0e-2,
    tri_ke: float = 40.0,
    tri_ka: float = 40.0,
    tri_kd: float = 0.0,
    damping: float = 5.0e-3,
    clamp_cols: int = 1,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> newton.ModelBuilder:
    b = newton.ModelBuilder(gravity=gravity)
    register_custom_attributes(b)
    v, f = grid(nx, ny, lx, ly)
    p = cloth(
        b,
        v,
        f,
        density=1.0,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        tri_kd=tri_kd,
        bend_stiffness=bend_k,
        damping=damping,
    )
    clamp_x = (clamp_cols - 1) * lx / max(nx - 1, 1) + 1.0e-6
    for idx in np.flatnonzero(v[:, 0] <= clamp_x):
        b.particle_flags[int(p[idx])] &= ~newton.ParticleFlags.ACTIVE
    return b


def _finalize_worlds(sub: newton.ModelBuilder, worlds: int, device: str, gravity=(0.0, 0.0, -9.81)) -> newton.Model:
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


def _make_model(elem_kind: str, worlds: int, device: str, **kw) -> newton.Model:
    sub = _build_rod(**kw) if elem_kind == "rod" else _build_shell(**kw)
    return _finalize_worlds(sub, worlds, device, gravity=kw.get("gravity", (0.0, 0.0, -9.81)))


def _mem_spec(elem_kind: str, mode: str, **kw) -> ResidualSpec:
    dims = kw.pop("enabled_dims", (2, 3, 4) if elem_kind == "rod" else (3, 4, 5))
    ms = MemorySpec(mode=mode, enabled_dims=dims, **kw)
    return ResidualSpec(
        rod_rff_dim=16,
        shell_rff_dim=16,
        memory=ms if elem_kind == "rod" else None,
        shell_memory=ms if elem_kind == "shell" else None,
    )


def _mem_np(solver: SolverDiSMech, state: newton.State, elem_kind: str) -> np.ndarray:
    attr = solver.triplets.memory if elem_kind == "rod" else solver.shells.memory
    return getattr(state.dismech, attr).numpy()


@wp.kernel
def _mem_loss_kernel(
    q: wp.array[float],
    q_ref: wp.array[float],
    mem_rod: wp.array[vec5f],
    mem_rod_ref: wp.array[vec5f],
    mem_sh: wp.array[vec6f],
    mem_sh_ref: wp.array[vec6f],
    use_rod: int,
    use_sh: int,
    w_h: float,
    loss: wp.array[float],
):
    i = wp.tid()
    if i < q.shape[0]:
        dq = q[i] - q_ref[i]
        wp.atomic_add(loss, 0, 0.5 * dq * dq)
    if use_rod != 0 and i < mem_rod.shape[0]:
        dhr = mem_rod[i] - mem_rod_ref[i]
        wp.atomic_add(loss, 0, 0.5 * w_h * wp.dot(dhr, dhr))
    if use_sh != 0 and i < mem_sh.shape[0]:
        dhs = mem_sh[i] - mem_sh_ref[i]
        wp.atomic_add(loss, 0, 0.5 * w_h * wp.dot(dhs, dhs))


def test_memory_spec_validation_and_dahl_parity(device):
    sub = _build_rod(n=6, dahl_eps_max=0.02, dahl_tau=0.05)
    model_dahl = _finalize_worlds(sub, 1, device)
    model_sh = _finalize_worlds(_build_shell(), 1, device)

    for bad_kw in (
        {"memory": MemorySpec(mode="unknown")},
        {"memory": MemorySpec(mode="elastoplastic", yield_eps=-0.1)},
        {"memory": MemorySpec(mode="elastoplastic", yield_delta=0.0)},
        {"memory": MemorySpec(mode="elastoplastic", enabled_dims=(5,))},
        {"shell_memory": MemorySpec(mode="dahl")},
        {"shell_memory": MemorySpec(mode="prony", enabled_dims=(6,))},
    ):
        with pytest.raises(ValueError):
            ConstitutiveResidual(model_sh, ResidualSpec(**bad_kw), device=device)

    with pytest.raises(ValueError):
        SolverDiSMech(model_dahl, residual=ResidualSpec(memory=MemorySpec(mode="elastoplastic")))

    # MemorySpec(mode="dahl") must remain 100% bit-identical to baseline Rods Dahl solver.
    s_base = SolverDiSMech(model_dahl, residual=None)
    s_dahl = SolverDiSMech(model_dahl, residual=ResidualSpec(rod_rff_dim=16, memory=MemorySpec(mode="dahl")))
    fin_b, _ = simulate(model_dahl, s_base, 15, 0.01)
    fin_d, _ = simulate(model_dahl, s_dahl, 15, 0.01)
    assert np.array_equal(fin_b.particle_q.numpy(), fin_d.particle_q.numpy())
    assert np.array_equal(fin_b.dismech.triplet_mem_q.numpy(), fin_d.dismech.triplet_mem_q.numpy())
    assert np.array_equal(fin_b.dismech.triplet_strain_q.numpy(), fin_d.dismech.triplet_strain_q.numpy())


@pytest.mark.parametrize("elem_kind", ["rod", "shell"])
def test_elastoplastic_zero_leakage(device, elem_kind: str):
    """50 cyclic bends strictly below yield_eps produce exact zero h_t and zero drift vs elastic."""
    model = _make_model(elem_kind, 1, device, gravity=(0.0, 0.0, 0.0))
    spec = _mem_spec(elem_kind, "elastoplastic", yield_eps=0.5, yield_delta=1.0e-3, flow=0.5)
    s_el = SolverDiSMech(model, residual=None)
    s_ep = SolverDiSMech(model, residual=spec)

    dt, steps = 0.01, 50
    f0, f1 = model.state(), model.state()
    m0, m1 = model.state(), model.state()
    flatten_state(f0)
    flatten_state(f1)
    flatten_state(m0)
    flatten_state(m1)

    max_mem = 0.0
    max_strain_diff = 0.0
    max_q_diff = 0.0
    for k in range(steps):
        gz = (4.0 if elem_kind == "rod" else 0.8) * np.sin(2.0 * np.pi * k / 20.0)
        model.gravity.assign((0.0, 0.0, gz))
        s_el.step(f0, f1, None, None, dt)
        s_ep.step(m0, m1, None, None, dt)
        h_np = _mem_np(s_ep, m1, elem_kind)
        max_mem = max(max_mem, float(np.max(np.abs(h_np))))
        max_q_diff = max(max_q_diff, float(np.max(np.abs(m1.particle_q.numpy() - f1.particle_q.numpy()))))
        el = s_ep.triplets if elem_kind == "rod" else s_ep.shells
        er = getattr(m1.dismech, el.stored).numpy() - el.rest.numpy()
        dims = (spec.memory if elem_kind == "rod" else spec.shell_memory).enabled_dims
        max_strain_diff = max(max_strain_diff, float(np.max(np.linalg.norm(er[:, dims], axis=-1))))
        f0, f1 = f1, f0
        m0, m1 = m1, m0

    assert max_strain_diff > 1.0e-4, "Test should actually deform elements below yield"
    assert max_strain_diff < 0.25, "Deformation must stay well below yield_eps=0.5"
    assert max_mem < 1.0e-12 and max_mem == 0.0
    assert max_q_diff < 1.0e-6


@pytest.mark.parametrize("elem_kind", ["rod", "shell"])
def test_prony_relaxation_theory(device, elem_kind: str):
    """Prony memory relaxation & restoring stress match analytical Maxwell exponential ODE (<= 2% error)."""
    sub = (
        _build_rod(n=3, length=0.4, bend_k=0.05)
        if elem_kind == "rod"
        else _build_shell(nx=2, ny=2, lx=0.2, ly=0.2, bend_k=0.05)
    )
    # Fix all nodes at a prescribed non-zero bent/deformed configuration so eps_0 is constant in time.
    for i in range(sub.particle_count):
        sub.particle_flags[i] &= ~newton.ParticleFlags.ACTIVE
    if elem_kind == "rod":
        SolverDiSMech.fix_segment(sub, edge=1)
    model = _finalize_worlds(sub, 1, device, gravity=(0.0, 0.0, 0.0))

    tau, k_mem, dt, steps = 0.08, 0.4, 0.01, 30
    dims_kw = {"enabled_dims": (0, 1, 2)} if elem_kind == "shell" else {}
    spec = _mem_spec(elem_kind, "prony", k_mem=k_mem, tau=tau, **dims_kw)
    solver = SolverDiSMech(model, residual=spec)
    el = solver.triplets if elem_kind == "rod" else solver.shells
    dims = spec.memory.enabled_dims if elem_kind == "rod" else spec.shell_memory.enabled_dims

    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    pq = s0.particle_q.numpy().copy()
    if elem_kind == "rod":
        pq[-1, 2] += 0.05
    else:
        pq[-1, 0] += 0.02
        pq[-1, 2] += 0.02
    s0.particle_q.assign(pq)
    s1.particle_q.assign(pq)
    solver.reset(s0)

    k0 = el.params.numpy()[:, :el.rest.numpy().shape[1]]
    r0 = el.rest.numpy()
    p_step = wp.empty_like(el.params)
    r_step = wp.empty_like(el.rest)

    # First step populates eps_0 in stored strain.
    solver.step(s0, s1, None, None, dt)
    eps0 = getattr(s1.dismech, el.stored).numpy()
    de0 = eps0[:, dims] - r0[:, dims]
    assert np.max(np.abs(de0)) > 1.0e-3

    # Verify h_n and step restoring stress F_n across n = 1..steps against exact analytical Maxwell ODE.
    max_h_rel = 0.0
    max_f_rel = 0.0
    h_n = _mem_np(solver, s1, elem_kind)[:, dims]
    max_h_rel = max(max_h_rel, float(np.linalg.norm(h_n - (1.0 - np.exp(-dt / tau)) * de0) / np.linalg.norm(de0)))
    s0, s1 = s1, s0

    for n in range(1, steps):
        el.effective(s0.dismech, dt, p_step, r_step)
        kp_eff = p_step.numpy()[:, dims]
        rp_eff = r_step.numpy()[:, dims]
        f_sim = kp_eff * (eps0[:, dims] - rp_eff)
        t_next = (n + 1) * dt
        f_exact = k0[:, dims] * (1.0 + k_mem * np.exp(-t_next / tau)) * de0
        max_f_rel = max(max_f_rel, float(np.linalg.norm(f_sim - f_exact) / np.linalg.norm(f_exact)))

        solver.step(s0, s1, None, None, dt)
        h_n = _mem_np(solver, s1, elem_kind)[:, dims]
        h_exact = (1.0 - np.exp(-t_next / tau)) * de0
        max_h_rel = max(max_h_rel, float(np.linalg.norm(h_n - h_exact) / np.linalg.norm(de0)))
        s0, s1 = s1, s0

    assert max_h_rel <= 0.02
    assert max_f_rel <= 0.02


@pytest.mark.parametrize("elem_kind", ["rod", "shell"])
def test_elastoplastic_permanent_set(device, elem_kind: str):
    """Above-yield loading then unloading leaves permanent plastic set h_t != 0 and shifted rest shape."""
    kw = (
        {"bend_k": 0.15, "bend_d": 0.04, "gravity": (0.0, 0.0, -45.0)}
        if elem_kind == "rod"
        else {
            "nx": 5,
            "ny": 3,
            "lx": 0.4,
            "ly": 0.2,
            "tri_ke": 1.0e3,
            "tri_ka": 1.0e3,
            "tri_kd": 1.0,
            "bend_k": 0.05,
            "damping": 0.05,
            "clamp_cols": 2,
            "gravity": (0.0, 0.0, -30.0),
        }
    )
    model = _make_model(elem_kind, 1, device, **kw)
    spec = _mem_spec(
        elem_kind,
        "elastoplastic",
        yield_eps=0.008 if elem_kind == "rod" else 0.025,
        yield_delta=5.0e-4,
        flow=0.6 if elem_kind == "rod" else 0.5,
    )
    s_el = SolverDiSMech(model, self_contact=False, residual=None)
    s_ep = SolverDiSMech(model, self_contact=False, residual=spec)
    dt = 0.01

    f0, f1 = model.state(), model.state()
    m0, m1 = model.state(), model.state()
    for s in (f0, f1, m0, m1):
        flatten_state(s)

    # Load well above yield, then unload with damping to settle at zero gravity.
    model.gravity.assign(kw["gravity"])
    for _ in range(35):
        s_el.step(f0, f1, None, None, dt)
        s_ep.step(m0, m1, None, None, dt)
        f0, f1 = f1, f0
        m0, m1 = m1, m0

    model.gravity.assign((0.0, 0.0, 0.0))
    for _ in range(200):
        s_el.step(f0, f1, None, None, dt)
        s_ep.step(m0, m1, None, None, dt)
        f0, f1 = f1, f0
        m0, m1 = m1, m0

    q_rest = model.particle_q.numpy()
    h_final = _mem_np(s_ep, m0, elem_kind)
    ep_shift = float(np.max(np.abs(m0.particle_q.numpy() - q_rest)))
    el_shift = float(np.max(np.abs(f0.particle_q.numpy() - q_rest)))
    assert np.max(np.abs(h_final)) > 1.0e-3
    assert ep_shift > el_shift + 1.0e-2


@pytest.mark.parametrize("elem_kind", ["rod", "shell"])
def test_bouc_wen_hysteresis_bounded(device, elem_kind: str):
    """Bouc-Wen carried state remains strictly bounded in [-bw_sat, bw_sat] and dissipates positive area."""
    bw_sat, k_mem = 0.015, 0.45
    model = _make_model(elem_kind, 1, device, bend_k=2.0e-2, gravity=(0.0, 0.0, 0.0))
    spec = _mem_spec(elem_kind, "bouc_wen", k_mem=k_mem, flow=5.0, bw_sat=bw_sat, yield_eps=0.0, yield_delta=5.0e-4)
    solver = SolverDiSMech(model, residual=spec)
    el = solver.triplets if elem_kind == "rod" else solver.shells
    dims = spec.memory.enabled_dims if elem_kind == "rod" else spec.shell_memory.enabled_dims

    dt, steps = 0.01, 80
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)

    eps_hist, h_hist = [], []
    max_h = 0.0
    for k in range(steps):
        gz = 25.0 * np.sin(2.0 * np.pi * k / 40.0)
        model.gravity.assign((0.0, 0.0, gz))
        solver.step(s0, s1, None, None, dt)
        h_np = _mem_np(solver, s1, elem_kind)[:, dims]
        eps_np = getattr(s1.dismech, el.stored).numpy()[:, dims] - el.rest.numpy()[:, dims]
        max_h = max(max_h, float(np.max(np.abs(h_np))))
        assert np.all(np.isfinite(h_np))
        if k >= 40:  # Second full steady cycle for closed hysteresis loop area.
            eps_hist.append(eps_np.copy())
            h_hist.append(h_np.copy())
        s0, s1 = s1, s0

    assert max_h > 0.25 * bw_sat
    assert max_h <= bw_sat + 1.0e-6
    eps_arr, h_arr = np.asarray(eps_hist), np.asarray(h_hist)
    area = float(np.sum(0.5 * (h_arr[1:] + h_arr[:-1]) * (eps_arr[1:] - eps_arr[:-1])))
    assert area > 1.0e-5


@pytest.mark.parametrize("elem_kind", ["rod", "shell"])
def test_spd_clamping_with_memory(device, elem_kind: str):
    """Composed elastic + memory step tangents remain strictly within [1 + clamp_lo, 1 + clamp_hi] * K_0."""
    model = _make_model(elem_kind, 1, device)
    rng = np.random.default_rng(7)
    dim = 5 if elem_kind == "rod" else 6
    w_large = 50.0 * rng.standard_normal((16, dim)).astype(np.float32)

    for mode in ("elastoplastic", "prony", "bouc_wen"):
        ms = MemorySpec(
            mode=mode,
            enabled_dims=tuple(range(dim)),
            k_mem=12.0,
            flow=5.0,
            tau=0.05,
            bw_sat=0.05,
            yield_eps=0.0,
            yield_delta=1.0e-4,
        )
        spec = ResidualSpec(
            rod_rff_dim=16,
            shell_rff_dim=16,
            clamp_lo=-0.35,
            clamp_hi=8.0,
            w_rod=w_large if elem_kind == "rod" else None,
            w_shell=w_large if elem_kind == "shell" else None,
            memory=ms if elem_kind == "rod" else None,
            shell_memory=ms if elem_kind == "shell" else None,
        )
        solver = SolverDiSMech(model, residual=spec)
        el = solver.triplets if elem_kind == "rod" else solver.shells
        s0, s1 = model.state(), model.state()
        flatten_state(s0)
        flatten_state(s1)
        solver.step(s0, s1, None, None, 0.01)
        p_eff, _ = el.effective(s1.dismech, 0.01)
        k0 = el.params.numpy()[:, :dim]
        ks = p_eff.numpy()[:, :dim]
        pos = k0 > 0.0
        assert np.all(ks[pos] >= (1.0 + spec.clamp_lo - 1.0e-5) * k0[pos])
        assert np.all(ks[pos] <= (1.0 + spec.clamp_hi + 1.0e-5) * k0[pos])
        assert np.all(ks[pos] >= 0.65 * k0[pos] - 1.0e-5)


def test_masked_reset_rod_and_shell(device):
    """Masked solver.reset clears ONLY masked worlds' triplet_mem_q and face_mem_q."""
    sub = newton.ModelBuilder(gravity=(0.0, 0.0, -25.0))
    register_custom_attributes(sub)
    pts = np.column_stack([np.linspace(0.0, 0.4, 7), np.zeros(7), np.zeros(7)])
    SolverDiSMech.add_rod(
        sub,
        rod=newton.Rod(pts, radius=0.006),
        stretch_stiffness=5.0e2,
        bend_stiffness=1.0e-2,
        twist_stiffness=1.0e-2,
        bend_damping=2.0e-4,
        collide=False,
        proxies=False,
    )
    SolverDiSMech.fix_segment(sub, edge=0)
    v, f = grid(3, 3, 0.25, 0.2)
    p = cloth(
        sub,
        v + np.array([0.0, 0.5, 0.0]),
        f,
        density=1.0,
        tri_ke=40.0,
        tri_ka=40.0,
        bend_stiffness=1.0e-2,
        damping=5.0e-3,
    )
    for idx in p[:3]:
        sub.particle_flags[int(idx)] &= ~newton.ParticleFlags.ACTIVE

    model = _finalize_worlds(sub, 2, device, gravity=(0.0, 0.0, -25.0))
    spec = ResidualSpec(
        rod_rff_dim=16,
        shell_rff_dim=16,
        memory=MemorySpec(mode="elastoplastic", yield_eps=0.005, yield_delta=5.0e-4, flow=0.5, enabled_dims=(2, 3, 4)),
        shell_memory=MemorySpec(
            mode="elastoplastic", yield_eps=0.005, yield_delta=5.0e-4, flow=0.5, enabled_dims=(3, 4, 5)
        ),
    )
    solver = SolverDiSMech(model, residual=spec)
    dt = 0.01

    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    for _ in range(15):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0

    # Clone unmasked continuation reference, then masked-reset world 0 only.
    u0, u1 = model.state(), model.state()
    flatten_state(u0)
    flatten_state(u1)
    u0.assign(s0)

    pw = model.particle_world.numpy()
    rw = solver.residual.elem_world.numpy()
    sw = solver.residual.shell_elem_world.numpy()

    tm_before = s0.dismech.triplet_mem_q.numpy().copy()
    fm_before = s0.dismech.face_mem_q.numpy().copy()
    assert np.max(np.abs(tm_before[rw == 0])) > 1.0e-3 and np.max(np.abs(tm_before[rw == 1])) > 1.0e-3
    assert np.max(np.abs(fm_before[sw == 0])) > 1.0e-3 and np.max(np.abs(fm_before[sw == 1])) > 1.0e-3

    x, v_arr = s0.particle_q.numpy(), s0.particle_qd.numpy()
    x[pw == 0] = model.particle_q.numpy()[pw == 0]
    v_arr[pw == 0] = 0.0
    s0.particle_q.assign(x)
    s0.particle_qd.assign(v_arr)
    mask = wp.array([True, False, False], dtype=wp.bool, device=device)
    solver.reset(s0, mask)

    tm_after = s0.dismech.triplet_mem_q.numpy()
    fm_after = s0.dismech.face_mem_q.numpy()
    assert np.max(np.abs(tm_after[rw == 0])) == 0.0
    assert np.max(np.abs(fm_after[sw == 0])) == 0.0
    assert np.array_equal(tm_after[rw == 1], tm_before[rw == 1])
    assert np.array_equal(fm_after[sw == 1], fm_before[sw == 1])

    # Continuing 10 steps: world 0 replays fresh start, world 1 matches unmasked continuation.
    f0, f1 = model.state(), model.state()
    flatten_state(f0)
    flatten_state(f1)
    for _ in range(10):
        solver.step(s0, s1, None, None, dt)
        solver.step(u0, u1, None, None, dt)
        solver.step(f0, f1, None, None, dt)
        s0, s1 = s1, s0
        u0, u1 = u1, u0
        f0, f1 = f1, f0

    np.testing.assert_allclose(s0.particle_q.numpy()[pw == 0], f0.particle_q.numpy()[pw == 0], atol=1.0e-6, rtol=1.0e-5)
    np.testing.assert_allclose(s0.particle_q.numpy()[pw == 1], u0.particle_q.numpy()[pw == 1], atol=1.0e-6, rtol=1.0e-5)
    np.testing.assert_allclose(
        s0.dismech.triplet_mem_q.numpy()[rw == 1], u0.dismech.triplet_mem_q.numpy()[rw == 1], atol=1.0e-6, rtol=1.0e-5
    )
    np.testing.assert_allclose(
        s0.dismech.face_mem_q.numpy()[sw == 1], u0.dismech.face_mem_q.numpy()[sw == 1], atol=1.0e-6, rtol=1.0e-5
    )


@pytest.mark.parametrize("worlds", [1, 2])
@pytest.mark.parametrize("mode", ["prony", "elastoplastic", "bouc_wen"])
@pytest.mark.parametrize("elem_kind", ["rod", "shell"])
def test_adjoint_fd_memory(device, monkeypatch, elem_kind: str, mode: str, worlds: int):
    """Multi-step StepAdjoint VJP through carried h_t wrt learnable memory params and initial (q0, h0)."""
    monkeypatch.setattr(dn_solver, "TOLERANCE", 1.0e-7)
    monkeypatch.setattr(dn_solver, "ULP", 1.0e-8)
    gz = -40.0 if mode == "elastoplastic" else -18.0
    kw = (
        {"stretch_k": 40.0, "bend_k": 0.08, "twist_k": 0.08, "gravity": (0.0, 0.0, gz)}
        if elem_kind == "rod"
        else {
            "tri_ke": 200.0,
            "tri_ka": 200.0,
            "tri_kd": 0.3,
            "bend_k": 0.08 if mode == "elastoplastic" else 0.15,
            "damping": 0.05,
            "clamp_cols": 2,
            "gravity": (0.0, 0.0, gz),
        }
    )
    model = _make_model(elem_kind, worlds, device, **kw)
    dim = 5 if elem_kind == "rod" else 6
    dims = (2, 3, 4) if elem_kind == "rod" else (3, 4, 5)

    def _p(base_val: float):
        if worlds == 1:
            return base_val
        arr = np.full((worlds, dim), base_val, dtype=np.float32)
        arr[1] *= 1.3
        return arr

    ye = (0.08 if elem_kind == "rod" else 0.04) if mode == "elastoplastic" else (0.002 if mode == "bouc_wen" else 0.0)
    ms = MemorySpec(
        mode=mode,
        enabled_dims=dims,
        yield_eps=ye,
        yield_delta=5.0e-4,
        k_mem=_p(0.35),
        flow=_p(0.55 if mode == "elastoplastic" else 1.8),
        tau=_p(0.05),
        bw_sat=_p(0.08),
        yield_weight=_p(1.1),
    )
    spec = ResidualSpec(
        rod_rff_dim=8,
        shell_rff_dim=8,
        memory=ms if elem_kind == "rod" else None,
        shell_memory=ms if elem_kind == "shell" else None,
    )
    solver = SolverDiSMech(model, iterations=50, exact=True, self_contact=False, residual=spec)

    dt, steps = 0.01, 3
    states = [model.state(requires_grad=True) for _ in range(steps + 1)]
    for s in states:
        flatten_state(s)

    # Pre-deform initial configuration across all bend/twist axes & seed h_0 away from kinks.
    q0_np = model.particle_q.numpy().copy()
    if elem_kind == "rod":
        x_rel = q0_np[:, 0] / max(float(q0_np[:, 0].max()), 1.0e-6)
        q0_np[:, 1] += 0.025 * x_rel * x_rel
        q0_np[:, 2] += 0.06 * x_rel * x_rel
        edge0 = states[0].dismech.edge_q.numpy() + 0.04 * np.linspace(0.0, 1.0, states[0].dismech.edge_q.shape[0])
        hinge0 = states[0].dismech.hinge_q.numpy()
    else:
        x_rel = q0_np[:, 0] / max(float(q0_np[:, 0].max()), 1.0e-6)
        y_rel = q0_np[:, 1] / max(float(q0_np[:, 1].max()), 1.0e-6)
        q0_np[:, 2] += 0.04 * x_rel * x_rel + 0.02 * x_rel * y_rel
        edge0 = states[0].dismech.edge_q.numpy()
        hinge0 = states[0].dismech.hinge_q.numpy() + 0.02
    q0_flat = np.concatenate([q0_np.ravel(), edge0, hinge0])
    states[0].dismech.q.assign(q0_flat.astype(np.float32))

    mem_name = "triplet_mem_q" if elem_kind == "rod" else "face_mem_q"
    h0_arr = getattr(states[0].dismech, mem_name)
    h0_np = np.zeros(h0_arr.numpy().shape, dtype=np.float32)
    h0_np[:, dims] = 0.015 if mode == "bouc_wen" else -0.008
    h0_arr.assign(h0_np)

    # Center loss targets near the unperturbed rollout end-state to minimize float32 ULP quantization noise.
    solver.reset(states[0])
    for k in range(steps):
        solver.step(states[k], states[k + 1], None, None, dt)
    rng_ref = np.random.default_rng(7)
    end_q = states[-1].dismech.q.numpy()
    q_ref_np = (end_q + rng_ref.normal(0.0, 0.015, end_q.shape)).astype(np.float32)
    q_ref = wp.array(q_ref_np, dtype=float, device=device, requires_grad=False)
    if elem_kind == "rod":
        end_mr = states[-1].dismech.triplet_mem_q.numpy()
        m_ref_np = (end_mr + rng_ref.normal(0.0, 0.01, end_mr.shape)).astype(np.float32)
        mr_ref = wp.array(m_ref_np, dtype=vec5f, device=device)
        ms_ref = wp.zeros(max(solver.shells.count, 1), dtype=vec6f, device=device)
    else:
        mr_ref = wp.zeros(max(solver.triplets.count, 1), dtype=vec5f, device=device)
        end_ms = states[-1].dismech.face_mem_q.numpy()
        m_ref_np = (end_ms + rng_ref.normal(0.0, 0.01, end_ms.shape)).astype(np.float32)
        ms_ref = wp.array(m_ref_np, dtype=vec6f, device=device)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    mr_end = states[-1].dismech.triplet_mem_q if elem_kind == "rod" else mr_ref
    ms_end = states[-1].dismech.face_mem_q if elem_kind == "shell" else ms_ref

    def _run_forward(record_tape: bool) -> float:
        solver.reset(states[0])
        if record_tape:
            for st in states:
                for attr in ("q", "qd", "triplet_strain_q", "face_strain_q", "triplet_mem_q", "face_mem_q",
                             "edge_d1_q", "triplet_ref_twist_q"):
                    arr = getattr(st.dismech, attr, None)
                    if arr is not None and arr.grad is not None:
                        arr.grad.zero_()
            for arr in solver.residual.grad_inputs():
                if arr.grad is not None:
                    arr.grad.zero_()
            loss.zero_()
            loss.grad.zero_()
            with wp.Tape() as tape:
                for k in range(steps):
                    solver.step(states[k], states[k + 1], None, None, dt)
                wp.launch(
                    _mem_loss_kernel,
                    dim=max(q_ref.shape[0], mr_ref.shape[0], ms_ref.shape[0]),
                    inputs=[
                        states[-1].dismech.q,
                        q_ref,
                        mr_end,
                        mr_ref,
                        ms_end,
                        ms_ref,
                        int(elem_kind == "rod"),
                        int(elem_kind == "shell"),
                        100.0,
                    ],
                    outputs=[loss],
                    device=device,
                )
            tape.backward(loss=loss)
            return float(loss.numpy()[0])
        for k in range(steps):
            solver.step(states[k], states[k + 1], None, None, dt)
        dq = states[-1].dismech.q.numpy().astype(np.float64) - q_ref_np.astype(np.float64)
        dm = getattr(states[-1].dismech, mem_name).numpy().astype(np.float64) - m_ref_np.astype(np.float64)
        return float(0.5 * np.sum(dq * dq) + 50.0 * np.sum(dm * dm))

    _run_forward(record_tape=True)
    mem_obj = solver.residual.mem_rod if elem_kind == "rod" else solver.residual.mem_shell
    check_names = {
        "prony": ("k_mem", "tau"),
        "elastoplastic": ("flow", "yield_weight"),
        "bouc_wen": ("k_mem", "flow", "bw_sat"),
    }[mode]

    targets: list[tuple[str, wp.array, np.ndarray, float]] = [
        ("q0", states[0].dismech.q, states[0].dismech.q.grad.numpy().copy(), 5.0e-5),
        ("h0", h0_arr, h0_arr.grad.numpy().copy(), 5.0e-4),
    ]
    for pname in check_names:
        arr = getattr(mem_obj, pname)
        step_p = (
            5.0e-3
            if (pname == "yield_weight" and elem_kind == "rod")
            else (
                8.0e-2
                if (pname == "yield_weight" and elem_kind == "shell")
                else (
                    2.0e-3
                    if pname in ("tau", "bw_sat")
                    else (5.0e-2 if (elem_kind == "shell" and mode == "bouc_wen" and pname == "flow") else 2.5e-2)
                )
            )
        )
        targets.append((pname, arr, arr.grad.numpy().copy(), step_p))

    for name, arr, grad_ad, step_h in targets:
        orig = arr.numpy().copy()
        flat_g = grad_ad.reshape(-1)
        gnorm = float(np.linalg.norm(flat_g))
        if gnorm == 0.0:
            raise AssertionError(f"Expected non-zero AD gradient for {elem_kind}/{mode}/W={worlds}/{name}")
        if orig.size > 12:
            d = (grad_ad / max(gnorm, 1.0e-12)).astype(np.float32)

            def _f_dir(ts: np.ndarray) -> np.ndarray:
                vals = []
                for t_val in np.asarray(ts).reshape(-1):
                    arr.assign((orig + float(t_val) * d).astype(np.float32))
                    vals.append([_run_forward(False)])
                return np.asarray(vals, dtype=np.float64)

            fd_g = jacobian(_f_dir, np.zeros(1), step_h).reshape(-1)
            arr.assign(orig)
            ad_dir = float(np.sum(d * grad_ad))
            assert_close(
                np.array([ad_dir]),
                fd_g,
                rtol=5.0e-3,
                name=f"{elem_kind}/{mode}/W={worlds}/{name}",
                scale=max(abs(ad_dir), 1.0e-6),
            )
        else:
            mask = np.zeros(orig.shape, dtype=bool)
            mask[:, dims] = True
            active = mask.reshape(-1)
            assert np.all(flat_g[~active] == 0.0)
            base_flat = orig.reshape(-1).astype(np.float64)

            def _f_vec(xs: np.ndarray) -> np.ndarray:
                vals = []
                for row in np.asarray(xs):
                    vec = base_flat.copy()
                    vec[active] = row
                    arr.assign(vec.reshape(orig.shape).astype(np.float32))
                    vals.append([_run_forward(False)])
                return np.asarray(vals, dtype=np.float64)

            fd_g = jacobian(_f_vec, base_flat[active], step_h).reshape(-1)
            arr.assign(orig)
            assert_close(
                flat_g[active].astype(np.float64),
                fd_g,
                rtol=5.0e-3,
                name=f"{elem_kind}/{mode}/W={worlds}/{name}",
            )


@pytest.mark.usefixtures("needs_cudss")
@pytest.mark.parametrize("elem_kind", ["rod", "shell"])
def test_framegraph_w256_memory_overhead(device, elem_kind: str):
    """W=256 FrameGraph with carried memory captures cleanly on CUDA and has <= +2.5% overhead."""
    if elem_kind == "rod":
        sub = _build_rod(n=25, length=1.0)
    else:
        sub = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        register_custom_attributes(sub)
        v, f = grid(5, 5, 0.4, 0.4)
        ids = cloth(
            sub, v, f, density=1.0, tri_ke=2.0e3, tri_ka=2.0e3, tri_kd=0.05, bend_stiffness=0.08, damping=1.0e-3
        )
        for k in np.flatnonzero(v[:, 0] <= 0.1 + 1e-6):
            sub.particle_flags[int(ids[k])] &= ~newton.ParticleFlags.ACTIVE
    model = _finalize_worlds(sub, 256, device)
    ms = MemorySpec(
        mode="elastoplastic",
        yield_eps=0.01,
        yield_delta=5.0e-4,
        flow=0.5,
        enabled_dims=(2, 3, 4) if elem_kind == "rod" else (3, 4, 5),
    )
    spec = ResidualSpec(
        rod_rff_dim=64,
        shell_rff_dim=64,
        memory=ms if elem_kind == "rod" else None,
        shell_memory=ms if elem_kind == "shell" else None,
    )

    dt, substeps, warm, iters, rounds = 0.01, 4, 15, 20, 7
    s0 = model.state()
    flatten_state(s0)
    copy_attrs = (
        "q",
        "qd",
        "edge_d1_q",
        "triplet_ref_twist_q",
        "triplet_strain_q",
        "face_strain_q",
        "triplet_mem_q",
        "face_mem_q",
    )
    stream = wp.get_stream(model.device)
    e0 = wp.Event(device=model.device, enable_timing=True)
    e1 = wp.Event(device=model.device, enable_timing=True)

    def _compile_and_warm(solver):
        a, b = model.state(), model.state()
        flatten_state(a)
        flatten_state(b)

        def _reset_a():
            wp.copy(a.particle_q, s0.particle_q)
            wp.copy(a.particle_qd, s0.particle_qd)
            for attr in copy_attrs:
                dst, src = getattr(a.dismech, attr, None), getattr(s0.dismech, attr, None)
                if dst is not None and src is not None:
                    wp.copy(dst, src)
            solver.reset(a)

        def work():
            for _ in range(substeps // 2):
                solver.step(a, b, None, None, dt)
                solver.step(b, a, None, None, dt)

        fg = FrameGraph(work, solver, enabled=True)
        _reset_a()
        for _ in range(warm):
            fg()
        assert fg.graph is not None
        wp.synchronize()
        return fg, a, _reset_a

    def _time(fg, reset_fn) -> float:
        reset_fn()
        wp.synchronize()
        stream.record_event(e0)
        for _ in range(iters):
            fg()
        stream.record_event(e1)
        wp.synchronize()
        return float(wp.get_event_elapsed_time(e0, e1)) / (substeps * iters)

    gc.collect()
    s_base = SolverDiSMech(model, self_contact=False, residual=None)
    s_mem = SolverDiSMech(model, self_contact=False, residual=spec)
    assert capturable(s_mem)
    fg_base, _, rst_base = _compile_and_warm(s_base)
    fg_mem, q_mem_st, rst_mem = _compile_and_warm(s_mem)
    round_pairs = []
    for r in range(rounds):
        if r % 2 == 0:
            tb = _time(fg_base, rst_base)
            tm = _time(fg_mem, rst_mem)
        else:
            tm = _time(fg_mem, rst_mem)
            tb = _time(fg_base, rst_base)
        round_pairs.append((tb, tm))
    q_mem_np = q_mem_st.particle_q.numpy()
    mem_arr = _mem_np(s_mem, q_mem_st, elem_kind)
    ms_base = float(np.median([tb for tb, _ in round_pairs]))
    ms_mem = float(np.median([tm for _, tm in round_pairs]))
    overhead_pct = float(np.median([100.0 * (tm - tb) / tb for tb, tm in round_pairs]))
    print(
        f"\n[{elem_kind} W=256 FrameGraph] base={ms_base:.4f} ms/step, "
        f"mem={ms_mem:.4f} ms/step, overhead={overhead_pct:+.2f}%"
    )
    assert np.all(np.isfinite(q_mem_np))
    assert np.all(np.isfinite(mem_arr))
    assert float(np.max(np.abs(mem_arr))) > 1e-4
    assert overhead_pct <= 2.5
