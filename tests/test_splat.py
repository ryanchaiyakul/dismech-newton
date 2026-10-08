"""The splat skin against a float64 NumPy reference, the proxies, finite differences and its invariants."""

import newton
import numpy as np
import pytest
import warp as wp
from fd import assert_close, derivative
from scipy.spatial.transform import Rotation

from dismech_newton import DiSMechSolver, add_rod, fix_segment, flatten_state
from dismech_newton.dofs import dof_constants
from dismech_newton.splat import SH_C0, Splats, edge_gaussians, export_usd, skin, tube

RADIUS, SIGMA = 0.02, (0.003, 0.008, 0.01)


# -- float64 NumPy reference of the skin -------------------------------------------------------

IU = np.triu_indices(3)  # cov6 order: xx xy xz yy yz zz


def frames_ref(q, d1, node0, node1):
    """Per edge ``(t, m1, m2)``, each ``(E, 3)``; ``q`` flat (nodes, then one twist per edge)."""
    E = len(node0)
    x = q[:-E].reshape(-1, 3)
    theta = q[-E:]
    t = x[node1] - x[node0]
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    d2 = np.cross(t, d1)
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    return t, c * d1 + s * d2, -s * d1 + c * d2


def skin_ref(q, d1, node0, node1, sp):
    """``(means (S, 3), cov6 (S, 6))``; ``sp``: :meth:`Splats.numpy`."""
    E = len(node0)
    x = q[:-E].reshape(-1, 3)
    t, m1, m2 = frames_ref(q, d1, node0, node1)
    e = sp.edge
    s = sp.s[:, None]
    mean = (1.0 - s) * x[node0[e]] + s * x[node1[e]] + sp.uv[:, :1] * m1[e] + sp.uv[:, 1:] * m2[e]
    R = np.stack([m1[e], m2[e], t[e]], axis=2) @ sp.rotation
    cov = (R * np.exp(2.0 * sp.log_scale)[:, None, :]) @ R.transpose(0, 2, 1)
    return mean, cov[:, IU[0], IU[1]]


def full(cov6):
    """``(..., 3, 3)`` from cov6."""
    c = np.zeros(cov6.shape[:-1] + (3, 3))
    c[..., IU[0], IU[1]] = cov6
    c[..., IU[1], IU[0]] = cov6
    return c


def turn_d1(d1, t, phi):
    """``d1`` turned about ``t`` by ``phi`` (per edge): the only perturbation of ``d1`` that keeps it a director."""
    return np.cos(phi)[:, None] * d1 + np.sin(phi)[:, None] * np.cross(t, d1)


def bent(proxies: bool = False):
    """A lone particle first (every rod index offset), a clamped rod and a free one, kicked and stepped: bent in
    3D, twisted unevenly (the clamp turned), ``d1`` transported by the solver."""
    builder = newton.ModelBuilder()
    builder.add_particle(wp.vec3(0.5, 0.5, 0.5), wp.vec3(0.0, 0.0, 0.0), 0.1, radius=RADIUS)
    rods = (((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), True), ((0.0, -0.3, 0.7), (0.6, 0.0, 0.8), False))
    for start, direction, clamped in rods:
        rod = newton.Rod.create_straight(start, direction, 0.5, segment_count=6, radius=RADIUS)
        ids = add_rod(builder, rod, stretch_stiffness=1.0e4, bend_stiffness=8.0, twist_stiffness=8.0, proxies=proxies)
        if clamped:
            fix_segment(builder, **({"body": ids[0]} if proxies else {"edge": ids[0]}))
    model = builder.finalize()
    solver = DiSMechSolver(model)
    a, b = model.state(), model.state()
    flatten_state(a)
    flatten_state(b)
    x = model.particle_q.numpy()
    v = np.zeros_like(x)
    v[:, 1], v[:, 2] = 0.8 * x[:, 0] ** 2, 0.3 * x[:, 0]
    spin = 20.0 * np.sin(np.arange(model.dismech.edge_length.shape[0]))
    fixed = dof_constants(model)[1].numpy() != 0
    a.dismech.qd.assign(np.where(fixed, 0.0, np.concatenate([v.ravel(), spin])))
    q = a.dismech.q.numpy()
    q[x.size] = 1.0  # the clamp (edge 0) turned
    a.dismech.q.assign(q)
    for _ in range(10):
        solver.step(a, b, None, None, 1.0 / 240.0)
        a, b = b, a
    return model, solver, a


@pytest.fixture
def scene():
    model, _, state = bent()
    splats, _ = tube(model, rings=2, per_ring=4, radius=RADIUS, sigma=SIGMA)
    d = model.dismech
    return dict(model=model, splats=splats, sp=splats.numpy(), q=state.dismech.q.numpy().astype(np.float64),
                d1=state.dismech.edge_d1_q.numpy().astype(np.float64), n0=d.edge_node0.numpy(),
                n1=d.edge_node1.numpy())


def run(sc, q, d1, wm=None, wc=None):
    """Forward (and, with weights, the adjoint of ``wm . means + wc . cov6``): outputs, grads."""
    grad = wm is not None
    qa = wp.array(q.astype(np.float32), dtype=float, requires_grad=grad)
    da = wp.array(d1.astype(np.float32), dtype=wp.vec3, requires_grad=grad)
    splats = sc["splats"]
    tape = wp.Tape()
    with tape:
        means, cov6 = skin(sc["model"], splats, qa, da)
    out = means.numpy().astype(np.float64), cov6.numpy().astype(np.float64)
    if not grad:
        return out, None
    tape.backward(grads={means: wp.array(wm.astype(np.float32), dtype=wp.vec3),
                         cov6: wp.array(wc.astype(np.float32), dtype=cov6.dtype)})
    g = {"q": qa.grad.numpy().astype(np.float64), "d1": da.grad.numpy().astype(np.float64)}
    return out, g


def test_skin_matches_reference(scene):
    sc = scene
    (means, cov6), _ = run(sc, sc["q"], sc["d1"])
    m, c = skin_ref(sc["q"], sc["d1"], sc["n0"], sc["n1"], sc["sp"])
    assert_close(means, m, 1.0e-6, "means", scale=1.0)
    assert_close(cov6, c, 1.0e-5, "cov6")


def test_skin_matches_proxies():
    """A centred splat with ``rotation = I`` sits on its edge's proxy: same position, same frame."""
    model, solver, state = bent(proxies=True)
    solver.update_proxies(state)
    E = model.dismech.edge_length.shape[0]
    log_scale = np.tile(np.log([0.004, 0.008, 0.02]), (E, 1))
    splats = Splats.from_numpy(np.arange(E), np.full(E, 0.5), np.zeros((E, 2)), np.tile(np.eye(3), (E, 1, 1)),
                               log_scale, device=model.device)
    means, cov6 = skin(model, splats, state.dismech.q, state.dismech.edge_d1_q)
    X = state.body_q.numpy()
    R = Rotation.from_quat(X[:, 3:]).as_matrix()
    assert_close(means.numpy(), X[:, :3], 1.0e-6, "position", scale=1.0)
    cov = (R * np.exp(2.0 * log_scale)[:, None, :]) @ R.transpose(0, 2, 1)
    assert_close(full(cov6.numpy()), cov, 1.0e-5, "covariance")


def test_skin_gradient_matches_finite_differences(scene, rng):
    """Nodes, twists, ``d1`` turned about ``t`` and each canonical parameter, against the float64 reference."""
    sc = scene
    q, d1, n0, n1, sp = sc["q"], sc["d1"], sc["n0"], sc["n1"], sc["sp"]
    S, E = len(sp.edge), len(n0)
    wm, wc = rng.normal(size=(S, 3)), rng.normal(size=(S, 6))
    splats = Splats.from_numpy(sp.edge, sp.s, sp.uv, sp.rotation, sp.log_scale, requires_grad=True)
    sc = dict(sc, splats=splats)
    _, g = run(sc, q, d1, wm, wc)

    def loss(q=q, d1=d1, **params):
        m, c = skin_ref(q, d1, n0, n1, type(sp)(**dict(vars(sp), **params)))
        return float(np.sum(wm * m) + np.sum(wc * c))

    t, _, _ = frames_ref(q, d1, n0, n1)
    checks = {}
    for name, mask in (("nodes", np.arange(q.size) < q.size - E), ("twists", np.arange(q.size) >= q.size - E)):
        dq = np.where(mask, rng.normal(size=q.size), 0.0)
        checks[name] = (g["q"] @ dq, derivative(lambda e: loss(q=q + e * dq), 1e-4), np.linalg.norm(g["q"][mask]))
    phi = rng.normal(size=E)
    g_turn = np.sum(g["d1"] * np.cross(t, d1), 1)
    checks["d1 about t"] = (g_turn @ phi, derivative(lambda e: loss(d1=turn_d1(d1, t, e * phi)), 1e-4),
                            np.linalg.norm(g_turn))
    for key in ("s", "uv", "rotation", "log_scale"):
        base = getattr(sp, key)
        dp = rng.normal(size=base.shape)
        gp = getattr(splats, key).grad.numpy().astype(np.float64)
        checks[key] = (np.sum(gp * dp), derivative(lambda e: loss(**{key: base + e * dp}), 1e-4),
                       np.linalg.norm(gp) * np.linalg.norm(dp))
    for name, (adjoint, fd, scale) in checks.items():
        assert_close(adjoint, fd, 1.0e-4, name, scale=scale)


def test_skin_invariants(scene, rng):
    """Translation, rotation, ``theta + 2 pi``, and the gauge: turning ``d1`` about ``t`` by ``phi`` with
    ``theta - phi`` changes nothing, so ``g_d1 . (t x d1) = g_theta`` per edge."""
    sc = scene
    q, d1, n0, n1 = sc["q"], sc["d1"], sc["n0"], sc["n1"]
    E, S = len(n0), len(sc["sp"].edge)
    wm, wc = rng.normal(size=(S, 3)), rng.normal(size=(S, 6))
    (m0, c0), g = run(sc, q, d1, wm, wc)
    nodes = q.size - E
    assert_close(g["q"][:nodes].reshape(-1, 3).sum(0), wm.sum(0), 1.0e-5, "translation")
    Q = Rotation.random(random_state=rng.integers(1 << 31)).as_matrix()
    qr = q.copy()
    qr[:nodes] = (q[:nodes].reshape(-1, 3) @ Q.T).ravel()
    (mr, cr), _ = run(sc, qr, d1 @ Q.T)
    assert_close(mr, m0 @ Q.T, 1.0e-6, "rotated means", scale=1.0)
    assert_close(full(cr), Q @ full(c0) @ Q.T, 1.0e-5, "rotated covariances")
    q2 = q.copy()
    q2[nodes:] += 2.0 * np.pi
    (m2, c2), g2 = run(sc, q2, d1, wm, wc)
    assert_close(m2, m0, 1.0e-6, "theta + 2 pi", scale=1.0)
    assert_close(g2["q"], g["q"], 1.0e-5, "theta + 2 pi grads")
    t, _, _ = frames_ref(q, d1, n0, n1)
    assert_close(np.sum(g["d1"] * np.cross(t, d1), 1), g["q"][nodes:], 1.0e-5, "gauge")


# -- export ---------------------------------------------------------------------------------


def rigid_ref(q, d1, n0, n1, sp, l0):
    """The skin with every edge at its rest length about its midpoint: what rides on the proxies."""
    E = len(n0)
    x = q[:-E].reshape(-1, 3)
    t, m1, m2 = frames_ref(q, d1, n0, n1)
    e = sp.edge
    mid = 0.5 * (x[n0] + x[n1])
    mean = mid[e] + ((sp.s - 0.5) * l0[e])[:, None] * t[e] + sp.uv[:, :1] * m1[e] + sp.uv[:, 1:] * m2[e]
    return mean, skin_ref(q, d1, n0, n1, sp)[1]


def world(gaussians, X):
    """World means and cov6 of per-body Gaussians at poses ``X`` (bodies, 7)."""
    means, covs = [], []
    for g, x in zip(gaussians, X):
        R = Rotation.from_quat(x[3:]).as_matrix() @ Rotation.from_quat(g.rotations).as_matrix()
        means.append(g.positions @ Rotation.from_quat(x[3:]).as_matrix().T + x[:3])
        covs.append((R * g.scales[:, None, :] ** 2) @ R.transpose(0, 2, 1))
    return np.concatenate(means), np.concatenate(covs)[:, IU[0], IU[1]]


def assert_appearance(gaussians, colours, opacity):
    """The Gaussians' degree-0 SH and opacities = ``colours`` and ``opacity`` (in their order)."""
    assert all(g.sh_degree == 0 for g in gaussians)
    sh = np.concatenate([g.sh_coeffs for g in gaussians])
    assert_close(SH_C0 * sh + 0.5, colours, 1.0e-6, "colours", scale=1.0)
    assert_close(np.concatenate([g.opacities for g in gaussians]), opacity, 1.0e-6, "opacity", scale=1.0)


def test_edge_gaussians_ride_the_proxies(rng):
    """The Gaussians of :func:`edge_gaussians` posed by the proxies = the skin at rest length; appearance as SH."""
    model, solver, state = bent(proxies=True)
    solver.update_proxies(state)
    splats, _ = tube(model, rings=2, per_ring=4, radius=RADIUS, sigma=SIGMA)
    sp, d, S = splats.numpy(), model.dismech, len(splats)
    colours, opacity = rng.random((S, 3)), rng.random(S)
    gaussians = edge_gaussians(splats, d.edge_length.numpy(), colours, opacity)
    order = np.argsort(sp.edge, kind="stable")
    means, cov6 = world(gaussians, state.body_q.numpy()[d.edge_body.numpy()])
    m, c = rigid_ref(state.dismech.q.numpy().astype(np.float64), state.dismech.edge_d1_q.numpy().astype(np.float64),
                     d.edge_node0.numpy(), d.edge_node1.numpy(), sp, d.edge_length.numpy())
    assert_close(means, m[order], 1.0e-6, "means", scale=1.0)
    assert_close(cov6, c[order], 1.0e-5, "cov6")
    assert_appearance(gaussians, colours[order], opacity[order])


def read_frame(stage, prims, k):
    """The fields ``prims`` at time ``k`` as Gaussians and their world poses ``(len(prims), 7)``."""
    from pxr import UsdGeom

    cache = UsdGeom.XformCache(k)
    gaussians, X = [], []
    for p in prims:
        gaussians.append(newton.Gaussian(*(np.array(p.GetAttribute(a).Get(k)) for a in
                                           ("positions", "orientations", "scales"))))
        M = np.array(cache.GetLocalToWorldTransform(p)).T  # USD: row vectors
        X.append(np.concatenate([M[:3, 3], Rotation.from_matrix(M[:3, :3]).as_quat()]))
    return gaussians, X


@pytest.mark.parametrize("baked", [False, True], ids=["rigged", "baked"])
def test_export_usd_round_trip(scene, tmp_path, rng, baked):
    """Two frames written and read back (appearance with Newton's reader): baked = the skin, rigged = the skin at
    rest length, grouped by edge."""
    pytest.importorskip("pxr.UsdVol")
    from pxr import Usd, UsdGeom

    sc = scene
    model, splats, sp = sc["model"], sc["splats"], sc["sp"]
    l0 = model.dismech.edge_length.numpy()
    E, S = len(l0), len(sp.edge)
    q1 = sc["q"].copy()
    q1[-E:] += rng.normal(size=E)  # twisted
    frames = [(sc["q"], sc["d1"]), (q1, sc["d1"])]
    colours, opacity = rng.random((S, 3)), rng.random(S)
    path = tmp_path / "rod.usda"
    export_usd(path, model, splats, colours, opacity, frames, fps=30.0, baked=baked)
    stage = Usd.Stage.Open(str(path))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z and stage.GetTimeCodesPerSecond() == 30.0
    paths = ["/rod/splats"] if baked else [f"/rod/edge_{e}/splats" for e in range(E)]
    prims = [stage.GetPrimAtPath(p) for p in paths]
    order = np.arange(S) if baked else np.argsort(sp.edge, kind="stable")
    for k, (q, d1) in enumerate(frames):
        ref = skin_ref(q, d1, sc["n0"], sc["n1"], sp) if baked else rigid_ref(q, d1, sc["n0"], sc["n1"], sp, l0)
        means, cov6 = world(*read_frame(stage, prims, k))
        assert_close(means, ref[0][order], 1.0e-6, f"frame {k} means", scale=1.0)
        assert_close(cov6, ref[1][order], 1.0e-5, f"frame {k} cov6")
    assert_appearance([newton.Gaussian.create_from_usd(p) for p in prims], colours[order], opacity[order])
