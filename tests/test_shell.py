"""Midedge shells: the geometry against its definition, rest, a plate against beam theory;
Offset Geometric Contact with the ground, shapes, rods and other sheets; rods against shapes."""

import newton
import numpy as np
import pytest
import warp as wp
from rollouts import cloth, grid

from dismech_newton import DiSMechSolver, flatten_state
from dismech_newton import shell as S


def simulate(model, solver, steps: int, dt: float, every: int = 0):
    """The last state, and the positions every ``every`` steps."""
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    frames = []
    for k in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        if every and k % every == 0:
            frames.append(s0.particle_q.numpy().copy())
    return s0, frames


# -- geometry ----------------------------------------------------------------------------------------


def reference_forms(x: np.ndarray, flap: np.ndarray, sign: np.ndarray):
    """The first form and the midedge (tan) second form from their definition (Chen et al. 2018): ``b`` from the
    edge normals ``m_i`` and the midedge normal components ``c_i``, ``b_jk = 2 (c_k m_k . e - c_j m_j . e)``."""
    p = x[:18].reshape(6, 3)
    n = np.cross(p[1] - p[0], p[2] - p[0])
    psi = np.zeros(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        t = p[k] - p[j]
        ng = np.cross(p[j] - p[k], p[3 + i] - p[k])
        psi[i] = 0.5 * np.arctan2(np.cross(n, ng) @ t / np.linalg.norm(t), n @ ng) if flap[i] else 0.0
    c = np.tan(psi + sign * x[18:21])
    nh = n / np.linalg.norm(n)
    m = [np.cross((p[(i + 2) % 3] - p[(i + 1) % 3]) / np.linalg.norm(p[(i + 2) % 3] - p[(i + 1) % 3]), nh)
         for i in range(3)]

    def b_edge(j, k):
        e = p[k] - p[j]
        return 2.0 * (c[k] * (m[k] @ e) - c[j] * (m[j] @ e))

    buu, bvv, b00 = b_edge(0, 1), b_edge(0, 2), b_edge(1, 2)
    eu, ev = p[1] - p[0], p[2] - p[0]
    return np.array([eu @ eu, eu @ ev, ev @ ev, buu, 0.5 * (buu + bvv - b00), bvv])


@wp.kernel
def _forms_kernel(P: wp.array[S.mat63f], th: wp.array[wp.vec3], sign: wp.array[wp.vec3], flap: wp.array[wp.vec3],
                  out: wp.array2d[float], J: wp.array3d[float]):
    f = wp.tid()
    B = S._b_of_g()
    m, g, Jf = S._derivatives(P[f], th[f], sign[f], flap[f], wp.identity(3, float), B)
    m2, g2 = S._forms(P[f], th[f], sign[f], flap[f])
    b = B * g
    for r in range(3):
        out[f, r] = m[r]
        out[f, 3 + r] = b[r]
        out[f, 6 + r] = m2[r] - m[r]
        out[f, 9 + r] = g2[r] - g[r]
        for k in range(21):
            J[f, r, k] = Jf[r, k]
            J[f, 3 + r, k] = Jf[3 + r, k]


def test_forms_and_derivatives(rng):
    """Random faces, flaps (or none), angles: the forms match their definition, the hand derivatives their finite
    differences."""
    F = 32
    base = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, -1, 1, 0, 1, -1, 0.0])
    X = base + 0.25 * rng.normal(size=(F, 18))
    th = 0.3 * rng.normal(size=(F, 3))
    sign = rng.choice([-1.0, 1.0], size=(F, 3))
    flap = (rng.random((F, 3)) < 0.7).astype(float)
    out = wp.zeros((F, 12), dtype=float)
    J = wp.zeros((F, 6, 21), dtype=float)
    wp.launch(_forms_kernel, dim=F, inputs=[wp.array(X.reshape(F, 6, 3), dtype=S.mat63f), wp.array(th, dtype=wp.vec3),
                                            wp.array(sign, dtype=wp.vec3), wp.array(flap, dtype=wp.vec3)],
              outputs=[out, J])
    out, J = out.numpy(), J.numpy()
    assert np.abs(out[:, 6:]).max() < 1.0e-6  # _forms and _derivatives agree (float32)
    for f in range(F):
        x = np.concatenate([X[f], th[f]])
        np.testing.assert_allclose(out[f, :6], reference_forms(x, flap[f] > 0, sign[f]), rtol=1e-5, atol=1e-5)
        fd = np.zeros((6, 21))
        for c in range(21):
            dx = np.zeros(21)
            dx[c] = 1.0e-6
            fd[:, c] = (reference_forms(x + dx, flap[f] > 0, sign[f]) - reference_forms(x - dx, flap[f] > 0, sign[f]))
        fd /= 2.0e-6
        # float32 round-off: 3.6e-6 at the 99.9th percentile of 2048 faces (64 seeds), 1.3e-5 at the worst (an
        # entry ~500, near tan's pole).
        assert np.abs(J[f] - fd).max() < 3.0e-5 * max(np.abs(fd).max(), 1.0)


def test_curved_shell_at_rest_stays(rng):
    """A curved irregular mesh without gravity: unstrained at rest, so it stays put."""
    v, f = grid(6, 5, 0.3, 0.2, rng, jitter=0.3)
    v[:, 2] = 0.2 * np.sin(4.0 * v[:, 0]) + 0.1 * v[:, 1] ** 2
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cloth(builder, v, f, density=1.0, tri_ke=1.0e4, tri_ka=1.0e4, edge_ke=1.0)
    model = builder.finalize()
    moved, _ = simulate(model, DiSMechSolver(model), 5, 0.01)
    assert np.abs(moved.particle_q.numpy() - v).max() < 1.0e-6


@pytest.mark.parametrize("exact", [False, True])
def test_plate_strip_converges_to_beam_theory(exact):
    """Tip deflection of a strip clamped at ``x = 0`` (two node columns) under its weight against ``p L^4 / (8 D)``
    (the clamp at the second column), nu = 0: within 20%, closer on the finer mesh."""
    E, h, rho, length, width = 1.0e10, 2.0e-3, 1000.0, 0.3, 0.06
    D = E * h**3 / 12.0
    ratios = []
    for nx, ny in ((16, 4), (31, 7)):
        v, f = grid(nx, ny, length, width)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        mu = 0.5 * E * h  # nu = 0: lam = 0
        ids = cloth(builder, v, f, density=rho * h, tri_ke=mu, tri_ka=0.0, bend_stiffness=D, damping=0.0)
        dx = length / (nx - 1)
        for k in np.nonzero(v[:, 0] < 1.5 * dx)[0]:
            builder.particle_flags[ids[k]] &= ~newton.ParticleFlags.ACTIVE
        model = builder.finalize()
        out, _ = simulate(model, DiSMechSolver(model, iterations=10, exact=exact), 200, 0.02)
        tip = out.particle_q.numpy()[v[:, 0] > length - 1.0e-9, 2].mean()
        ratios.append(tip / (-rho * h * 9.81 * (length - dx) ** 4 / (8.0 * D)))
    assert 1.0 < ratios[1] < ratios[0] < 1.2, ratios


def test_shell_lands_on_the_ground():
    """A falling plate, the ground found by the solver itself (no collision pipeline): it comes to rest at its
    contact thickness, without sliding."""
    v, f = grid(6, 6, 0.2, 0.2)
    v[:, 2] = 0.05
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cloth(builder, v, f, density=0.5, tri_ke=2.0e3, tri_ka=1.0e3, edge_ke=1.0e-3, particle_radius=0.004,
          damping=1.0e-3)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    model = builder.finalize()
    model.particle_mu = 0.5
    out, _ = simulate(model, DiSMechSolver(model, iterations=10), 240, 1.0 / 240.0)
    x = out.particle_q.numpy()
    assert np.abs(x[:, 2] - 0.004).max() < 2.0e-4
    assert np.abs(x[:, :2] - v[:, :2]).max() < 1.0e-3


# -- Offset Geometric Contact --------------------------------------------------------------------------


@wp.kernel
def _mesh_distance_kernel(mesh: wp.uint64, points: wp.array[wp.vec3], out: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_point_no_sign(mesh, points[i], 1.0e6)
    out[i] = wp.length(points[i] - wp.mesh_eval_position(mesh, q.face, q.u, q.v))


def mesh_distance(points, x, tris):
    """Each point's distance to the triangle mesh (Warp's BVH, not the solver's contact)."""
    mesh = wp.Mesh(wp.array(x, dtype=wp.vec3), wp.array(np.ravel(tris), dtype=wp.int32))
    out = wp.zeros(len(points), dtype=float)
    wp.launch(_mesh_distance_kernel, dim=len(points), inputs=[mesh.id, wp.array(points, dtype=wp.vec3)],
              outputs=[out])
    return out.numpy()


def mesh_gaps(frames, points, tris) -> list:
    """Per frame after the first five: the closest of ``points(x)`` to the triangles ``tris``."""
    return [mesh_distance(points(x), x, tris).min() for x in frames[5:]]


def overflow(solver):
    return int(solver.contact.source.overflow.numpy()[0])


@pytest.mark.parametrize("self_contact", [True, False])
def test_sheet_on_sheet(self_contact):
    """A sheet dropped on a clamped one, offset by half a cell (its vertices land on faces): it rests at two
    thicknesses; without self_contact it falls through."""
    r = 0.004
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    v, f = grid(9, 9, 0.2, 0.2)
    lower = cloth(builder, v, f, density=0.2, tri_ke=1.0e3, tri_ka=1.0e3, edge_ke=1.0e-3, particle_radius=r)
    for k in lower:
        builder.particle_flags[k] &= ~newton.ParticleFlags.ACTIVE
    v2 = v + [0.0125, 0.0125, 0.03]
    upper = cloth(builder, v2, f, density=0.2, tri_ke=1.0e3, tri_ka=1.0e3, edge_ke=1.0e-3, particle_radius=r,
                  damping=1.0e-2)
    model = builder.finalize()
    solver = DiSMechSolver(model, self_contact=self_contact)
    out, _ = simulate(model, solver, 240, 1.0 / 240.0)
    z = out.particle_q.numpy()[upper, 2]
    if self_contact:
        assert overflow(solver) == 0
        assert np.abs(z - 2 * r).max() < 0.25 * r, (z.min(), z.max())
    else:
        assert z.max() < 0.0


def test_sheet_over_rod():
    """A coarse sheet dropped across a clamped rod of three long edges: the rod passes between its vertices, so
    the contacts are vertex-edge and edge-edge; nothing comes closer than the summed radii."""
    rs, rr = 0.004, 0.01
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    pts = np.column_stack([np.zeros(4), np.linspace(-0.3, 0.3, 4), np.zeros(4)])
    bodies = DiSMechSolver.add_rod(builder, newton.Rod(pts, radius=rr), stretch_stiffness=1.0e5,
                                   bend_stiffness=1.0, twist_stiffness=1.0)
    for b in bodies:
        DiSMechSolver.fix_segment(builder, b)
    v, f = grid(7, 7, 0.3, 0.3)
    v += [-0.15 + 0.021, -0.15 + 0.013, 0.03]  # rod between vertex columns
    ids = cloth(builder, v, f, density=0.2, tri_ke=1.0e3, tri_ka=1.0e3, edge_ke=1.0e-4, particle_radius=rs,
                damping=1.0e-2)
    model = builder.finalize()
    solver = DiSMechSolver(model)
    out, frames = simulate(model, solver, 240, 1.0 / 240.0, every=4)
    tris = np.asarray(ids)[f]
    seg = np.column_stack([np.zeros(61), np.linspace(-0.3, 0.3, 61), np.zeros(61)])
    gaps = mesh_gaps(frames, lambda x: seg, tris)
    x = out.particle_q.numpy()
    assert overflow(solver) == 0
    assert min(gaps) > 0.85 * (rs + rr)
    assert x[ids, 2].max() < 0.04 and x[ids, 2].min() < -0.05  # it hangs over the rod (coarse faces tent above it)


def polyline_gap(a: np.ndarray, b: np.ndarray, samples: int = 41) -> float:
    """The closest two polylines come: points along ``a``'s edges against ``b``'s edges."""
    s = np.linspace(0.0, 1.0, samples)[:, None, None]
    p = ((1.0 - s) * a[:-1] + s * a[1:]).reshape(-1, 1, 3)  # (samples * edges, 1, 3)
    d = b[1:] - b[:-1]
    u = np.clip(((p - b[:-1]) * d).sum(-1) / (d * d).sum(-1), 0.0, 1.0)
    return float(np.linalg.norm(p - (b[:-1] + u[..., None] * d), axis=-1).min())


def test_rod_on_rod():
    """A rod dropped across a clamped one at an angle, a node landing right over it (the edge-edge pair's end, left
    to the node: a node rounded past itself was once neither's, and fell through): nothing comes closer than the
    summed radii, and it rests on it."""
    r = 0.01
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    lower = np.column_stack([np.linspace(-0.3, 0.3, 4), np.zeros(4), np.zeros(4)])
    for b in DiSMechSolver.add_rod(builder, newton.Rod(lower, radius=r), stretch_stiffness=1.0e5, bend_stiffness=1.0,
                                   twist_stiffness=1.0, collide=True):
        DiSMechSolver.fix_segment(builder, b)
    s = np.linspace(-0.3, 0.3, 7)
    upper = np.column_stack([0.013 + s * np.sin(0.6), s * np.cos(0.6), np.full(7, 0.05)])
    DiSMechSolver.add_rod(builder, newton.Rod(upper, radius=r), stretch_stiffness=1.0e4, bend_stiffness=0.05,
                          twist_stiffness=0.05, collide=True)
    model = builder.finalize()
    solver = DiSMechSolver(model)
    _, frames = simulate(model, solver, 120, 1.0 / 240.0, every=1)
    gaps = [polyline_gap(x[:4], x[4:]) for x in frames]
    assert overflow(solver) == 0
    assert min(gaps) > 0.85 * 2 * r
    assert abs(gaps[-1] - 2 * r) < 0.05 * r  # resting on it


def test_rod_on_hammock():
    """A rod dropped on a sheet clamped at its border: it is held above the sheet by the summed radii."""
    rs, rr = 0.002, 0.005
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    v, f = grid(9, 9, 0.4, 0.4)
    v -= [0.2, 0.2, 0.0]
    ids = cloth(builder, v, f, density=0.5, tri_ke=5.0e3, tri_ka=5.0e3, edge_ke=1.0e-3, particle_radius=rs,
                damping=1.0e-2)
    border = (np.abs(v[:, 0]) > 0.19) | (np.abs(v[:, 1]) > 0.19)
    for k in np.asarray(ids)[border]:
        builder.particle_flags[k] &= ~newton.ParticleFlags.ACTIVE
    pts = np.column_stack([np.linspace(-0.12, 0.12, 13), np.full(13, 0.031), np.full(13, 0.03)])
    DiSMechSolver.add_rod(builder, newton.Rod(pts, radius=rr), stretch_stiffness=1.0e4, bend_stiffness=1.0e-2,
                          twist_stiffness=1.0e-2)
    model = builder.finalize()
    solver = DiSMechSolver(model)
    out, frames = simulate(model, solver, 240, 1.0 / 240.0, every=4)
    rod = np.arange(len(v), model.particle_count)
    tris = np.asarray(ids)[f]
    gaps = mesh_gaps(frames, lambda x: np.concatenate([x[rod], 0.5 * (x[rod[1:]] + x[rod[:-1]])]), tris)
    x = out.particle_q.numpy()
    assert overflow(solver) == 0
    assert min(gaps) > 0.85 * (rs + rr)
    assert gaps[-1] < 1.1 * (rs + rr)  # resting on it
    assert (x[rod, 2] > x[ids, 2].min()).all()


def box_mesh(h):
    """A closed box of half extent ``h``, outward-wound triangles."""
    v = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d in quads:
        tris += [(a, b, c), (a, c, d)]
    return newton.Mesh(v, np.array(tris).ravel())


def add_shape(builder, shape: str, h: float, z: float = 0.0):
    """A sphere of radius ``h``, or a box (primitive or mesh) of half extent ``h``, centred at height ``z``."""
    xf = wp.transform((0.0, 0.0, z), wp.quat_identity())
    if shape == "sphere":
        builder.add_shape_sphere(-1, xform=xf, radius=h)
    elif shape == "box":
        builder.add_shape_box(-1, xform=xf, hx=h, hy=h, hz=h)
    else:
        builder.add_shape_mesh(-1, xform=xf, mesh=box_mesh(h))


def sdf(p, shape: str, h: float, z: float = 0.0):
    """The signed distance of the points ``p`` to :func:`add_shape`'s shape."""
    p = p - [0.0, 0.0, z]
    if shape == "sphere":
        return np.linalg.norm(p, axis=1) - h
    q = np.abs(p) - h
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0)


SHAPES = ["sphere", "box", "mesh"]


@pytest.mark.parametrize("shape", SHAPES)
def test_sheet_on_shape(shape):
    """A sheet dropped on a shape (no collision pipeline): no vertex comes inside its offset surface, and the vertices
    over the top rest on it at their thickness."""
    r, h = 0.004, 0.08
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    add_shape(builder, shape, h, z=h)
    builder.add_ground_plane()
    v, f = grid(15, 15, 0.42, 0.42)
    v += [-0.21, -0.21, 2 * h + 0.03]
    ids = cloth(builder, v, f, density=0.2, tri_ke=1.0e3, tri_ka=1.0e3, edge_ke=1.0e-4, particle_radius=r,
                damping=1.0e-2)
    model = builder.finalize()
    model.particle_mu = 0.5
    solver = DiSMechSolver(model)
    out, frames = simulate(model, solver, 360, 1.0 / 240.0, every=4)
    gaps = np.array([np.minimum(sdf(x[ids], shape, h, h), x[ids, 2]) for x in frames])  # (frames, vertices)
    k, v = np.unravel_index(gaps.argmin(), gaps.shape)
    x = out.particle_q.numpy()[ids]
    top = (np.abs(x[:, 0]) < 0.5 * h) & (np.abs(x[:, 1]) < 0.5 * h)
    # 0.80-0.93 r over 40 runs (2026-10-27, mesh and box alike): the rim sliding down the side; [mesh] once < 0 (once
    # in ~14 full runs, not reproduced since), so say where.
    assert gaps[k, v] > 0.7 * r, f"vertex {v} at step {4 * k}: gap {gaps[k, v] / r:.2f} r at {frames[k][ids[v]]}"
    gap = np.abs(sdf(x[top], shape, h, h) - r)
    assert (gap.min() if shape == "sphere" else gap.max()) < 0.15 * r  # a sphere's top touches at one vertex


@pytest.mark.parametrize("shape", SHAPES)
def test_rod_on_shape(shape):
    """A rod of three long edges dropped across a shape: its middle edge spans the top, so the contact is at the
    edge's interior (a sphere's top, a box's rims); no point of the rod comes inside the offset surface, and it
    rests on it at its radius."""
    r, h = 0.01, 0.1
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    add_shape(builder, shape, h)
    pts = np.column_stack([np.linspace(-0.45, 0.45, 4), np.full(4, 0.013), np.full(4, h + 2 * r)])
    DiSMechSolver.add_rod(builder, newton.Rod(pts, radius=r), stretch_stiffness=1.0e5, bend_stiffness=10.0,
                          twist_stiffness=10.0)
    model = builder.finalize()
    solver = DiSMechSolver(model)
    _, frames = simulate(model, solver, 240, 1.0 / 240.0, every=4)
    s = np.linspace(0.0, 1.0, 31)[:, None]
    gaps = [sdf(np.concatenate([(1 - s) * x[k] + s * x[k + 1] for k in range(3)]), shape, h).min() for x in frames]
    assert overflow(solver) == 0
    assert min(gaps) > 0.85 * r
    assert abs(gaps[-1] - r) < 0.1 * r  # resting on it
