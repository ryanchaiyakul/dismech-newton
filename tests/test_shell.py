"""Midedge shells in the unified solver: the geometry against its definition, rest, a plate against beam theory;
Offset Geometric Contact with the ground, shapes, rods and other sheets; rods against shapes."""

import newton
import numpy as np
import pytest
import warp as wp

from dismech_newton import flatten_state
from dismech_newton.unified import DiSMechSolver
from dismech_newton.unified import shell as S


def grid(nx: int, ny: int, lx: float, ly: float, rng=None, jitter: float = 0.0):
    """An ``nx x ny``-node plate in the xy plane, faces counter-clockwise; ``jitter`` moves interior nodes."""
    xs, ys = np.linspace(0.0, lx, nx), np.linspace(0.0, ly, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    v = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    if jitter:
        inner = (X.ravel() > 0) & (X.ravel() < lx) & (Y.ravel() > 0) & (Y.ravel() < ly)
        v[inner, :2] += jitter * rng.uniform(-1.0, 1.0, (inner.sum(), 2)) * [lx / (nx - 1), ly / (ny - 1)]
    f = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
            f += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    return v, np.array(f)


def cloth(builder, v, f, **kwargs):
    return S.add_cloth_mesh(builder, vertices=v.tolist(), indices=f.ravel().tolist(), **kwargs)


def run(model, solver, steps: int, dt: float, pipeline=None):
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    contacts = pipeline.contacts() if pipeline is not None else None
    for _ in range(steps):
        if pipeline is not None:
            pipeline.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, dt)
        s0, s1 = s1, s0
    return s0


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
def _forms_kernel(P: wp.array[S.mat63d], th: wp.array[S.vec3d], sign: wp.array[wp.vec3], flap: wp.array[wp.vec3],
                  out: wp.array2d[wp.float64], J: wp.array3d[wp.float64]):
    f = wp.tid()
    B = S._b_of_g()
    m, g, Jf = S._derivatives(P[f], th[f], sign[f], flap[f], wp.identity(3, wp.float64), B)
    m2, g2 = S._forms(P[f], th[f], sign[f], flap[f])
    b = B * g
    for r in range(3):
        out[f, r] = m[r]
        out[f, 3 + r] = b[r]
        out[f, 6 + r] = m2[r] - m[r]
        out[f, 9 + r] = g2[r] - g[r]
        for k in range(21):
            J[f, r, k] = wp.float64(Jf[r, k])
            J[f, 3 + r, k] = wp.float64(Jf[3 + r, k])


def test_forms_and_derivatives(rng):
    """Random faces, flaps (or none), angles: the forms match their definition, the hand derivatives their finite
    differences."""
    F = 32
    base = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, -1, 1, 0, 1, -1, 0.0])
    X = base + 0.25 * rng.normal(size=(F, 18))
    th = 0.3 * rng.normal(size=(F, 3))
    sign = rng.choice([-1.0, 1.0], size=(F, 3))
    flap = (rng.random((F, 3)) < 0.7).astype(float)
    out = wp.zeros((F, 12), dtype=wp.float64)
    J = wp.zeros((F, 6, 21), dtype=wp.float64)
    wp.launch(_forms_kernel, dim=F, inputs=[wp.array(X.reshape(F, 6, 3), dtype=S.mat63d), wp.array(th, dtype=S.vec3d),
                                            wp.array(sign, dtype=wp.vec3), wp.array(flap, dtype=wp.vec3)],
              outputs=[out, J])
    out, J = out.numpy(), J.numpy()
    assert np.abs(out[:, 6:]).max() < 1.0e-12  # _forms and _derivatives agree
    for f in range(F):
        x = np.concatenate([X[f], th[f]])
        np.testing.assert_allclose(out[f, :6], reference_forms(x, flap[f] > 0, sign[f]), rtol=1e-10, atol=1e-10)
        fd = np.zeros((6, 21))
        for c in range(21):
            dx = np.zeros(21)
            dx[c] = 1.0e-6
            fd[:, c] = (reference_forms(x + dx, flap[f] > 0, sign[f]) - reference_forms(x - dx, flap[f] > 0, sign[f]))
        fd /= 2.0e-6
        assert np.abs(J[f] - fd).max() < 1.0e-5 * max(np.abs(fd).max(), 1.0)  # J is float32


def test_curved_shell_at_rest_stays(rng):
    """A curved irregular mesh without gravity: unstrained at rest, so it stays put."""
    v, f = grid(6, 5, 0.3, 0.2, rng, jitter=0.3)
    v[:, 2] = 0.2 * np.sin(4.0 * v[:, 0]) + 0.1 * v[:, 1] ** 2
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cloth(builder, v, f, density=1.0, tri_ke=1.0e4, tri_ka=1.0e4, edge_ke=1.0)
    model = builder.finalize()
    moved = run(model, DiSMechSolver(model), 5, 0.01)
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
        out = run(model, DiSMechSolver(model, iterations=10, exact=exact), 200, 0.02)
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
    out = run(model, DiSMechSolver(model, iterations=10), 240, 1.0 / 240.0)
    x = out.particle_q.numpy()
    assert np.abs(x[:, 2] - 0.004).max() < 2.0e-4
    assert np.abs(x[:, :2] - v[:, :2]).max() < 1.0e-3


# -- Offset Geometric Contact --------------------------------------------------------------------------


def simulate(model, solver, steps, dt, every=0):
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    worst = []
    for k in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        if every and k % every == 0:
            worst.append(s0.particle_q.numpy().copy())
    return s0, worst


def closest_on_triangle(p, a, b, c):
    """Ericson's closest point, numpy, one point."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + d1 / (d1 - d3) * ab
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + d2 / (d2 - d6) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
        return b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b)
    den = 1.0 / (va + vb + vc)
    return a + ab * vb * den + ac * vc * den


def mesh_distance(points, x, tris):
    """Each point's distance to the triangle mesh."""
    return np.array([min(np.linalg.norm(p - closest_on_triangle(p, *x[t])) for t in tris) for p in points])


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
    print("upper z: min", z.min(), "mean", z.mean(), "overflow", overflow(solver))
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
    gaps = [mesh_distance(seg, x, tris).min() for x in frames[5:]]
    x = out.particle_q.numpy()
    print("min rod-sheet distance", min(gaps), "th", rs + rr, "sheet z range", x[ids, 2].min(), x[ids, 2].max(),
          "overflow", overflow(solver))
    assert overflow(solver) == 0
    assert min(gaps) > 0.85 * (rs + rr)
    assert x[ids, 2].max() < 0.04 and x[ids, 2].min() < -0.05  # it hangs over the rod (coarse faces tent above it)


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
    gaps = []
    for x in frames[5:]:
        p = np.concatenate([x[rod], 0.5 * (x[rod[1:]] + x[rod[:-1]])])
        gaps.append(mesh_distance(p, x, tris).min())
    x = out.particle_q.numpy()
    print("min rod-sheet distance", min(gaps), "th", rs + rr, "rod z", x[rod, 2].min(), "sheet z", x[ids, 2].min(),
          "overflow", overflow(solver))
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


@pytest.mark.parametrize("shape", ["sphere", "box", "mesh"])
def test_sheet_on_shape(shape):
    """A sheet dropped on a shape (no collision pipeline): no vertex comes inside its offset surface, and the vertices
    over the top rest on it at their thickness."""
    r, h = 0.004, 0.08
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    xf = wp.transform((0.0, 0.0, h), wp.quat_identity())
    if shape == "sphere":
        builder.add_shape_sphere(-1, xform=xf, radius=h)
    elif shape == "box":
        builder.add_shape_box(-1, xform=xf, hx=h, hy=h, hz=h)
    else:
        builder.add_shape_mesh(-1, xform=xf, mesh=box_mesh(h))
    builder.add_ground_plane()
    v, f = grid(15, 15, 0.42, 0.42)
    v += [-0.21, -0.21, 2 * h + 0.03]
    ids = cloth(builder, v, f, density=0.2, tri_ke=1.0e3, tri_ka=1.0e3, edge_ke=1.0e-4, particle_radius=r,
                damping=1.0e-2)
    model = builder.finalize()
    model.particle_mu = 0.5
    solver = DiSMechSolver(model)
    out, frames = simulate(model, solver, 360, 1.0 / 240.0, every=4)
    c = np.array([0.0, 0.0, h])

    def sdf(x):
        p = x - c
        if shape == "sphere":
            return np.linalg.norm(p, axis=1) - h
        q = np.abs(p) - h
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0)

    worst = min(min(sdf(x[ids]).min(), x[ids, 2].min()) for x in frames)
    x = out.particle_q.numpy()[ids]
    top = (np.abs(x[:, 0]) < 0.5 * h) & (np.abs(x[:, 1]) < 0.5 * h)
    print(shape, "closest", worst, "top gap", (sdf(x[top]) - r).round(5).min(), (sdf(x[top]) - r).max(),
          "overflow", overflow(solver))
    assert worst > 0.7 * r
    gap = np.abs(sdf(x[top]) - r)
    assert (gap.min() if shape == "sphere" else gap.max()) < 0.15 * r  # a sphere's top touches at one vertex


@pytest.mark.parametrize("shape", ["sphere", "box", "mesh"])
def test_rod_on_shape(shape):
    """A rod of three long edges dropped across a shape: its middle edge spans the top, so the contact is at the
    edge's interior (a sphere's top, a box's rims); no point of the rod comes inside the offset surface, and it
    rests on it at its radius."""
    r, h = 0.01, 0.1
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    xf = wp.transform((0.0, 0.0, 0.0), wp.quat_identity())
    if shape == "sphere":
        builder.add_shape_sphere(-1, xform=xf, radius=h)
    elif shape == "box":
        builder.add_shape_box(-1, xform=xf, hx=h, hy=h, hz=h)
    else:
        builder.add_shape_mesh(-1, xform=xf, mesh=box_mesh(h))
    pts = np.column_stack([np.linspace(-0.45, 0.45, 4), np.full(4, 0.013), np.full(4, h + 2 * r)])
    DiSMechSolver.add_rod(builder, newton.Rod(pts, radius=r), stretch_stiffness=1.0e5, bend_stiffness=10.0,
                          twist_stiffness=10.0)
    model = builder.finalize()
    solver = DiSMechSolver(model)
    out, frames = simulate(model, solver, 240, 1.0 / 240.0, every=4)

    def sdf(p):
        if shape == "sphere":
            return np.linalg.norm(p, axis=1) - h
        q = np.abs(p) - h
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0)

    s = np.linspace(0.0, 1.0, 31)[:, None]
    gaps = []
    for x in frames:
        pts = np.concatenate([(1 - s) * x[k] + s * x[k + 1] for k in range(3)])
        gaps.append(sdf(pts).min())
    print(shape, "closest", min(gaps), "final", gaps[-1], "r", r, "z", out.particle_q.numpy()[:, 2],
          "overflow", overflow(solver))
    assert overflow(solver) == 0
    assert min(gaps) > 0.85 * r
    assert abs(gaps[-1] - r) < 0.1 * r  # resting on it
