"""A cloth strip flops over onto itself: lying on a table with its far half raised and leaning back over the near
half, it is let go, falls onto itself and settles into a fold that stays open, an eyelet. The eyelet is a heavy
elastica that lifts off the lower layer and lands back on it, with no curvature at either end (the layers are flat);
its shape has no free parameter in units of the elasto-gravity length ``l = (D / rho g)^(1/3)``: height
``0.907 l``, arc length ``4.68 l``, solved here as a boundary value problem.

Shows: shell self-contact with friction (Offset Geometric Contact, vertex-triangle and edge-edge), against a closed
shape. The theory's eyelet has no horizontal force and is frictionless; the strip needs friction on itself only to
stop: without it the fold rolls out along the strip until it lies flat (the fold is in neutral equilibrium along
it), and without friction on the table the whole strip slides away with the fall's momentum. At rest the friction
holds almost nothing, and the eyelet is within about 1% of the frictionless one.

A light cloth's fall is calmed by the air: drag normal to the sheet (``state.particle_f``, quadratic, a flat plate's)
and a limp cloth's strong damping (0.4 s) keep the landing from whipping the free end back up; without them it
rebounds and rings for a second. Neither acts at rest. 4 substeps a frame are enough with this damping (at 0.2 s they
need 8 to calm the landing): the eyelet is the same.

The strip is built flat (its rest shape, and the shell-shell pairs that are its own neighbours) and bent in the state.

    uv run examples/shell_fold.py
    uv run examples/shell_fold.py --viewer null --test
"""

import newton
import newton.examples
import numpy as np
import warp as wp
from cantilever import GRAVITY
from utils.common import THEORY, CableExample, default_frames, grid, inset, inset_scale

from dismech_newton import DiSMechSolver


def eyelet(n: int = 400):
    """The frictionless eyelet in units of ``l``: ``(x, z)`` from where it lifts off the lower layer (heading
    ``+x``) to where it lands back on it (heading ``-x``), ``n`` points evenly along it.

    Its internal force is vertical, ``V0 + s`` (its weight from the lift-off), so ``theta'' = -(V0 + s) cos theta``;
    ``theta = theta' = 0`` at the lift-off, ``theta = pi`` and ``z = 0`` at the landing, which then has
    ``theta' = 0`` too (the first integral ``theta'^2 / 2 = z - (V0 + s) sin theta``). Unknowns ``V0`` and the arc
    length ``S``."""
    from scipy.integrate import solve_bvp  # noqa: PLC0415

    def rhs(t, y, p):
        v0, S = p
        return S * np.array([y[1], -(v0 + S * t) * np.cos(y[0]), np.cos(y[0]), np.sin(y[0])])

    def bc(a, b, p):
        return np.array([a[0], a[1], a[2], a[3], b[0] - np.pi, b[3]])

    t = np.linspace(0.0, 1.0, 200)
    theta = np.pi * t + np.sin(np.pi * t)  # a teardrop, out past pi and back
    guess = np.vstack([theta, np.gradient(theta, t) / 4.0, np.cumsum(np.cos(theta)) / 50.0,
                       np.cumsum(np.sin(theta)) / 50.0])
    sol = solve_bvp(rhs, bc, t, guess, p=[-2.0, 4.0], tol=1.0e-8, max_nodes=100000)
    assert sol.success, sol.message
    return sol.sol(np.linspace(0.0, 1.0, n))[[2, 3]].T


@wp.kernel
def _air_drag(q: wp.array[wp.vec3], qd: wp.array[wp.vec3], area: wp.array[float], nx: int, ny: int, k: float,
              f: wp.array[wp.vec3]):
    """``f = -k A |v_n| v_n n`` on each node of an ``nx x ny`` grid, ``n`` its normal from its grid neighbours."""
    tid = wp.tid()
    i, j = tid // ny, tid % ny
    u = q[wp.min(i + 1, nx - 1) * ny + j] - q[wp.max(i - 1, 0) * ny + j]
    w = q[i * ny + wp.min(j + 1, ny - 1)] - q[i * ny + wp.max(j - 1, 0)]
    n = wp.normalize(wp.cross(u, w))
    vn = wp.dot(qd[tid], n)
    f[tid] = -k * area[tid] * wp.abs(vn) * vn * n


class AirDrag:
    """Air drag on a sheet's grid nodes, normal to it, into ``state.particle_f`` at each substep's start (explicit;
    it is gone at rest)."""

    def __init__(self, model, nx: int, ny: int, density: float, k: float):
        self.nx, self.ny, self.k = nx, ny, k
        self.area = wp.array(model.particle_mass.numpy() / density, dtype=float, device=model.device)

    def __call__(self, state, s: float) -> None:
        wp.launch(_air_drag, dim=self.nx * self.ny,
                  inputs=[state.particle_q, state.particle_qd, self.area, self.nx, self.ny, self.k],
                  outputs=[state.particle_f])


class Example(CableExample):
    duration, substeps = 2.0, 4  # [s]: it lands by 0.3 s and is still by 1.2 s
    plot_every, plot_name = 6, "eyelet"
    length, width, spacing = 0.8, 0.15, 0.005  # [m]: 161 x 31 nodes
    fold_at, bend, lean = 0.35, 0.05, np.deg2rad(150.0)  # the start: flat to 0.35 m, bent on 5 cm to lean back 150 deg
    radius = 2.0e-3  # the vertices' contact thickness: layers 4 mm apart
    density, tri_ke, bend_stiffness, damping = 0.2, 1.0e4, 1.0e-3, 0.4  # [kg/m^2], [N/m], D [N m], [s]: a limp cloth
    mu, table_mu = 0.3, 1.0  # on itself; the table's: the strip on the table 0.65 (the mean)
    drag = 0.5 * 1.2 * 1.2  # air's rho C_d / 2 for a flat plate [kg/m^3]: the strip falls flat at about 1.6 m/s

    def __init__(self, viewer, args=None):
        self.ell = (self.bend_stiffness / (self.density * GRAVITY)) ** (1.0 / 3.0)
        builder = newton.ModelBuilder()
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=self.table_mu))
        self.nx, self.ny = int(round(self.length / self.spacing)) + 1, int(round(self.width / self.spacing)) + 1
        v, f = grid(self.nx, self.ny, self.length, self.width)
        v[:, 2] += self.radius  # on the table
        DiSMechSolver.add_cloth_mesh(builder, vertices=v.tolist(), indices=f.ravel().tolist(), density=self.density,
                                     tri_ke=self.tri_ke, tri_ka=0.0, edge_ke=0.0, bend_stiffness=self.bend_stiffness,
                                     damping=self.damping, particle_radius=self.radius)
        model = builder.finalize()
        model.particle_mu = self.mu
        # 5 iterations: the eyelet height within 0.5% of 10's, in 0.75 of the time
        self.start(viewer, model, DiSMechSolver(model, iterations=5))
        if self.drag:
            self.drives = (AirDrag(model, self.nx, self.ny, self.density, self.drag),)

        # Bend the far half up over the near one: flat, an arc, then straight.
        s, q = v[:, 0], v.copy()
        arc_end = self.fold_at + self.bend * self.lean
        arc, line = (s > self.fold_at) & (s <= arc_end), s > arc_end
        angle = (s[arc] - self.fold_at) / self.bend
        q[arc, 0] = self.fold_at + self.bend * np.sin(angle)
        q[arc, 2] += self.bend * (1.0 - np.cos(angle))
        q[line, 0] = self.fold_at + self.bend * np.sin(self.lean) + (s[line] - arc_end) * np.cos(self.lean)
        q[line, 2] += self.bend * (1.0 - np.cos(self.lean)) + (s[line] - arc_end) * np.sin(self.lean)
        for state in (self.state_0, self.state_1):
            state.particle_q.assign(q)

        self.theory = self.ell * eyelet()
        self.row = np.arange(self.nx) * self.ny + self.ny // 2  # the middle row
        self.t, self.heights, self.profiles = [0.0], [self.height()], [self.profile()]
        if viewer is not None and hasattr(viewer, "set_camera"):
            viewer.set_camera(pos=wp.vec3(0.38, -0.5, 0.22), pitch=-18.0, yaw=90.0)

    def profile(self) -> np.ndarray:
        """The middle row's ``(x, z)``."""
        return self.state_0.particle_q.numpy()[self.row][:, [0, 2]].astype(np.float64)

    def height(self) -> float:
        """The eyelet's top above the lower layer's midplane, along the middle row."""
        return float(self.profile()[:, 1].max() - self.radius)

    def step(self):
        super().step()
        self.t.append(self.sim_time)
        self.heights.append(self.height())
        self.profiles.append(self.profile())

    def fitted(self, x: np.ndarray) -> np.ndarray:
        """The theory's eyelet on the lower layer's midplane, moved along ``x`` onto the profile ``x``'s fold (its
        nose at the profile's)."""
        th = self.theory
        return th + [x[:, 0].max() - th[:, 0].max(), self.radius]

    def image(self, size: tuple[int, int] = (800, 400)) -> np.ndarray:
        """Left: the middle row's cross-section now, node by node, on the frictionless eyelet (dashed). Right: its
        height over time against the eyelet's ``0.907 l``."""
        import matplotlib  # noqa: PLC0415

        pt = inset_scale(size, 2)
        color = matplotlib.colormaps["viridis"](0.5)
        x = self.profiles[-1]
        th = self.fitted(x)
        top = 0.907 * self.ell

        def draw(axes):
            left, right = axes
            left.plot(100.0 * th[:, 0], 100.0 * th[:, 1], color=THEORY, ls="--", lw=1.2 * pt, zorder=3,
                      label="elastica")
            left.plot(100.0 * x[:, 0], 100.0 * x[:, 1], "o", color=color, ms=2.5 * pt, mew=0, label="cloth")
            both = np.concatenate([x, th])  # the strip as it falls, the eyelet once it is down
            left.set(xlim=(100.0 * (both[:, 0].min() - 0.02), 100.0 * (both[:, 0].max() + 0.02)),
                     ylim=(-0.5, 100.0 * max(both[:, 1].max() + 0.01, 2.0 * top)), aspect="equal")
            left.set_xlabel(r"$x$ (cm)")
            left.set_ylabel(r"$z$ (cm)")
            left.legend(loc="upper left")
            right.axhline(100.0 * top, color=THEORY, ls="--", lw=1.2 * pt, label=r"$0.907\,\ell$")
            right.plot(self.t, 100.0 * np.array(self.heights), color=color, lw=1.6 * pt)
            right.set(xlim=(0.0, self.duration), ylim=(0.0, 100.0 * 1.5 * top))
            right.set_xlabel(r"time $t$ (s)")
            right.set_ylabel(r"eyelet height (cm)")
            right.legend(loc="lower right")

        return inset(draw, size, ncols=2)

    def test_final(self):
        q = self.state_0.particle_q.numpy().astype(np.float64)
        assert np.isfinite(q).all(), "non-finite positions"
        assert np.abs(self.state_0.particle_qd.numpy()).max() < 1.0e-2, "the strip has not settled"
        # The same eyelet across the width, of the frictionless one's height and shape.
        rows = q.reshape(self.nx, self.ny, 3)
        heights = rows[:, :, 2].max(0) - self.radius
        assert np.ptp(heights) < 0.02 * self.ell, f"heights across the strip {heights.min():.4f}-{heights.max():.4f} m"
        h = self.heights[-1]
        assert abs(h / (0.907 * self.ell) - 1.0) < 0.03, f"eyelet height {h:.4f} m, theory {0.907 * self.ell:.4f} m"
        x, th = self.profiles[-1], self.fitted(self.profiles[-1])
        loop = x[:, 1] > 3.0 * self.radius + 0.1 * self.ell  # off both layers
        off = np.min(np.linalg.norm(x[loop, None] - th[None], axis=2), axis=1)
        assert off.max() < 0.05 * self.ell, f"the eyelet is {off.max():.4f} m off the elastica's"
        # The upper layer on the lower one, not in it: no two nodes that were apart are closer than a contact
        # thickness.
        from scipy.spatial import cKDTree  # noqa: PLC0415

        assert q[:, 2].min() > 0.5 * self.radius, "the strip went through the table"
        rest = self.model.particle_q.numpy()
        i, j = cKDTree(q).query_pairs(2.0 * self.radius, output_type="ndarray").T
        apart = np.linalg.norm(rest[i] - rest[j], axis=1) > 4.0 * self.radius
        closest = np.linalg.norm(q[i[apart]] - q[j[apart]], axis=1).min(initial=np.inf)
        assert closest > 0.9 * 2.0 * self.radius, f"the strip went into itself: {closest:.2e} m apart"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    default_frames(parser, lambda args: Example.duration)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
