"""Clamped plates released flat under gravity: the midedge shell beside Newton's VBD cloth, from the same cloth
parameters. The shell runs on ADMM twice, with the theta method at implicit midpoint (``theta=0.5``) and implicit
Euler, so the difference is the integrator's numerical damping alone.

Shows: :mod:`cantilever`'s comparison for shells. A strip clamped along one edge (its first two lattice lines) bends
cylindrically, so with Poisson's ratio 0 it is a beam per unit width: load ``rho g``, stiffness ``D``. Both solvers
read Newton's ``edge_ke``, whose continuum limit on an equilateral mesh is ``D = edge_ke A_e / L_e`` (``A_e`` the
area of a hinge's two faces). The mesh is equilateral because VBD's hinges are isotropic only there (on a square
grid's right triangles their stiffness varies by 20% with the bending direction). The midedge shell's sag converges
to the plate's under refinement (2.3%, 1.4%, 0.5% over at 2.3, 1.15, 0.58 cm triangles).

VBD sags and creeps: its per-vertex solves pass a correction about one ring of the mesh per iteration, while the
bending mode spans the strip beside a membrane ~1e5 times stiffer, so a fixed iteration count stops far from the
step's solution; smaller steps help, more iterations hardly do. Its energy is not the problem: solved exactly, its
model sags about 6% more than the plate, from its hinges' own bending Poisson ratio, -1/3 (the midedge shell's
bending and membrane are one material, here Poisson's ratio 0). So the second VBD strip runs 300 substeps of 100
iterations, where it has converged to that model (5% over, 2% slow; more of either changes nothing, and past ~1000
substeps it degrades again); at 100 of 50 it sags 40% over and swings at 0.6 of the frequency.

    uv run examples/shell_cantilever.py
    uv run examples/shell_cantilever.py --viewer null --test
"""

import newton
import newton.examples
import numpy as np
import warp as wp
from cantilever import GRAVITY
from cantilever import Example as RodExample

from dismech_newton import DiSMechSolver

# name: (solver, its options, color); both on ADMM, the theta method the only difference. Without contact 3
# iterations give the same sag and frequency as 20, in 0.6 of the default 10's time.
SHELL_SOLVERS = {
    "Shell ADMM, implicit midpoint": (DiSMechSolver, {"theta": 0.5, "iterations": 3}, (0.6, 0.45, 0.9)),
    "Shell ADMM, implicit Euler": (DiSMechSolver, {"iterations": 3}, (0.3, 0.8, 0.6)),
}


def equilateral_strip(length: float, width: float, a: float):
    """An equilateral mesh of side ``a``: lattice lines ``x = j sqrt(3)/2 a`` (every other one shifted by ``a/2``
    along y), faces counter-clockwise; with its line spacing."""
    dx = 0.5 * np.sqrt(3.0) * a
    nx, ny = int(round(length / dx)) + 1, int(round(width / a)) + 1
    v = np.array([[j * dx, i * a + 0.5 * a * (j % 2), 0.0] for j in range(nx) for i in range(ny)])
    f = []
    for j in range(nx - 1):
        for i in range(ny - 1):
            a0, a1, b0, b1 = j * ny + i, j * ny + i + 1, (j + 1) * ny + i, (j + 1) * ny + i + 1
            f += [(a0, b0, a1), (b0, b1, a1)] if j % 2 == 0 else [(a0, b0, b1), (a0, b1, a1)]
    return v, np.array(f, dtype=np.int32), dx


@wp.kernel
def _record_sheet_tip(particle_q: wp.array[wp.vec3], tip: wp.array[wp.int32], z0: float, count: wp.array[wp.int32],
                      sag: wp.array[float]):
    """Append the free edge's mean sag (below its start) at ``count``."""
    k = count[0]
    if k < sag.shape[0]:
        z = float(0.0)
        for i in range(tip.shape[0]):
            z += particle_q[tip[i]][2]
        sag[k] = z0 - z / float(tip.shape[0])
    count[0] = k + 1


class Example(RodExample):
    length, width, spacing = 0.4, 0.2, 0.0115  # the strip [m]; its triangles' side (lattice lines 1 cm apart)
    plate_stiffness, density = 1.0, 1.0  # D [N m], per area [kg/m^2]
    tri_ke = 1.0e5  # the 2D Lame mu [N/m]; tri_ka = lam = 0: Poisson's ratio 0
    radius = 2.0e-3  # the vertices' contact thickness (no contact here: only the view's margin)
    height, gap, row_gap = 0.5, 0.15, 0.5
    line_y = 0.5 * width
    sag_tolerance = 0.05
    solvers = SHELL_SOLVERS

    def __init__(self, viewer, args=None, **kwargs):
        self.v, self.f, self.dx = equilateral_strip(self.length, self.width, self.spacing)
        self.length = float(self.v[:, 0].max())  # to the last lattice line
        self.edge_ke = self.plate_stiffness / self.dx  # A_e / L_e = sqrt(3)/2 a = dx
        kwargs.setdefault("vbd", ((10, 5), (300, 100)))  # (substeps, iterations): a few, and converged
        super().__init__(viewer, args, **kwargs)

    def _load(self, scene) -> None:
        """A plate in cylindrical bending, per unit width. The clamp's two lattice lines are fixed and the first
        free hinges lie on the second, so (as a rod's joint) the continuum clamp sits midway between them."""
        self.w = self.density * GRAVITY
        self.ei = self.plate_stiffness
        self.clamp = 0.5 * self.dx

    def _sheet(self, builder, clamp, add) -> np.ndarray:
        v = self.v + [clamp[0], 0.0, clamp[1]]
        ids = np.asarray(add(builder, vertices=v.tolist(), indices=self.f.ravel().tolist(), density=self.density,
                             tri_ke=self.tri_ke, tri_ka=0.0, tri_kd=0.0, edge_ke=self.edge_ke, edge_kd=0.0,
                             particle_radius=self.radius))
        for k in ids[self.v[:, 0] < 1.5 * self.dx]:  # the clamp: two lattice lines, so the edge's slope is held
            builder.particle_flags[k] &= ~newton.ParticleFlags.ACTIVE
            builder.particle_mass[k] = 0.0
        return ids

    def _der_adder(self, solver, color):
        return lambda builder, clamp: self._sheet(builder, clamp, solver.add_cloth_mesh)

    def _vbd_adder(self, color):
        def add(builder, **kw):
            n = builder.particle_count
            builder.add_cloth_mesh(pos=wp.vec3(0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0), **kw)
            return np.arange(n, builder.particle_count)

        return lambda builder, clamp: self._sheet(builder, clamp, add)

    def _vbd_solver(self, model, iterations: int):
        return newton.solvers.SolverVBD(model, iterations=iterations)

    def _tip(self, model, ids: np.ndarray) -> tuple:
        """The free edge's vertices (Poisson's ratio 0: it stays straight across)."""
        tip = ids[self.v[:, 0] > self.length - 0.5 * self.dx]
        z0 = float(model.particle_q.numpy()[tip, 2].mean())
        return "particle_q", _record_sheet_tip, [wp.array(tip, dtype=wp.int32, device=model.device), z0]


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=240)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
