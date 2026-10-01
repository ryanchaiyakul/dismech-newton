"""Flagella bundling from the IMC paper (Tong, Choi et al., arXiv:2205.10309, Sec. 4.1).

``M`` right-handed helical flagella hang from clamps on a regular polygon of side 0.03 m. Each
clamped first edge lies on its helix axis and spins at ``omega``; the helices rotate in a viscous
fluid (regularized Stokeslet segments, :mod:`stokes`: one drag force per step, backward Euler in the
drag), wrap around one another and bundle, held apart by contact (frictionless in the paper;
``--mu`` adds friction).
Geometry and the clamp follow the paper's code (``world::rodGeometry``): a straight clamped edge, a
45-degree lead-in, then 4 helical turns, 68 nodes. Parameters are the paper's; see ``PAPER``.

Two rods: our DER with ADMM (:class:`DERFlagella`), and Newton's VBD cable (:class:`VBDFlagella`),
whose segments are rigid capsules; its drag is computed on the capsule joints and applied as wrenches.

    uv run examples/flagella.py --flagella 3
    uv run examples/flagella.py --flagella 3 --solver vbd
    uv run examples/flagella.py --viewer null --test
"""

import math

import newton
import newton.examples
import numpy as np
import warp as wp
from common import segment_distance, segment_dofs
from stokes import Stokeslets

from dismech_newton import ADMMDiSMechSolver, flatten_state

PAPER = {
    "youngs_modulus": 3.0e6,  # [Pa]
    "poissons_ratio": 0.5,
    "density": 1000.0,  # [kg / m^3]
    "radius": 1.0e-3,  # h [m]
    "viscosity": 0.1,  # [Pa s]
    "epsilon": 1.02e-3,  # RSS regularization, 1.02 h
    "helix_radius": 0.01,  # a [m]
    "helix_pitch": 0.05,  # lambda [m]
    "axial_length": 0.2,  # z0 [m]
    "edge_length": 5.0e-3,  # target edge length of the helix (68 nodes)
    "spacing": 0.03,  # side of the clamp polygon [m]
    "omega": 15.0,  # [rad / s]
    "dt": 1.0e-3,  # [s]
    "total_time": 250.0,  # [s]
    "friction": 0.0,  # Coulomb mu between flagella (the paper's contact is frictionless)
}

TOP = 0.3  # height of the clamps [m]


def flagellum(p: dict, offset: tuple[float, float]) -> np.ndarray:
    """Nodes of one flagellum, clamp at the top (``world::rodGeometry``, its x axis turned to -z)."""
    a, b = p["helix_radius"], p["helix_pitch"] / (2.0 * math.pi)
    turns = p["axial_length"] / b
    length = turns * math.hypot(a, b)
    ne = int(length / p["edge_length"])
    dl = length / ne
    lead = int(a * math.sqrt(2.0) / dl)  # nodes on the 45-degree lead-in
    x = [(-dl - a, 0.0, 0.0)]
    x += [(-a + k * a / lead, k * a / lead, 0.0) for k in range(lead)]
    x += [(b * s, a * math.cos(s), a * math.sin(s)) for s in np.arange(ne + 1) * turns / ne]
    x = np.array(x)
    # (x, y, z) of the reference to (z, y, -x): a proper rotation, so the helix stays right-handed.
    return np.column_stack((x[:, 2] + offset[1], x[:, 1] + offset[0], TOP - (x[:, 0] + dl + a)))


def _rod(points: np.ndarray, p: dict) -> newton.Rod:
    # Newton's Rod takes nu < 0.5 only; the paper's incompressible nu = 0.5 is G = E / 3.
    g = p["youngs_modulus"] / (2.0 * (1.0 + p["poissons_ratio"]))
    return newton.Rod(points, radius=p["radius"], youngs_modulus=p["youngs_modulus"], shear_modulus=g)


def clamp_offsets(count: int, spacing: float) -> list[tuple[float, float]]:
    """Clamps on a regular ``count``-gon of side ``spacing`` (``world::setRodStepper``)."""
    theta = (count - 2) * math.pi / (2 * count)
    r = spacing / 2.0 / math.cos(theta)
    return [(r - r * math.cos(2 * math.pi * i / count), r * math.sin(2 * math.pi * i / count)) for i in range(count)]


@wp.kernel
def _spin_twist_kernel(dofs: wp.array[wp.int32], rest: wp.array[float], omega: float, dt: float,
                       step: wp.array[wp.int32], q: wp.array[float]):
    """Clamped twist at the end of step ``step``, right-handed about the edge tangent (down): ``+omega`` about +z."""
    i = wp.tid()
    q[dofs[i]] = rest[i] - omega * dt * float(step[0] + 1)


@wp.kernel
def _spin_body_kernel(bodies: wp.array[wp.int32], rest: wp.array[wp.transform], omega: float, dt: float,
                      step: wp.array[wp.int32], frac: float, q0: wp.array[wp.transform], q1: wp.array[wp.transform]):
    """Kinematic clamp capsule turned ``omega t`` about +z (its own axis), ``t`` at ``frac`` of step ``step``."""
    i = wp.tid()
    b = bodies[i]
    t = rest[i]
    dq = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), omega * dt * (float(step[0]) + frac))
    x = wp.transform(wp.transform_get_translation(t), wp.mul(dq, wp.transform_get_rotation(t)))
    q0[b] = x
    q1[b] = x


@wp.kernel
def _count_kernel(step: wp.array[wp.int32], iterations: wp.array[wp.int32], total: wp.array[wp.int32]):
    step[0] = step[0] + 1
    total[0] = total[0] + iterations[0]


@wp.kernel
def _capsule_nodes_kernel(
    body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], half: wp.array[float], segments: int,
    # outputs
    pos: wp.array[wp.vec3], vel: wp.array[wp.vec3],
):
    """Joint ``j`` of a chain of capsule bodies (body frame at the COM, +Z along the capsule): the
    mean of the touching capsule ends, and their mean velocity."""
    tid = wp.tid()
    rod = tid // (segments + 1)
    j = tid - rod * (segments + 1)
    p = wp.vec3()
    v = wp.vec3()
    n = float(0.0)
    for side in range(2):
        s = j - 1 + side  # the capsule ending (side 0) or starting (side 1) at joint j
        if s >= 0 and s < segments:
            b = rod * segments + s
            x = body_q[b]
            end = wp.transform_point(x, wp.vec3(0.0, 0.0, half[b] * (1.0 - 2.0 * float(side))))
            r = end - wp.transform_get_translation(x)
            qd = body_qd[b]
            p += end
            v += wp.spatial_top(qd) + wp.cross(wp.spatial_bottom(qd), r)
            n += 1.0
    pos[tid] = p / n
    vel[tid] = v / n


@wp.kernel
def _capsule_wrench_kernel(
    body_q: wp.array[wp.transform], half: wp.array[float], force: wp.array[wp.vec3], segments: int,
    body_f: wp.array[wp.spatial_vector],
):
    """Each joint's force split between its capsules, applied at their ends: a wrench about the COM."""
    b = wp.tid()
    rod = b // segments
    s = b - rod * segments
    x = body_q[b]
    c = wp.transform_get_translation(x)
    f = wp.vec3()
    tau = wp.vec3()
    for side in range(2):
        j = s + side  # joint at the start (side 0) or end (side 1) of capsule s
        shared = j > 0 and j < segments
        fj = force[rod * (segments + 1) + j]
        if shared:
            fj = 0.5 * fj
        end = wp.transform_point(x, wp.vec3(0.0, 0.0, half[b] * (2.0 * float(side) - 1.0)))
        f += fj
        tau += wp.cross(end - c, fj)
    body_f[b] = wp.spatial_vector(f, tau)


class _Flagella:
    """One step is: drag at the step's start (eager: cuSOLVER does not capture), then drive, collide
    and solve (a CUDA graph, one per state parity)."""

    name = ""

    def __init__(self, count: int, p: dict):
        self.p = dict(p)
        self.count, self.dt = count, p["dt"]
        self.points = [flagellum(p, o) for o in clamp_offsets(count, p["spacing"])]
        self.nv = len(self.points[0])
        self.stokes = Stokeslets(count, self.nv, p["viscosity"], p["epsilon"], wp.get_device())
        self._step = wp.zeros(1, dtype=wp.int32)
        self._iterations = wp.zeros(1, dtype=wp.int32)  # this step's (overwritten by the solver)
        self._total_iterations = wp.zeros(1, dtype=wp.int32)
        self._graphs = [None, None]
        self._parity = 0
        self.steps = 0

    # subclass hooks
    def _drag(self) -> None: ...

    def _solve(self) -> None: ...

    def step(self) -> None:
        self._drag()
        graph = self._graphs[self._parity]
        if graph is not None:
            wp.capture_launch(graph)
        else:
            self._simulate()
            if self.steps >= 1 and wp.get_device().is_cuda:  # the solver sets itself up on its first step
                with wp.ScopedCapture() as capture:
                    self._simulate()
                self._graphs[self._parity] = capture.graph
        self.state_0, self.state_1 = self.state_1, self.state_0
        self._parity ^= 1
        self.steps += 1

    def _simulate(self) -> None:
        self._solve()
        wp.launch(_count_kernel, dim=1, inputs=[self._step, self._iterations], outputs=[self._total_iterations])

    @property
    def time(self) -> float:
        return self.steps * self.dt

    @property
    def total_iterations(self) -> int:
        return int(self._total_iterations.numpy()[0])

    def nodes(self) -> np.ndarray:
        """``(count, nv, 3)`` node positions."""
        ...

    def body_q(self) -> np.ndarray:
        """Capsule poses, for replay in the viewer."""
        ...

    def gaps(self) -> tuple[float, float]:
        """Closest approach between flagella, and within one (edges > 2 apart), in diameters."""
        x = self.nodes()
        a, b = x[:, :-1].reshape(-1, 3), x[:, 1:].reshape(-1, 3)
        rod = np.repeat(np.arange(self.count), self.nv - 1)
        idx = np.tile(np.arange(self.nv - 1), self.count)
        i, j = np.triu_indices(len(a), k=1)
        other = rod[i] != rod[j]
        own = ~other & (np.abs(idx[i] - idx[j]) > 2)
        d = segment_distance(a[i], b[i], a[j], b[j]) / (2.0 * self.p["radius"])
        return float(d[other].min()) if other.any() else np.inf, float(d[own].min())

    def tip_spread(self) -> float:
        """Mean distance between the free ends [m], the reference code's bundling measure."""
        tips = self.nodes()[:, -1]
        i, j = np.triu_indices(self.count, k=1)
        return float(np.linalg.norm(tips[i] - tips[j], axis=1).mean())

    def max_stretch(self) -> float:
        x = self.nodes()
        l = np.linalg.norm(np.diff(x, axis=1), axis=2)
        l0 = np.linalg.norm(np.diff(np.array(self.points), axis=1), axis=2)
        return float(np.abs(l / l0 - 1.0).max())


class DERFlagella(_Flagella):
    """Our DER, ADMM with contact."""

    name = "DER ADMM"

    def __init__(self, count: int, p: dict = PAPER, *, implicit_drag: bool = True, **solver_options):
        super().__init__(count, p)
        h = p["radius"]
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        self.rods = []
        for points in self.points:
            rod = _rod(points, p)
            bodies = ADMMDiSMechSolver.add_rod(builder, rod, cfg=newton.ModelBuilder.ShapeConfig(density=p["density"]))
            ADMMDiSMechSolver.fix_segment(builder, bodies[0])  # nodes 0, 1 and the first twist
            self.rods.append(bodies)
        self.model = model = builder.finalize()
        self.solver = ADMMDiSMechSolver(model, friction=p["friction"], **solver_options)
        self._iterations = self.solver._count
        if implicit_drag:  # the paper's explicit drag diverges at the tips of a tight bundle (stokes.py)
            self.stokes.implicit(self.solver.mass.numpy()[: 3 * count * self.nv], self.dt)
        self.pipeline = newton.CollisionPipeline(model, soft_contact_max=0, verify_buffers=False,
                                                 speculative_contact_gap_max=2.0 * h, contact_matching="latest")
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = model.state(), model.state()
        for s in (self.state_0, self.state_1):
            flatten_state(s)
            s.particle_f = self.stokes.force  # the drag, shared by both states
        dofs = [segment_dofs(model, b[0], twist_only=True)[0] for b in self.rods]
        self._spin_dofs = wp.array(dofs, dtype=wp.int32)
        self._spin_rest = wp.array(self.state_0.dismech.q.numpy()[dofs], dtype=float)

    def _drag(self):
        self.stokes.compute(self.state_0.particle_q, self.state_0.particle_qd)

    def _solve(self):
        wp.launch(_spin_twist_kernel, dim=self.count,
                  inputs=[self._spin_dofs, self._spin_rest, self.p["omega"], self.dt, self._step],
                  outputs=[self.state_0.dismech.q])
        self.pipeline.collide(self.state_0, self.contacts, dt=2.0 * self.dt)
        self.solver.step(self.state_0, self.state_1, None, self.contacts, self.dt)

    def nodes(self):
        return self.state_0.particle_q.numpy().reshape(self.count, self.nv, 3)

    def body_q(self):
        self.solver.update_proxies(self.state_0)
        return self.state_0.body_q.numpy()


class VBDFlagella(_Flagella):
    """Newton's VBD cable: rigid capsules on rod joints, the clamp capsule kinematic."""

    def __init__(self, count: int, p: dict = PAPER, *, iterations: int = 10, substeps: int = 1,
                 contact_ke: float = 1.0e3, **solver_options):
        super().__init__(count, p)
        self.name = f"Newton VBD ({iterations} it" + (f", {substeps} substeps)" if substeps > 1 else ")")
        self.substeps = substeps
        h = p["radius"]
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        cfg = newton.ModelBuilder.ShapeConfig(density=p["density"], mu=p["friction"], ke=contact_ke, gap=h)
        self.rods, clamps = [], []
        for points in self.points:
            rod = _rod(points, p)
            bodies, _ = builder.add_rod(rod=rod, cfg=cfg, body_frame_origin="com")
            b = bodies[0]
            builder.body_mass[b], builder.body_inv_mass[b] = 0.0, 0.0
            builder.body_inertia[b], builder.body_inv_inertia[b] = wp.mat33(0.0), wp.mat33(0.0)
            self.rods.append(list(bodies))
            clamps.append(b)
        builder.color()
        self.model = model = builder.finalize()
        self._cylinder_inertia(model, clamps)
        self.pipeline = newton.CollisionPipeline(model)
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(model, iterations=iterations, rigid_compliant_alm=True, **solver_options)
        self._iterations.fill_(iterations * substeps)
        self.state_0, self.state_1 = model.state(), model.state()
        self.control = model.control()
        self._segments = self.nv - 1
        self._half = wp.array(model.shape_scale.numpy()[:, 1], dtype=float)  # one capsule per body, in order
        self._pos = wp.zeros(self.count * self.nv, dtype=wp.vec3)
        self._vel = wp.zeros(self.count * self.nv, dtype=wp.vec3)
        self._body_f = wp.zeros(model.body_count, dtype=wp.spatial_vector)
        for s in (self.state_0, self.state_1):
            s.body_f = self._body_f
        self._clamps = wp.array(clamps, dtype=wp.int32)
        self._clamp_rest = wp.array(model.body_q.numpy()[clamps], dtype=wp.transform)

    def _cylinder_inertia(self, model, clamps: list[int]) -> None:
        """Solid-cylinder mass and inertia per capsule body (no end caps: the DER lumped mass).

        ``finalize`` adds 1e-6 kg m^2 to any inertia whose smallest moment is below 1e-10, which a
        1 mm capsule's axial moment (~1e-11) is: 10^4 times its transverse moment, the twist could
        not propagate. The solver reads the model's values when it is created, so set them first.
        """
        r, rho = self.p["radius"], self.p["density"]
        length = 2.0 * model.shape_scale.numpy()[:, 1]  # one capsule per body, in order
        m = rho * np.pi * r * r * length
        axial, transverse = 0.5 * m * r * r, m * (3.0 * r * r + length**2) / 12.0
        inertia = np.zeros((model.body_count, 3, 3), dtype=np.float32)
        inertia[:, 0, 0] = inertia[:, 1, 1] = transverse  # body +Z along the capsule
        inertia[:, 2, 2] = axial
        inv = np.zeros_like(inertia)
        for k in range(3):
            inv[:, k, k] = 1.0 / inertia[:, k, k]
        m, inv_m = m.astype(np.float32), (1.0 / m).astype(np.float32)
        m[clamps] = inv_m[clamps] = inertia[clamps] = inv[clamps] = 0.0  # the kinematic clamps
        model.body_mass.assign(m)
        model.body_inv_mass.assign(inv_m)
        model.body_inertia.assign(inertia)
        model.body_inv_inertia.assign(inv)

    def _update_nodes(self, state):
        wp.launch(_capsule_nodes_kernel, dim=self.count * self.nv,
                  inputs=[state.body_q, state.body_qd, self._half, self._segments], outputs=[self._pos, self._vel])

    def _drag(self):
        self._update_nodes(self.state_0)
        force = self.stokes.compute(self._pos, self._vel)
        wp.launch(_capsule_wrench_kernel, dim=self.model.body_count,
                  inputs=[self.state_0.body_q, self._half, force, self._segments], outputs=[self._body_f])

    def _solve(self):
        # Substeps keep the step's drag; states swap back to the pair this graph started from.
        sub_dt = self.dt / self.substeps
        s0, s1 = self.state_0, self.state_1
        for k in range(self.substeps):
            wp.launch(_spin_body_kernel, dim=self.count,
                      inputs=[self._clamps, self._clamp_rest, self.p["omega"], self.dt, self._step,
                              (k + 1) / self.substeps],
                      outputs=[s0.body_q, s1.body_q])
            self.pipeline.collide(s0, self.contacts)
            self.solver.set_rigid_history_update(True)
            self.solver.step(s0, s1, self.control, self.contacts, sub_dt)
            if k < self.substeps - 1:
                wp.copy(s0.body_q, s1.body_q)
                wp.copy(s0.body_qd, s1.body_qd)

    def nodes(self):
        self._update_nodes(self.state_0)
        return self._pos.numpy().reshape(self.count, self.nv, 3)

    def body_q(self):
        return self.state_0.body_q.numpy()


SOLVERS = {"der": DERFlagella, "vbd": VBDFlagella}


class Example:
    def __init__(self, viewer, args=None):
        count = getattr(args, "flagella", 3) if args is not None else 3
        solver = getattr(args, "solver", "der") if args is not None else "der"
        mu = getattr(args, "mu", PAPER["friction"]) if args is not None else PAPER["friction"]
        self.sim = SOLVERS[solver](count, {**PAPER, "friction": mu})
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.steps_per_frame = max(1, round(self.frame_dt / self.sim.dt))
        self.min_gap = np.inf
        viewer.set_model(self.sim.model)
        if hasattr(viewer, "set_camera"):
            viewer.set_camera(pos=wp.vec3(0.015, -0.55, 0.2), pitch=0.0, yaw=90.0)

    def step(self):
        for _ in range(self.steps_per_frame):
            self.sim.step()

    def render(self):
        self.viewer.begin_frame(self.sim.time)
        self.viewer.log_state(self.sim.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        self.min_gap = min(self.min_gap, *self.sim.gaps())

    def test_final(self):
        x = self.sim.nodes()
        assert np.isfinite(x).all(), "non-finite positions"
        assert self.min_gap > 0.8, f"flagella passed into each other (closest {self.min_gap:.2f} diameters)"
        assert self.sim.max_stretch() < 0.05, "flagellum stretched by more than 5%"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--flagella", type=int, default=3, help="number of flagella M")
    parser.add_argument("--solver", choices=tuple(SOLVERS), default="der")
    parser.add_argument("--mu", type=float, default=PAPER["friction"], help="friction between flagella")
    parser.set_defaults(num_frames=600)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
