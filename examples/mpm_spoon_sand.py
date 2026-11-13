"""Deformable curved BAC shell spoon scooping 3D Drucker-Prager sand via two-way MPM coupling.

A pre-curved midedge BAC shell spoon (``add_cloth_mesh`` with clamped actuated handle) plunges, plows,
and lifts dry granular sand through ``SolverCoupledProxy`` (``proxy_particles`` deformable mesh collider)
while rendering fine sand grains with ``SolverImplicitMPM``'s ``render_grains``.

    uv run examples/mpm_spoon_sand.py
    uv run examples/mpm_spoon_sand.py --viewer null --test
"""

import newton
import newton.examples
import numpy as np
import warp as wp
from newton._src.solvers.coupled.solver_coupled import SolverCoupled
from newton._src.solvers.coupled.solver_coupled_proxy import SolverCoupledProxy
from newton.solvers import SolverImplicitMPM
from utils.common import MUTED, SIM, THEORY, Drive, inset

from dismech_newton import SolverDiSMech, flatten_state

GRAVITY = 9.81


def _box_mesh_data(center: tuple[float, float, float], hx: float, hy: float, hz: float, base: int):
    cx, cy, cz = center
    signs = np.array(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
        dtype=np.float32,
    )
    verts = signs * np.array([hx, hy, hz], dtype=np.float32) + np.array([cx, cy, cz], dtype=np.float32)
    tris = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    ) + base
    return verts, tris


def _tray_mesh(lx: float, ly: float, wh: float, wt: float, device) -> wp.Mesh:
    """Single watertight-box composite MPM mesh for sand bed floor and four container walls."""
    boxes = [
        ((0.0, 0.0, -0.5 * wt), 0.5 * lx + 2.0 * wt, 0.5 * ly + 2.0 * wt, 0.5 * wt),
        ((-0.5 * lx - 0.5 * wt, 0.0, 0.5 * wh), 0.5 * wt, 0.5 * ly + wt, 0.5 * wh),
        ((0.5 * lx + 0.5 * wt, 0.0, 0.5 * wh), 0.5 * wt, 0.5 * ly + wt, 0.5 * wh),
        ((0.0, -0.5 * ly - 0.5 * wt, 0.5 * wh), 0.5 * lx + 2.0 * wt, 0.5 * wt, 0.5 * wh),
        ((0.0, 0.5 * ly + 0.5 * wt, 0.5 * wh), 0.5 * lx + 2.0 * wt, 0.5 * wt, 0.5 * wh),
    ]
    vs, fs = [], []
    base = 0
    for c, hx, hy, hz in boxes:
        v, f = _box_mesh_data(c, hx, hy, hz, base)
        vs.append(v)
        fs.append(f)
        base += len(v)
    pts = np.vstack(vs)
    idx = np.vstack(fs).ravel()
    return wp.Mesh(
        points=wp.array(pts, dtype=wp.vec3, device=device),
        indices=wp.array(idx, dtype=wp.int32, device=device),
        velocities=wp.zeros(len(pts), dtype=wp.vec3, device=device),
    )


def _spoon_geometry(nx: int = 13, ny: int = 7, length: float = 0.16, width: float = 0.088):
    """Curved handle+bowl shell mesh in local spoon coordinates (pivot at handle root ``u = 0``)."""
    us = np.linspace(0.0, length, nx)
    vs = np.linspace(-0.5 * width, 0.5 * width, ny)
    U, V = np.meshgrid(us, vs, indexing="ij")
    bowl_start = 0.04
    s = np.clip((U - bowl_start) / (length - bowl_start), 0.0, 1.0)
    t = 2.0 * V / width
    # Smooth handle-to-bowl width flare & rounded tip taper; concave bowl pocket beyond ``bowl_start``.
    half_w = 0.5 * width * np.where(
        U <= bowl_start,
        0.38 + 0.62 * (U / bowl_start) ** 2,
        1.0 - 0.35 * s**4,
    )
    Y = t * half_w
    Z = np.where(
        U > bowl_start,
        -0.022 * np.sin(np.pi * s) * (1.0 - 0.90 * t**2),
        0.0,
    )
    verts = np.column_stack([U.ravel(), Y.ravel(), Z.ravel()])
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
            tris += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    handle_ids = [i * ny + j for i in range(3) for j in range(ny)]
    tip_ids = [(nx - 1) * ny + (ny // 2)]
    return verts, np.array(tris, dtype=np.int32), handle_ids, tip_ids


def _handle_pose(t: float) -> tuple[np.ndarray, float]:
    """Piecewise-smooth rigid handle target ``(pivot_xyz, pitch_rad)`` around ``+y`` (dip -> plow -> lift)."""
    knots = np.array([0.00, 0.10, 0.45, 1.00, 1.60])
    xs = np.array([-0.14, -0.14, -0.09, -0.02, -0.02])
    zs = np.array([0.145, 0.145, 0.088, 0.088, 0.175])
    deg = np.array([22.0, 22.0, 8.0, -24.0, -26.0])
    px = float(np.interp(t, knots, xs))
    pz = float(np.interp(t, knots, zs))
    pitch = float(np.deg2rad(np.interp(t, knots, deg)))
    return np.array([px, 0.0, pz]), pitch


def _transform_local(pts: np.ndarray, pivot: np.ndarray, pitch: float) -> np.ndarray:
    c, s = np.cos(pitch), np.sin(pitch)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return pts @ R.T + pivot


class Example:
    fps = 60
    substeps = 3
    rho_s = 1600.0
    mu_s = 0.65
    vs = 0.014
    bed_lx, bed_ly, bed_lz = 0.32, 0.20, 0.07

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.frame_dt = 1.0 / self.fps
        self.dt = self.frame_dt / self.substeps
        self.sim_time = 0.0

        b = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
        SolverDiSMech.register_custom_attributes(b)
        SolverImplicitMPM.register_custom_attributes(b)

        # Dry sand bed in [-lx/2, lx/2] x [-ly/2, ly/2] x [0, bed_lz].
        sp = self.vs * 0.5
        xs = np.arange(-0.5 * self.bed_lx + 0.5 * sp, 0.5 * self.bed_lx, sp)
        ys = np.arange(-0.5 * self.bed_ly + 0.5 * sp, 0.5 * self.bed_ly, sp)
        zs = np.arange(0.5 * sp, self.bed_lz, sp)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        sand_pos = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        self.n_sand = len(sand_pos)
        self.p_mass = float(self.rho_s * sp**3)
        b.add_particles(
            pos=sand_pos.tolist(),
            vel=[[0.0, 0.0, 0.0]] * self.n_sand,
            mass=[self.p_mass] * self.n_sand,
            radius=[0.4 * self.vs] * self.n_sand,
            custom_attributes={
                "mpm:friction": [self.mu_s] * self.n_sand,
                "mpm:young_modulus": [5.0e5] * self.n_sand,
                "mpm:poisson_ratio": [0.3] * self.n_sand,
            },
        )
        mpm_p = list(range(self.n_sand))

        # Visible container walls & ground plane (MPM collisions use ``tray_mesh`` below).
        b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        wt, wh = 0.02, 0.14
        wcfg = newton.ModelBuilder.ShapeConfig(mu=0.4)
        for px, py, hx, hy in [
            (-0.5 * self.bed_lx - 0.5 * wt, 0.0, 0.5 * wt, 0.5 * self.bed_ly + wt),
            (0.5 * self.bed_lx + 0.5 * wt, 0.0, 0.5 * wt, 0.5 * self.bed_ly + wt),
            (0.0, -0.5 * self.bed_ly - 0.5 * wt, 0.5 * self.bed_lx + 2.0 * wt, 0.5 * wt),
            (0.0, 0.5 * self.bed_ly + 0.5 * wt, 0.5 * self.bed_lx + 2.0 * wt, 0.5 * wt),
        ]:
            b.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(px, py, 0.5 * wh), wp.quat_identity()),
                hx=hx,
                hy=hy,
                hz=0.5 * wh,
                cfg=wcfg,
            )

        # Curved BAC shell spoon placed above sand in initial plunge pose.
        local_v, tris, handle_local, tip_local = _spoon_geometry()
        piv0, pitch0 = _handle_pose(0.0)
        init_v = _transform_local(local_v, piv0, pitch0)
        self.spoon_p = SolverDiSMech.add_cloth_mesh(
            b,
            vertices=init_v.tolist(),
            indices=tris.ravel().tolist(),
            density=6.0,
            particle_radius=0.002,
            tri_ke=5.0e4,
            tri_ka=5.0e4,
            tri_kd=10.0,
            edge_ke=0.0,
            bend_stiffness=4.0,
            damping=2.0e-2,
        ).tolist()
        for idx in handle_local:
            pid = self.spoon_p[idx]
            b.particle_flags[pid] &= ~newton.ParticleFlags.ACTIVE
            b.particle_mass[pid] = 0.0

        self.local_v = local_v
        self.handle_local = handle_local
        self.handle_pids = [self.spoon_p[i] for i in handle_local]
        self.tip_local = tip_local[0]
        self.tip_pid = self.spoon_p[self.tip_local]
        self.handle_rest = init_v[handle_local].copy()

        self.model = b.finalize()

        cfg = SolverImplicitMPM.Config()
        cfg.voxel_size = self.vs
        cfg.max_iterations = 25
        self.solver = SolverCoupledProxy(
            self.model,
            entries=[
                SolverCoupled.Entry(
                    name="soft",
                    solver=lambda v: SolverDiSMech(v, refactor=False),
                    particles=self.spoon_p,
                ),
                SolverCoupled.Entry(
                    name="mpm",
                    solver=lambda v: SolverImplicitMPM(v, config=cfg),
                    particles=mpm_p,
                    in_place=True,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="soft",
                        destination="mpm",
                        particles=self.spoon_p,
                        proxy_particles=self.spoon_p,
                        mode="lagged",
                        mass_scale=1.0,
                        collision_pipeline=lambda _m: None,
                    )
                ],
                iterations=1,
            ),
        )
        m_mpm = self.solver.view("mpm")
        dev = self.model.device
        # Locate local particle IDs of spoon vertices inside the MPM solver view.
        g2l = self.solver._entries["mpm"].particle_global_to_local.numpy()
        local_spoon_pids = [int(g2l[pid]) for pid in self.spoon_p]
        spoon_pts0 = self.model.particle_q.numpy()[self.spoon_p]
        self._spoon_mesh = wp.Mesh(
            points=wp.array(spoon_pts0, dtype=wp.vec3, device=dev),
            indices=wp.array(tris.ravel(), dtype=wp.int32, device=dev),
            velocities=wp.zeros(len(spoon_pts0), dtype=wp.vec3, device=dev),
        )
        self._tray_mesh = _tray_mesh(self.bed_lx, self.bed_ly, wh, wt, dev)
        self._mpm_solver = self.solver._entries["mpm"].solver
        self._mpm_solver.setup_collider(
            collider_meshes=[self._spoon_mesh, self._tray_mesh],
            collider_particle_ids=[local_spoon_pids, None],
            collider_margins=[0.004, 0.002],
            collider_friction=[0.4, 0.5],
            model=m_mpm,
        )

        self.state_0, self.state_1 = self.model.state(), self.model.state()
        flatten_state(self.state_0)
        flatten_state(self.state_1)

        # Track sand grains inside MPM entry for crisp granular rendering.
        self._mpm_state_prev = m_mpm.state()
        self._mpm_state_prev.particle_q.assign(self.solver._entries["mpm"].state_0.particle_q)
        self._grains = self._mpm_solver.sample_render_grains(self.solver._entries["mpm"].state_0, 2)

        dofs = [3 * pid + c for pid in self.handle_pids for c in range(3)]
        self.drive = Drive(self.model, dofs)
        self._prev_spoon_pos = init_v.copy()

        self.t_hist, self.work_hist, self.mass_hist, self.tip_hist, self.bal_hist = [], [], [], [], []
        self._cum_work = 0.0
        if getattr(viewer, "camera", None) is not None:
            viewer.set_model(self.model)
            viewer.set_camera(pos=wp.vec3(0.0, -0.36, 0.22), pitch=-22.0, yaw=90.0)

    def _substep(self):
        self.solver.step(self.state_0, self.state_1, None, None, self.dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        t0, t1 = self.sim_time, self.sim_time + self.frame_dt
        piv0, pitch0 = _handle_pose(t0)
        piv1, pitch1 = _handle_pose(t1)
        d0 = (_transform_local(self.local_v[self.handle_local], piv0, pitch0) - self.handle_rest).ravel()
        d1 = (_transform_local(self.local_v[self.handle_local], piv1, pitch1) - self.handle_rest).ravel()
        self.drive.set(d0, d1)

        # Snapshot previous MPM particle positions before stepping for grain advection.
        mpm_state_cur = self.solver._entries["mpm"].state_0
        self._mpm_state_prev.particle_q.assign(mpm_state_cur.particle_q)
        if mpm_state_cur.particle_qd is not None:
            self._mpm_state_prev.particle_qd.assign(mpm_state_cur.particle_qd)

        for sub in range(self.substeps):
            self.drive(self.state_0, (sub + 1) / self.substeps)
            self._substep()

        # Advance visual sand grains alongside particle deformation frames.
        self._mpm_solver.update_particle_frames(self._mpm_state_prev, mpm_state_cur, self.frame_dt)
        self._mpm_solver.update_render_grains(self._mpm_state_prev, mpm_state_cur, self._grains, self.frame_dt)

        # Compute physical scoop diagnostics from full particle states & MPM reaction.
        q_all = self.state_0.particle_q.numpy().astype(np.float64)
        v_all = self.state_0.particle_qd.numpy().astype(np.float64)
        q_sand = q_all[:self.n_sand]
        q_spoon = q_all[self.spoon_p]
        v_spoon = v_all[self.spoon_p]

        piv_t, pitch_t = _handle_pose(t1)
        target_spoon = _transform_local(self.local_v, piv_t, pitch_t)
        tip_defl = float(target_spoon[self.tip_local, 2] - q_spoon[self.tip_local, 2])

        # Scooped mass: sand carried cleanly above the initial bed surface (>= 2.5 cm clearance).
        scooped_mask = q_sand[:, 2] > self.bed_lz + 0.025
        lifted_mass = float(scooped_mask.sum() * self.p_mass)

        # Harvested particle coupling forces (transferred from MPM deformable mesh collider to soft entry).
        f_coupl = np.zeros((len(self.spoon_p), 3))
        if self.solver._proxy_particle_mappings:
            pmap = self.solver._proxy_particle_mappings[0]
            cf = getattr(pmap, "coupling_forces", None)
            if cf is not None:
                cf_np = cf.numpy()
                f_coupl = cf_np[:len(self.spoon_p)] if len(cf_np) == len(self.spoon_p) else cf_np[self.spoon_p]
        v_mean = 0.5 * ((q_spoon - self._prev_spoon_pos) / self.frame_dt + v_spoon)
        self._cum_work += float((-f_coupl * v_mean).sum() * self.frame_dt)
        self._prev_spoon_pos = q_spoon.copy()

        # Scooped payload vs vertical sand reaction residual (during free lift) / reaction norm.
        fz_sand_on_spoon = float(f_coupl[:, 2].sum())
        bal = float(abs(-fz_sand_on_spoon - lifted_mass * GRAVITY) / max(lifted_mass * GRAVITY, abs(fz_sand_on_spoon), 0.5))

        self.sim_time = t1
        self.t_hist.append(self.sim_time)
        self.work_hist.append(self._cum_work)
        self.mass_hist.append(lifted_mass * 1e3)  # grams
        self.tip_hist.append(tip_defl * 1e3)      # mm
        self.bal_hist.append(bal)

    def image(self, size: tuple[int, int] = (920, 230)) -> np.ndarray:
        """4-panel diagnostics: mechanical scoop ``Work``, lifted ``Mass``, ``Tip`` sag, and ``Balance``."""
        t = np.asarray(self.t_hist)

        def draw(axes):
            specs = [
                (axes[0], self.work_hist, "Work", "$W$ (J)", SIM),
                (axes[1], self.mass_hist, "Mass", "$m_{\\mathrm{lift}}$ (g)", SIM),
                (axes[2], self.tip_hist, "Tip", "$\\delta z$ (mm)", THEORY),
                (axes[3], self.bal_hist, "Balance", "$\\Delta F / F$", MUTED),
            ]
            for ax, y, title, ylabel, col in specs:
                ax.plot(t, y, color=col, linewidth=1.6)
                ax.set_title(title)
                ax.set_xlabel("$t$ (s)")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.25)

        return inset(draw, size, ncols=4)

    def render(self):
        if round(self.sim_time * self.fps) % 10 == 0 and self.t_hist:
            self.viewer.log_image("scoop", self.image())
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        sand_pts = self._grains[:self.n_sand].reshape((-1,))
        self.viewer.log_points("sand_grains", sand_pts, radii=wp.full(sand_pts.shape[0], 0.0022, dtype=float,
                                                                      device=self.model.device))
        self.viewer.end_frame()

    def test_final(self):
        q_all = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(q_all)), "non-finite particle positions in spoon-sand rollout"
        max_mass_g = max(self.mass_hist)
        final_mass_g = self.mass_hist[-1]
        max_tip_mm = max(abs(x) for x in self.tip_hist)
        final_tip_mm = abs(self.tip_hist[-1])
        assert max_mass_g > 8.0 and final_mass_g > 5.0, (
            f"spoon failed to carry sand clear of bed (peak={max_mass_g:.2f} g, final={final_mass_g:.2f} g)"
        )
        assert 0.3 < max_tip_mm < 40.0 and final_tip_mm < 35.0, (
            f"unexpected shell tip deflection: peak={max_tip_mm:.2f} mm, final={final_tip_mm:.2f} mm"
        )
        assert self.work_hist[-1] > 0.02, f"unexpectedly small scoop work: {self.work_hist[-1]:.4f} J"
        assert self.bal_hist[-1] < 0.35, f"lifted sand weight vs vertical spoon reaction mismatch: {self.bal_hist[-1]:.3f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=96)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
