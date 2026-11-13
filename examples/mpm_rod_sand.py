"""Granular resistive force theory (RFT): stiff and flexible DER rods dragged through dry Drucker-Prager sand.

Three vertical DER rods immersed to different depths (including a compliant finite-$EI$ beam) are translated
horizontally through dry granular sand via two-way ``SolverDiSMech`` <-> ``SolverImplicitMPM`` proxy coupling:
steady drag scales quadratically with immersion depth, $F_x(H) \\approx \\tfrac{1}{2} K_h \\rho_s g D H^2$.

    uv run examples/mpm_rod_sand.py
    uv run examples/mpm_rod_sand.py --viewer null --test
"""

import newton
import newton.examples
import numpy as np
import warp as wp
from newton._src.solvers.coupled.solver_coupled import SolverCoupled
from newton._src.solvers.coupled.solver_coupled_proxy import SolverCoupledProxy
from newton.solvers import SolverImplicitMPM
from utils.common import SIM, THEORY, Drive, inset

from dismech_newton import SolverDiSMech, flatten_state
from dismech_newton.coupling import mpm_collider_inertia

GRAVITY = 9.81


class Example:
    fps = 60
    substeps = 3
    rho_s = 1600.0
    mu_s = 0.65
    vs = 0.014
    rod_radius = 0.006  # D = 1.2 cm
    drag_speed = 0.10  # [m/s]
    depths = (0.05, 0.07, 0.09, 0.08)
    ys = (-0.105, -0.035, 0.035, 0.105)
    eis = (1.0, 1.0, 1.0, 0.025)  # fourth rod (H = 0.08 m) is compliant

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.frame_dt = 1.0 / self.fps
        self.dt = self.frame_dt / self.substeps
        self.sim_time = 0.0

        b = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
        SolverDiSMech.register_custom_attributes(b)
        SolverImplicitMPM.register_custom_attributes(b)

        # Dry sand bed in [-0.14, 0.14] x [-0.18, 0.18] x [0, 0.14] with enclosing container walls.
        lx, ly, lz = 0.28, 0.36, 0.14
        sp = self.vs * 0.5
        xs = np.arange(-0.5 * lx + 0.5 * sp, 0.5 * lx, sp)
        ys = np.arange(-0.5 * ly + 0.5 * sp, 0.5 * ly, sp)
        zs = np.arange(0.5 * sp, lz, sp)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pos = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        n_p = len(pos)
        p0 = b.particle_count
        b.add_particles(
            pos=pos.tolist(),
            vel=[[0.0, 0.0, 0.0]] * n_p,
            mass=[float(self.rho_s * sp**3)] * n_p,
            radius=[0.4 * self.vs] * n_p,
            custom_attributes={
                "mpm:friction": [self.mu_s] * n_p,
                "mpm:young_modulus": [5.0e5] * n_p,
                "mpm:poisson_ratio": [0.3] * n_p,
            },
        )
        mpm_p = list(range(p0, b.particle_count))

        # Bed floor & vertical side walls for the sand container.
        b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
        wt, wh = 0.02, 0.16
        wcfg = newton.ModelBuilder.ShapeConfig(mu=0.4)
        for x, y, hx, hy in [
            (-0.5 * lx - wt, 0.0, wt, 0.5 * ly + wt),
            (0.5 * lx + wt, 0.0, wt, 0.5 * ly + wt),
            (0.0, -0.5 * ly - wt, 0.5 * lx + 2.0 * wt, wt),
            (0.0, 0.5 * ly + wt, 0.5 * lx + 2.0 * wt, wt),
        ]:
            b.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(x, y, 0.5 * wh), wp.quat_identity()),
                hx=hx,
                hy=hy,
                hz=0.5 * wh,
                cfg=wcfg,
            )

        # Vertical DER rods: top two segments clamped & kinematically driven in +x.
        x0, n_seg, z_top = -0.08, 10, lz + 0.03
        self.rod_pids, self.rod_bodies, self.clamps = [], [], []
        rod_p, rod_b = [], []
        for y, h_imm, ei in zip(self.ys, self.depths, self.eis):
            z_bot = lz - h_imm
            h_seg = (z_top - z_bot) / n_seg
            zs = np.linspace(z_top, z_bot, n_seg + 1)
            pts = np.column_stack([np.full(n_seg + 1, x0), np.full(n_seg + 1, y), zs])
            rp0 = b.particle_count
            rod = newton.Rod(points=pts.tolist(), radius=self.rod_radius)
            rb, _ = SolverDiSMech.add_rod(
                b,
                rod=rod,
                stretch_stiffness=1.0e4,
                bend_stiffness=ei / h_seg,
                twist_stiffness=ei / h_seg,
                bend_damping=0.02 if ei < 0.1 else 0.0,
            )
            pids = list(range(rp0, b.particle_count))
            SolverDiSMech.fix_segment(b, rb[0])
            SolverDiSMech.fix_segment(b, rb[1])
            self.rod_pids.append(pids)
            self.rod_bodies.append(rb)
            self.clamps.extend(pids[:3])
            rod_p.extend(pids)
            rod_b.extend(rb)

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
                    particles=rod_p,
                    bodies=rod_b,
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
                        bodies=rod_b,
                        mode="lagged",
                        mass_scale=1.0,
                        collision_pipeline=lambda _m: None,
                    )
                ],
                iterations=1,
            ),
        )
        m_mpm = self.solver.view("mpm")
        bm, binv = mpm_collider_inertia(self.model, self.solver._entries["mpm"].body_local_to_global)
        body_ids = list(range(m_mpm.body_count))
        self.solver._entries["mpm"].solver.setup_collider(
            collider_body_ids=body_ids + [-1],
            collider_margins=[0.002] * (len(body_ids) + 1),
            model=m_mpm,
            body_mass=bm,
            body_inv_inertia=binv,
        )

        self.state_0, self.state_1 = self.model.state(), self.model.state()
        flatten_state(self.state_0)
        flatten_state(self.state_1)
        self.drive = Drive(self.model, [3 * pid for pid in self.clamps])

        self.t_hist, self.fx_hist = [], [[] for _ in self.depths]
        if getattr(viewer, "camera", None) is not None:
            viewer.set_model(self.model)
            viewer.set_camera(pos=wp.vec3(0.0, -0.45, 0.25), pitch=-22.0, yaw=90.0)

    def _substep(self):
        self.solver.step(self.state_0, self.state_1, None, None, self.dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        t0, t1 = self.sim_time, self.sim_time + self.frame_dt
        # Brief 0.1 s settling before horizontal stroke so initial hydrostatic bed stress stabilizes.
        d0 = self.drag_speed * max(0.0, t0 - 0.1)
        d1 = self.drag_speed * max(0.0, t1 - 0.1)
        self.drive.set(np.full(len(self.clamps), d0), np.full(len(self.clamps), d1))
        for sub in range(self.substeps):
            self.drive(self.state_0, (sub + 1) / self.substeps)
            self._substep()
        w = self.solver._proxy_mappings[0].coupling_forces.numpy()
        self.sim_time = t1
        self.t_hist.append(self.sim_time)
        for i, rb in enumerate(self.rod_bodies):
            self.fx_hist[i].append(-float(w[rb, 0].sum()))

    def _steady_drags(self) -> tuple[np.ndarray, float]:
        n_tail = max(1, len(self.t_hist) // 3)
        drags = np.array([float(np.mean(h[-n_tail:])) for h in self.fx_hist])
        h_arr = np.array(self.depths)
        denom = 0.5 * self.rho_s * GRAVITY * (2.0 * self.rod_radius) * h_arr**2
        stiff = np.array([ei > 0.1 for ei in self.eis])
        kh = float(np.mean(drags[stiff] / denom[stiff]))
        return drags, kh

    def image(self, size: tuple[int, int] = (400, 400)) -> np.ndarray:
        """Plot steady granular drag $F_x(H)$ against quadratic RFT $0.5 K_h \\rho_s g D H^2$."""
        drags, kh = self._steady_drags()

        def draw(ax):
            stiff = np.array([ei > 0.1 for ei in self.eis])
            h_dense = np.linspace(0.0, 0.105, 80)
            f_th = 0.5 * kh * self.rho_s * GRAVITY * (2.0 * self.rod_radius) * h_dense**2
            ax.plot(h_dense * 1e2, f_th, "--", color=THEORY,
                    label=rf"RFT ($K_h={kh:.2f}$, $\rho={self.rho_s:.0f}$)")
            ax.plot(np.array(self.depths)[stiff] * 1e2, drags[stiff], "o", color=SIM, label="SIM stiff")
            ax.plot(np.array(self.depths)[~stiff] * 1e2, drags[~stiff], "s", color=SIM,
                    label=rf"SIM ($EI={self.eis[-1]:.3f}$ N$\cdot$m$^2$)")
            ax.set_xlabel("$H$ (cm)")
            ax.set_ylabel("$F_x$ (N)")
            ax.set_title("Drag")
            ax.legend(frameon=False, fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.25)

        return inset(draw, size)

    def render(self):
        if round(self.sim_time * self.fps) % 10 == 0 and self.t_hist:
            self.viewer.log_image("drag", self.image())
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        drags, kh = self._steady_drags()
        stiff = np.array([ei > 0.1 for ei in self.eis])
        assert np.all(np.isfinite(drags)) and np.all(drags > 0.2), f"non-physical drags: {drags}"
        h_stiff = np.array(self.depths)[stiff]
        d_stiff = drags[stiff]
        slope = float(np.polyfit(np.log(h_stiff), np.log(d_stiff), 1)[0])
        pred = 0.5 * kh * self.rho_s * GRAVITY * (2.0 * self.rod_radius) * h_stiff**2
        rel = float(np.max(np.abs(d_stiff - pred) / pred))
        # Check compliant fourth rod deflects backward relative to its clamp under sand drag.
        q = self.state_0.particle_q.numpy()
        p_flex = self.rod_pids[-1]
        tip_lag = float(q[p_flex[0], 0] - q[p_flex[-1], 0])
        assert 0.8 < kh < 5.0, f"unexpected RFT coefficient K_h={kh:.3f}"
        assert 1.5 < slope < 2.2, f"drag depth slope {slope:.3f} far from quadratic RFT (drags={d_stiff})"
        assert rel < 0.20, f"drag vs RFT parabola mismatch {rel:.3f} ({d_stiff} vs {pred})"
        assert tip_lag > 0.002, f"compliant rod failed to deflect under granular drag: {tip_lag:.4f} m"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=90)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
