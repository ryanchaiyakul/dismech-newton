import math

import newton
import newton.viewer
import warp as wp

from dismech_newton import DiSMechSolver
from dismech_newton.system import flatten_state


@wp.kernel
def drive_clamp_kernel(
    node0: int,
    x0: wp.vec3,
    x1: wp.vec3,
    twist_dof: int,
    theta: float,
    q: wp.array[float],
):
    """Prescribe the clamped end segment: both node positions and its twist angle."""
    for k in range(3):
        q[3 * node0 + k] = x0[k]
        q[3 * node0 + 3 + k] = x1[k]
    q[twist_dof] = theta


if __name__ == "__main__":
    length, radius, youngs_modulus, density = 1.0, 0.02, 1.0e7, 1000.0
    segments = 50
    frame_dt, substeps = 1.0 / 60.0, 4  # 2 diverges when the twisted rod snaps through
    sim_dt = frame_dt / substeps

    # Driven end: twist `turns` about the rod axis while sliding `compression` toward the other
    # end and shearing `shear` sideways, all ramped in over `ramp` seconds. Past about two turns
    # the rod buckles into a helix (Greenhill); the compression lets it loop into a plectoneme.
    turns, compression, shear, ramp = 4.0, 0.4, 0.15, 8.0

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    rod = newton.Rod.create_straight(
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        length,
        segment_count=segments,
        radius=radius,
        youngs_modulus=youngs_modulus,
        poissons_ratio=0.3,
    )
    bodies = DiSMechSolver.add_rod(builder, rod=rod, cfg=newton.ModelBuilder.ShapeConfig(density=density))
    DiSMechSolver.fix_segment(builder, bodies[0])
    DiSMechSolver.fix_segment(builder, bodies[-1])
    builder.add_ground_plane()
    model = builder.finalize()

    solver = DiSMechSolver(model)
    state_0, state_1, control = model.state(), model.state(), model.control()
    flatten_state(state_0)
    flatten_state(state_1)

    # The last edge's nodes are the last two particles; its twist DOF follows the node DOFs.
    end_node = model.particle_count - 2
    end_x0, end_x1 = (wp.vec3(*p) for p in state_0.particle_q.numpy()[end_node:])
    twist_dof = 3 * model.particle_count + (segments - 1)

    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)

    sim_time = 0.0
    while viewer.is_running():
        with wp.ScopedTimer("step"):
            for _ in range(substeps):
                t = sim_time + sim_dt
                s = min(t / ramp, 1.0)
                s = s * s * (3.0 - 2.0 * s)  # smoothstep
                offset = wp.vec3(-compression * s, shear * s, 0.0)
                wp.launch(
                    drive_clamp_kernel,
                    dim=1,
                    inputs=[end_node, end_x0 + offset, end_x1 + offset, twist_dof, 2.0 * math.pi * turns * s],
                    outputs=[state_0.dismech.q],
                )
                solver.step(state_0, state_1, control, None, sim_dt)
                state_0, state_1 = state_1, state_0
                sim_time += sim_dt

        viewer.begin_frame(sim_time)
        viewer.log_state(state_0)
        viewer.end_frame()
