import warp as wp
from newton import ModelBuilder
from newton.viewer import ViewerGL

from dismech_newton import ADMMSpringMassSolver


def main():
    dt = 1e-2
    builder = ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 0.4),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=10,
        dim_y=10,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
        add_springs=True,
        spring_ke=1000.0,
        spring_kd=100.0,
        fix_top=True,
    )
    # builder.add_ground_plane()
    ADMMSpringMassSolver.register_custom_attributes(builder)
    """builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        hx=0.3,
        hy=0.3,
        hz=0.2,
    )"""

    model = builder.finalize(device="cpu")

    # Manual friction changes
    model.particle_ke = 1e4
    model.particle_kd = 100.0
    model.particle_kf = 10.0
    model.particle_mu = 0.6

    state0 = model.state()
    state1 = model.state()
    contacts = model.contacts()

    solver = ADMMSpringMassSolver(model)

    viewer = ViewerGL()
    viewer.set_model(model)

    time = 0.0
    frame_rate = 60.0
    substeps = int(1.0 / frame_rate // dt)
    while True:
        with wp.ScopedTimer("frame"):
            for _ in range(substeps):
                model.collide(state0, contacts)
                solver.step(state0, state1, None, contacts, dt)
                state0, state1 = state1, state0
            viewer.begin_frame(time)
            viewer.log_state(state0)
            viewer.end_frame()
        time += dt


if __name__ == "__main__":
    main()
