import warp as wp

from newton import State, Model, Contacts, Control, ParticleFlags
from newton.solvers import SolverBase


@wp.func
def spring_force(
    pos_i: wp.vec3,
    pos_j: wp.vec3,
    vel_i: wp.vec3,
    vel_j: wp.vec3,
    stiffness: float,
    rest_length: float,
    damping: float,
) -> wp.vec3:
    e = pos_j - pos_i
    e_len = wp.length(e)
    if e_len > 1e-6:
        tangent = e / e_len
        f_elastic = (e_len - rest_length) * stiffness
        f_damping = wp.dot(vel_j - vel_i, tangent) * damping
        return tangent * (f_elastic + f_damping)
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def particle_collision_force(
    n: wp.vec3,
    v: wp.vec3,
    c: float,
    k_n: float,
    k_d: float,
    k_f: float,
    k_mu: float,
):
    vn = wp.dot(n, v)
    jn = c * k_n
    jd = wp.min(vn, 0.0) * k_d
    fn = jn + jd
    vt = v - n * vn
    vs = wp.length(vt)
    ft = wp.min(vs * k_f, k_mu * wp.abs(fn))
    return -n * fn - vt * ft


@wp.kernel
def compute_spring_forces(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    spring_indices: wp.array[wp.int32],
    spring_stiffness: wp.array[wp.float32],
    spring_rest_length: wp.array[wp.float32],
    spring_damping: wp.array[wp.float32],
    particle_f: wp.array[wp.vec3],
):
    tid = wp.tid()
    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]

    f_total = spring_force(
        particle_q[i],
        particle_q[j],
        particle_qd[i],
        particle_qd[j],
        spring_stiffness[tid],
        spring_rest_length[tid],
        spring_damping[tid],
    )

    wp.atomic_add(particle_f, i, f_total)
    wp.atomic_add(particle_f, j, -f_total)


@wp.kernel
def compute_particle_collision_forces(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    contact_count: wp.array[int],
    particle_id: wp.array[int],
    body_pos: wp.array[wp.vec3],
    body_vel: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    contact_max: int,
    kd: float,
    ke: float,
    kf: float,
    mu: float,
    particle_f: wp.array[wp.vec3],
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    p_idx = particle_id[tid]
    if (particle_flags[p_idx] & ParticleFlags.ACTIVE) == 0:
        return

    n = normal[tid]
    c = wp.dot(n, particle_q[p_idx] - body_pos[tid]) - particle_radius[p_idx]
    if c < 0.0:
        v_rel = particle_qd[p_idx] - body_vel[tid]
        f_contact = particle_collision_force(n, v_rel, c, ke, kd, kf, mu)
        wp.atomic_add(particle_f, p_idx, f_contact)


class SpringMassSolver(SolverBase):
    def __init__(self, model: Model):
        super().__init__(model)
        if model.spring_indices is None:
            raise ValueError(
                "Cannot initialize spring mass solver on model without springs"
            )

        self.n_springs = model.spring_count

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        if state_in.particle_f is None:
            raise ValueError("Cannot run spring mass solver on state without particles")

        state_in.particle_f.zero_()
        wp.launch(
            kernel=compute_spring_forces,
            dim=self.n_springs,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                self.model.spring_indices,
                self.model.spring_stiffness,
                self.model.spring_rest_length,
                self.model.spring_damping,
            ],
            outputs=[state_in.particle_f],
            device=self.model.device,
        )
        if contacts is not None:
            wp.launch(
                kernel=compute_particle_collision_forces,
                dim=contacts.soft_contact_max,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    self.model.particle_radius,
                    self.model.particle_flags,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_max,
                    self.model.soft_contact_ke,
                    self.model.soft_contact_kd,
                    self.model.soft_contact_kf,
                    self.model.soft_contact_mu,
                ],
                outputs=[state_in.particle_f],
                device=self.model.device,
            )

        self.integrate_particles(self.model, state_in, state_out, dt)
