"""Step per-DOF inertia, external force, prediction, diagonal assembly, and theta extrapolation."""

import warp as wp
from newton import Model, State

from .layout import DOFS_MODULE


@wp.struct
class Dofs:
    """Per DOF its mass and whether it is fixed; the external force's inputs (gravity, ``particle_f`` on the first
    ``num_node_dofs``); ``1/dt^2``."""

    mass: wp.array[float]
    fixed: wp.array[wp.int32]
    gravity: wp.array[wp.vec3]
    particle_f: wp.array[wp.vec3]
    num_node_dofs: int
    inv_dt2: float


@wp.func
def _external(f: Dofs, i: int) -> float:
    """``m g + particle_f`` on node DOF ``i``."""
    if i < f.num_node_dofs:
        n = i // 3
        k = i - 3 * n
        return f.mass[i] * f.gravity[0][k] + f.particle_f[n][k]
    return 0.0


@wp.func
def inertia_grad(f: Dofs, i: int, r: float) -> float:
    """DOF ``i``'s part of the step's gradient at ``r = q - q_pred``: ``M r / dt^2 - f_ext``."""
    return f.mass[i] * f.inv_dt2 * r - _external(f, i)


@wp.func
def inertia_energy(f: Dofs, i: int, r: float) -> float:
    return 0.5 * f.mass[i] * f.inv_dt2 * r * r - _external(f, i) * r


@wp.kernel(module=DOFS_MODULE)
def _dof_mass_kernel(particle_mass: wp.array[float], edge_inertia: wp.array[float], hinge_inertia: wp.array[float],
                     mass: wp.array[float]):
    i = wp.tid()
    nd = 3 * particle_mass.shape[0]
    if i < nd:
        mass[i] = particle_mass[i // 3]
    elif i < nd + edge_inertia.shape[0]:
        mass[i] = edge_inertia[i - nd]
    else:
        mass[i] = hinge_inertia[i - nd - edge_inertia.shape[0]]


def dof_mass(model: Model, mass: wp.array) -> None:
    """The per-DOF masses into ``mass`` (differentiable): the particles', the twist and midedge angle inertias."""
    wp.launch(_dof_mass_kernel, dim=mass.shape[0], inputs=[model.particle_mass, model.dismech.edge_inertia,
              model.dismech.hinge_inertia], outputs=[mass], device=model.device)


def bind_dofs(f: Dofs, model: Model, mass: wp.array, fixed: wp.array, no_force: wp.array, state_in: State,
              dt: float) -> Dofs:
    """``f`` set for a step of ``dt`` from ``state_in`` (``no_force``: zeros for a state without ``particle_f``)."""
    f.mass, f.fixed, f.gravity, f.num_node_dofs = mass, fixed, model.gravity, 3 * model.particle_count
    f.inv_dt2 = 1.0 / (dt * dt)
    f.particle_f = state_in.particle_f if state_in.particle_f is not None else no_force
    return f


@wp.kernel(module=DOFS_MODULE)
def predict_kernel(fixed: wp.array[wp.int32], q0: wp.array[float], v0: wp.array[float], dt: float,
                   q_pred: wp.array[float], q: wp.array[float]):
    """``q_pred = q0 + dt v0``; given ``q``, also the initial guess (fixed DOFs keep ``q0``)."""
    i = wp.tid()
    q_pred[i] = q0[i] + dt * v0[i]
    if q.shape[0] > 0:
        q[i] = wp.where(fixed[i] != 0, q0[i], q_pred[i])


@wp.kernel(module=DOFS_MODULE, enable_backward=False)
def diagonal_kernel(f: Dofs, indptr: wp.array[wp.int32], vals: wp.array[wp.float64]):
    """``A``'s diagonal: ``+= M / dt^2``, ``1`` on fixed DOFs."""
    i = wp.tid()
    slot = indptr[i]
    if f.fixed[i] != 0:
        vals[slot] = wp.float64(1.0)
    else:
        vals[slot] = vals[slot] + wp.float64(f.mass[i] * f.inv_dt2)


@wp.kernel(module=DOFS_MODULE)
def end_kernel(fixed: wp.array[wp.int32], q0: wp.array[float], v0: wp.array[float], q_star: wp.array[float],
               theta: float, dt: float, q: wp.array[float], qd: wp.array[float]):
    """The step's end from its solution, the implicit Euler step ``theta dt`` (``q`` may be ``q_star``): the theta
    method's extrapolation to ``dt`` (``theta = 1``: backward Euler), ``0`` velocity on fixed DOFs."""
    i = wp.tid()
    d = q_star[i] - q0[i]
    free = fixed[i] == 0
    q[i] = wp.where(free, q0[i] + d / theta, q_star[i])
    qd[i] = wp.where(free, v0[i] + (d - theta * dt * v0[i]) / (theta * theta * dt), 0.0)
