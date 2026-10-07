"""A single triplet's inputs, the test cases, and a kernel evaluating its strains and derivatives."""

import functools
from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from dismech_newton.strains import (
    local_strain_derivatives,
    mat58f,
    strain_gradient,
    triplet_geometry,
    vec5f,
    vec11f,
    vec36f,
)

NODE_DOFS = [0, 1, 2, 4, 5, 6, 8, 9, 10]
THETA_DOFS = [3, 7]


def _unit(v):
    return v / np.linalg.norm(v)


def _perpendicular(t, rng):
    a = rng.normal(size=3)
    return _unit(a - (a @ t) * t)


def _rotate(v, axis, angle):
    """Rodrigues' rotation."""
    return v * np.cos(angle) + np.cross(axis, v) * np.sin(angle) + axis * (axis @ v) * (1.0 - np.cos(angle))


@dataclass(frozen=True)
class TripletConfig:
    """DOFs ``[x0, theta_e, x1, theta_f, x2]`` and the start-of-step frames."""

    q: np.ndarray
    d1e: np.ndarray
    te_old: np.ndarray
    d1f: np.ndarray
    tf_old: np.ndarray
    ref_twist: float
    l0e: float
    l0f: float

    @classmethod
    def current(cls, x0, x1, x2, theta_e=0.0, theta_f=0.0, ref_twist=0.0, l0e=None, l0f=None, seed=0):
        """Frames current at ``q``; rest lengths default to the edge lengths."""
        x0, x1, x2 = (np.asarray(x, dtype=np.float64) for x in (x0, x1, x2))
        rng = np.random.default_rng(seed)
        te, tf = _unit(x1 - x0), _unit(x2 - x1)
        return cls(
            q=np.concatenate([x0, [theta_e], x1, [theta_f], x2]),
            d1e=_perpendicular(te, rng), te_old=te, d1f=_perpendicular(tf, rng), tf_old=tf,
            ref_twist=ref_twist,
            l0e=np.linalg.norm(x1 - x0) if l0e is None else l0e,
            l0f=np.linalg.norm(x2 - x1) if l0f is None else l0f,
        )

    @property
    def edge_length(self) -> float:
        return 0.5 * (np.linalg.norm(self.q[4:7] - self.q[0:3]) + np.linalg.norm(self.q[8:11] - self.q[4:7]))

    @property
    def dof_scale(self) -> np.ndarray:
        """Per-DOF units (edge length on nodes, radian on twists) to compare blocks on one scale."""
        s = np.full(11, self.edge_length)
        s[THETA_DOFS] = 1.0
        return s

    def steps(self, rel: float) -> np.ndarray:
        return rel * self.dof_scale

    def lagged(self, angle: float, seed: int = 0) -> "TripletConfig":
        """Old tangents and directors rotated by ``angle``, as after ``q`` has moved."""
        rng = np.random.default_rng(seed)
        out = {}
        for d1, t, name in ((self.d1e, self.te_old, "e"), (self.d1f, self.tf_old, "f")):
            axis = _perpendicular(t, rng)
            out[f"d1{name}"] = _rotate(d1, axis, angle)
            out[f"t{name}_old"] = _rotate(t, axis, angle)
        return replace(self, **out)


def z_map() -> np.ndarray:
    """``R`` (8 x 11): ADMM's ``z = [x1 - x0, theta_e, x2 - x1, theta_f - theta_e] = R q``."""
    R = np.zeros((8, 11))
    for k in range(3):
        R[k, k], R[k, 4 + k] = -1.0, 1.0
        R[4 + k, 4 + k], R[4 + k, 8 + k] = -1.0, 1.0
    R[3, 3] = 1.0
    R[7, 3], R[7, 7] = -1.0, 1.0
    return R


def unpack_sym8(packed: np.ndarray) -> np.ndarray:
    """``(..., 36)`` packed upper triangles (row-major, ``i <= j``) to ``(..., 8, 8)`` symmetric matrices."""
    i, j = np.triu_indices(8)
    out = np.zeros(packed.shape[:-1] + (8, 8))
    out[..., i, j] = packed
    out[..., j, i] = packed
    return out


@wp.kernel
def _triplet_kernel(
    q: wp.array[vec11f], frames: wp.array[wp.vec3], ref_twist: float, l0e: float, l0f: float, sigma: vec5f,
    # outputs
    strain: wp.array[vec5f], J: wp.array[mat58f], H: wp.array2d[vec36f], H_sigma: wp.array[vec36f],
    grad: wp.array[vec11f],
):
    i = wp.tid()
    x = q[i]
    g = triplet_geometry(
        wp.vec3(x[0], x[1], x[2]), wp.vec3(x[4], x[5], x[6]), wp.vec3(x[8], x[9], x[10]), x[3], x[7],
        frames[0], frames[1], frames[2], frames[3], ref_twist, l0e, l0f,
    )
    strain[i] = g.strain
    for s in range(5):
        unit = vec5f()
        unit[s] = 1.0
        Js, Hs = local_strain_derivatives(g, unit)
        J[i] = Js
        H[i, s] = Hs
    _J, Hw = local_strain_derivatives(g, sigma)
    H_sigma[i] = Hw
    grad[i] = strain_gradient(g, sigma)


@dataclass
class TripletEval:
    strain: np.ndarray  # (N, 5)
    J: np.ndarray  # (N, 5, 11)
    H: np.ndarray  # (N, 5, 11, 11), H[:, i] the Hessian of strain i
    H_sigma: np.ndarray  # (N, 11, 11), local_strain_derivatives(sigma)'s Hessian
    grad: np.ndarray  # (N, 11), strain_gradient(sigma)


def eval_triplet(Q: np.ndarray, cfg: TripletConfig, sigma=np.zeros(5)) -> TripletEval:
    """Strains and derivatives at ``Q`` ``(N, 11)``, in float64: the z-derivatives mapped to the DOFs by ``R``."""
    Q = np.atleast_2d(Q)
    n = Q.shape[0]
    frames = np.stack([cfg.d1e, cfg.te_old, cfg.d1f, cfg.tf_old])
    strain = wp.zeros(n, dtype=vec5f)
    J = wp.zeros(n, dtype=mat58f)
    H = wp.zeros((n, 5), dtype=vec36f)
    H_sigma = wp.zeros(n, dtype=vec36f)
    grad = wp.zeros(n, dtype=vec11f)
    wp.launch(
        _triplet_kernel,
        dim=n,
        inputs=[wp.array(Q, dtype=vec11f), wp.array(frames, dtype=wp.vec3), cfg.ref_twist, cfg.l0e, cfg.l0f,
                vec5f(*sigma)],
        outputs=[strain, J, H, H_sigma, grad],
    )
    R = z_map()
    strain, J, H, H_sigma, grad = (a.numpy().astype(np.float64) for a in (strain, J, H, H_sigma, grad))
    return TripletEval(strain, J @ R, R.T @ unpack_sym8(H) @ R, R.T @ unpack_sym8(H_sigma) @ R, grad)


# -- cases ----------------------------------------------------------------------------------


def _random(seed: int) -> TripletConfig:
    rng = np.random.default_rng(seed)
    x0 = 0.1 * rng.normal(size=3)
    x1 = x0 + 0.1 * (np.array([1.0, 0.0, 0.0]) + 0.4 * rng.normal(size=3))
    x2 = x1 + 0.1 * (np.array([1.0, 0.0, 0.0]) + 0.4 * rng.normal(size=3))
    theta_e, theta_f = rng.uniform(-np.pi, np.pi, size=2)
    l0e, l0f = 0.1 * rng.uniform(0.8, 1.2, size=2)
    return TripletConfig.current(x0, x1, x2, theta_e, theta_f, ref_twist=rng.normal(), l0e=l0e, l0f=l0f, seed=seed)


def _bend(angle: float, l: float = 0.1):
    """Two edges of length ``l`` bent by ``angle`` in the xy-plane."""
    return (0.0, 0.0, 0.0), (l, 0.0, 0.0), (l + l * np.cos(angle), l * np.sin(angle), 0.0)


CASES = {
    "straight": lambda: TripletConfig.current(*_bend(0.0)),
    "bent": lambda: TripletConfig.current(*_bend(np.radians(30.0))),
    "sharp_bend": lambda: TripletConfig.current(*_bend(np.radians(120.0))),
    "twisted": lambda: TripletConfig.current(*_bend(np.radians(30.0)), theta_e=0.4, theta_f=-0.9, ref_twist=0.5),
    "stretched": lambda: TripletConfig.current(*_bend(np.radians(45.0)), l0e=0.08, l0f=0.13),
    **{f"random-{i}": functools.partial(_random, i) for i in range(3)},
}
