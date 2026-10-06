"""A single triplet's inputs and a kernel evaluating its strains and derivatives."""

from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from dismech_newton.strains import (
    local_strain_derivatives,
    mat5_11f,
    mat11f,
    mat58f,
    strain_derivatives,
    strain_gradient,
    triplet_geometry,
    vec5f,
    vec8f,
    vec11f,
    vec36f,
)
from dismech_newton.triplet import _local_geometry, _reduction, mat88f

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


@wp.kernel
def _triplet_kernel(
    q: wp.array[vec11f], frames: wp.array[wp.vec3], ref_twist: float, l0e: float, l0f: float, sigma: vec5f,
    # outputs
    strain: wp.array[vec5f], J: wp.array[mat5_11f], H: wp.array2d[mat11f], grad: wp.array[vec11f],
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
        Js, Hs = strain_derivatives(g, unit)
        J[i] = Js
        H[i, s] = Hs
    grad[i] = strain_gradient(g, sigma)


@dataclass
class TripletEval:
    strain: np.ndarray  # (N, 5)
    J: np.ndarray  # (N, 5, 11)
    H: np.ndarray  # (N, 5, 11, 11), H[:, i] the Hessian of strain i
    grad: np.ndarray  # (N, 11), strain_gradient(sigma)


def eval_triplet(Q: np.ndarray, cfg: TripletConfig, device, sigma=np.zeros(5)) -> TripletEval:
    """Strains and derivatives at ``Q`` ``(N, 11)``, in float64."""
    Q = np.atleast_2d(Q)
    n = Q.shape[0]
    frames = np.stack([cfg.d1e, cfg.te_old, cfg.d1f, cfg.tf_old])
    strain = wp.zeros(n, dtype=vec5f, device=device)
    J = wp.zeros(n, dtype=mat5_11f, device=device)
    H = wp.zeros((n, 5), dtype=mat11f, device=device)
    grad = wp.zeros(n, dtype=vec11f, device=device)
    wp.launch(
        _triplet_kernel,
        dim=n,
        inputs=[
            wp.array(Q, dtype=vec11f, device=device), wp.array(frames, dtype=wp.vec3, device=device),
            cfg.ref_twist, cfg.l0e, cfg.l0f, vec5f(*sigma),
        ],
        outputs=[strain, J, H, grad],
        device=device,
    )
    return TripletEval(*(a.numpy().astype(np.float64) for a in (strain, J, H, grad)))


# -- the local variable z = [e, theta_e, f, theta_f - theta_e] ----------------------------

Z_EDGE_DOFS = [0, 1, 2, 4, 5, 6]
Z_THETA_DOFS = [3, 7]


def to_z(q: np.ndarray) -> np.ndarray:
    """``z`` of DOFs ``[x0, theta_e, x1, theta_f, x2]``."""
    q = np.atleast_2d(q)
    return np.column_stack([q[:, 4:7] - q[:, 0:3], q[:, 3], q[:, 8:11] - q[:, 4:7], q[:, 7] - q[:, 3]])


def unpack_sym8(packed: np.ndarray) -> np.ndarray:
    """``(..., 36)`` packed upper triangles (row-major, ``i <= j``) to ``(..., 8, 8)`` symmetric matrices."""
    i, j = np.triu_indices(8)
    out = np.zeros(packed.shape[:-1] + (8, 8))
    out[..., i, j] = packed
    out[..., j, i] = packed
    return out


@wp.kernel
def _local_kernel(
    z: wp.array[vec8f], frames: wp.array[wp.vec3], ref_twist: float, l0e: float, l0f: float, sigma: vec5f,
    # outputs
    strain: wp.array[vec5f], J: wp.array[mat58f], H: wp.array2d[vec36f], J_ref: wp.array[mat58f],
    H_ref: wp.array2d[mat88f], H_sigma: wp.array[vec36f],
):
    i = wp.tid()
    g = _local_geometry(z[i], frames[0], frames[1], frames[2], frames[3], ref_twist, l0e, l0f)
    strain[i] = g.strain
    T = _reduction()
    for s in range(5):
        unit = vec5f()
        unit[s] = 1.0
        Js, Hs = local_strain_derivatives(g, unit)
        J[i] = Js
        H[i, s] = Hs
        Jq, Hq = strain_derivatives(g, unit)
        J_ref[i] = Jq * T
        H_ref[i, s] = wp.transpose(T) * Hq * T
    Js, Hs = local_strain_derivatives(g, sigma)
    H_sigma[i] = Hs


@dataclass
class LocalEval:
    strain: np.ndarray  # (N, 5)
    J: np.ndarray  # (N, 5, 8), local_strain_derivatives
    H: np.ndarray  # (N, 5, 8, 8), unpacked, H[:, i] the Hessian of strain i
    J_ref: np.ndarray  # (N, 5, 8), strain_derivatives reduced: J T
    H_ref: np.ndarray  # (N, 5, 8, 8), T^T H T
    H_sigma: np.ndarray  # (N, 8, 8), sum_i sigma_i H_i, unpacked


def eval_local(Z: np.ndarray, cfg: TripletConfig, device, sigma=np.zeros(5)) -> LocalEval:
    """Strains and z-derivatives at ``Z`` ``(N, 8)``, in float64."""
    Z = np.atleast_2d(Z)
    n = Z.shape[0]
    frames = np.stack([cfg.d1e, cfg.te_old, cfg.d1f, cfg.tf_old])
    strain = wp.zeros(n, dtype=vec5f, device=device)
    J = wp.zeros(n, dtype=mat58f, device=device)
    H = wp.zeros((n, 5), dtype=vec36f, device=device)
    J_ref = wp.zeros(n, dtype=mat58f, device=device)
    H_ref = wp.zeros((n, 5), dtype=mat88f, device=device)
    H_sigma = wp.zeros(n, dtype=vec36f, device=device)
    wp.launch(
        _local_kernel,
        dim=n,
        inputs=[
            wp.array(Z, dtype=vec8f, device=device), wp.array(frames, dtype=wp.vec3, device=device),
            cfg.ref_twist, cfg.l0e, cfg.l0f, vec5f(*sigma),
        ],
        outputs=[strain, J, H, J_ref, H_ref, H_sigma],
        device=device,
    )
    out = [a.numpy().astype(np.float64) for a in (strain, J, H, J_ref, H_ref, H_sigma)]
    out[2] = unpack_sym8(out[2])
    out[5] = unpack_sym8(out[5])
    return LocalEval(*out)
