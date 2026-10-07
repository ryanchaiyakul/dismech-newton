"""Gaussian splats skinned to rods, for rendering a rollout and differentiating through the render.

Each splat belongs to one edge and is rigid in that edge's material frame ``(m1, m2, t)``, the frame the capsule
proxies take (seams at the nodes are expected). The canonical parameters are constant in time; the skin maps a state
to world-space means and covariances in a kernel that records on a ``wp.Tape``. It reads the flat ``q`` (node
positions and twists) and ``edge_d1_q``, both seeds of the step adjoint, so a loss on the splats backpropagates through
:meth:`DiSMechSolver.step` with no solver changes.

Covariances are returned as ``cov6 = (xx, xy, xz, yy, yz, zz)`` (``cov3D_precomp`` of the Inria 3DGS rasterizer),
smooth in the state where quaternions are not.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import warp as wp
from newton import Model

from .strains import material_frame

vec6 = wp.types.vector(6, float)


@wp.kernel
def skin_kernel(
    q: wp.array[float], edge_d1: wp.array[wp.vec3], edge_node0: wp.array[wp.int32], edge_node1: wp.array[wp.int32],
    twist0: int, splat_edge: wp.array[wp.int32], splat_s: wp.array[float], splat_uv: wp.array[wp.vec2],
    splat_rotation: wp.array[wp.mat33], splat_log_scale: wp.array[wp.vec3],
    # outputs
    means: wp.array[wp.vec3], cov6: wp.array[vec6],
):
    """One thread per splat; ``q[twist0 + e]`` is edge ``e``'s twist."""
    i = wp.tid()
    e = splat_edge[i]
    n0 = edge_node0[e]
    n1 = edge_node1[e]
    x0 = wp.vec3(q[3 * n0], q[3 * n0 + 1], q[3 * n0 + 2])
    x1 = wp.vec3(q[3 * n1], q[3 * n1 + 1], q[3 * n1 + 2])
    t = wp.normalize(x1 - x0)
    m1, m2 = material_frame(edge_d1[e], t, q[twist0 + e])
    s = splat_s[i]
    uv = splat_uv[i]
    means[i] = (1.0 - s) * x0 + s * x1 + uv[0] * m1 + uv[1] * m2
    # fmt: off
    frame = wp.mat33(
        m1[0], m2[0], t[0],
        m1[1], m2[1], t[1],
        m1[2], m2[2], t[2],
    )
    # fmt: on
    R = frame * splat_rotation[i]
    ls = splat_log_scale[i]
    C = R * wp.diag(wp.vec3(wp.exp(2.0 * ls[0]), wp.exp(2.0 * ls[1]), wp.exp(2.0 * ls[2]))) * wp.transpose(R)
    cov6[i] = vec6(C[0, 0], C[0, 1], C[0, 2], C[1, 1], C[1, 2], C[2, 2])


@dataclass
class Splats:
    """Canonical splats, one entry per splat, constant in time.

    Attributes:
        edge: The owning edge.
        s: Position along the edge, 0 at ``edge_node0``, 1 at ``edge_node1``.
        uv: Offset from the centreline along ``(m1, m2)``.
        rotation: Local axes in the edge frame ``(m1, m2, t)`` (columns).
        log_scale: Log standard deviations along the local axes.
    """

    edge: wp.array
    s: wp.array
    uv: wp.array
    rotation: wp.array
    log_scale: wp.array

    @classmethod
    def from_numpy(cls, edge, s, uv, rotation, log_scale, device=None, requires_grad: bool = False) -> "Splats":
        params = {name: wp.array(np.asarray(value, dtype=np.float32), dtype=dtype, device=device,
                                 requires_grad=requires_grad)
                  for name, value, dtype in (("s", s, float), ("uv", uv, wp.vec2), ("rotation", rotation, wp.mat33),
                                             ("log_scale", log_scale, wp.vec3))}
        return cls(edge=wp.array(np.asarray(edge, dtype=np.int32), dtype=wp.int32, device=device), **params)

    @property
    def params(self) -> dict[str, wp.array]:
        """The differentiable parameters (all but ``edge``)."""
        return {"s": self.s, "uv": self.uv, "rotation": self.rotation, "log_scale": self.log_scale}

    def numpy(self) -> SimpleNamespace:
        """The parameters as float64 NumPy arrays."""
        return SimpleNamespace(edge=self.edge.numpy(), **{k: a.numpy().astype(np.float64) for k, a in self.params.items()})

    def __len__(self) -> int:
        return self.edge.shape[0]


def tube(model: Model, rings: int, per_ring: int, radius: float, sigma, device=None,
         requires_grad: bool = False) -> tuple[Splats, np.ndarray]:
    """Rings of splats on every edge's surface: ``rings`` evenly along the edge, ``per_ring`` evenly around it,
    at ``radius``; ``sigma`` = local standard deviations (radial, tangential, along ``t``).

    Returns the splats and each one's index around its ring (0 lies along ``m1``), e.g. to colour a stripe."""
    E = model.dismech.edge_length.shape[0]
    edge, s, k = (a.ravel() for a in np.meshgrid(np.arange(E), (np.arange(rings) + 0.5) / rings, np.arange(per_ring),
                                                 indexing="ij"))
    phi = 2.0 * np.pi * k / per_ring
    c, sn = np.cos(phi), np.sin(phi)
    rotation = np.zeros((len(edge), 3, 3))
    rotation[:, :2, 0] = np.stack([c, sn], 1)  # radial
    rotation[:, :2, 1] = np.stack([-sn, c], 1)  # tangential
    rotation[:, 2, 2] = 1.0
    log_scale = np.tile(np.log(sigma), (len(edge), 1))
    splats = Splats.from_numpy(edge, s, radius * np.stack([c, sn], 1), rotation, log_scale,
                               device=device or model.device, requires_grad=requires_grad)
    return splats, k


def skin(model: Model, splats: Splats, q: wp.array, edge_d1: wp.array,
         out: tuple[wp.array, wp.array] | None = None) -> tuple[wp.array, wp.array]:
    """``(means, cov6)`` of ``splats`` at the flat DOFs ``q`` and reference directors ``edge_d1`` (a state's
    ``dismech.q`` and ``dismech.edge_d1_q``, after :func:`flatten_state`); records on an active ``wp.Tape``.

    ``out``: arrays to write (else new ones, with ``requires_grad`` if any input has it). A taped rollout should give
    every frame its own outputs."""
    if out is None:
        grad = any(a.requires_grad for a in (q, edge_d1, *splats.params.values()))
        n = len(splats)
        out = (wp.zeros(n, dtype=wp.vec3, device=q.device, requires_grad=grad),
               wp.zeros(n, dtype=vec6, device=q.device, requires_grad=grad))
    d = model.dismech
    wp.launch(skin_kernel, dim=len(splats),
              inputs=[q, edge_d1, d.edge_node0, d.edge_node1, 3 * model.particle_count, splats.edge, splats.s,
                      splats.uv, splats.rotation, splats.log_scale],
              outputs=list(out), device=q.device)
    return out
