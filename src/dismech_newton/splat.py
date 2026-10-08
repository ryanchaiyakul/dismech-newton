"""Gaussian splats skinned to rods, for rendering a rollout and differentiating through the render.

Each splat belongs to one edge and is rigid in that edge's material frame ``(m1, m2, t)``, the frame the capsule
proxies take (seams at the nodes are expected). The canonical parameters are constant in time; the skin maps a state
to world-space means and covariances in a kernel that records on a ``wp.Tape``. It reads the flat ``q`` (node
positions and twists) and ``edge_d1_q``, both seeds of the step adjoint, so a loss on the splats backpropagates through
:meth:`DiSMechSolver.step` with no solver changes.

Covariances are returned as ``cov6 = (xx, xy, xz, yy, yz, zz)`` (``cov3D_precomp`` of the Inria 3DGS rasterizer),
smooth in the state where quaternions are not.

Export (not differentiable; colours and opacities are the appearance the skin leaves out): :func:`edge_gaussians`
gives one :class:`newton.Gaussian` per edge in its proxy's frame, for ``builder.add_shape_gaussian(body, ...)``;
:func:`export_usd` writes a rollout as OpenUSD ``ParticleField3DGaussianSplat`` prims (``usd-core``, in the ``splat``
extra).
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import warp as wp
from newton import Gaussian, Model
from scipy.spatial.transform import Rotation

from .strains import material_frame

vec6 = wp.types.vector(6, float)
SH_C0 = 0.28209479177387814  # the degree-0 spherical harmonic: colour = SH_C0 * coefficient + 0.5


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


# -- export -----------------------------------------------------------------------------------


@wp.kernel
def _proxy_pose_kernel(q: wp.array[float], edge_d1: wp.array[wp.vec3], edge_node0: wp.array[wp.int32],
                       edge_node1: wp.array[wp.int32], twist0: int, poses: wp.array[wp.transform]):
    """As ``pose_proxies_kernel``: the midpoint, local ``(x, y, z)`` along ``(m1, m2, t)``."""
    e = wp.tid()
    n0 = edge_node0[e]
    n1 = edge_node1[e]
    x0 = wp.vec3(q[3 * n0], q[3 * n0 + 1], q[3 * n0 + 2])
    x1 = wp.vec3(q[3 * n1], q[3 * n1 + 1], q[3 * n1 + 2])
    t = wp.normalize(x1 - x0)
    m1, m2 = material_frame(edge_d1[e], t, q[twist0 + e])
    # fmt: off
    R = wp.mat33(
        m1[0], m2[0], t[0],
        m1[1], m2[1], t[1],
        m1[2], m2[2], t[2],
    )
    # fmt: on
    poses[e] = wp.transform(0.5 * (x0 + x1), wp.quat_from_matrix(R))


def proxy_poses(model: Model, q, edge_d1) -> np.ndarray:
    """Every edge's proxy pose ``(E, 7)`` (position, quaternion ``(x, y, z, w)``) at the flat DOFs ``q`` and reference
    directors ``edge_d1``, as the solver poses its proxies; needs none."""
    d, dev = model.dismech, model.device
    poses = wp.empty(d.edge_length.shape[0], dtype=wp.transform, device=dev)
    wp.launch(_proxy_pose_kernel, dim=poses.shape[0],
              inputs=[wp.array(q, dtype=float, device=dev), wp.array(edge_d1, dtype=wp.vec3, device=dev),
                      d.edge_node0, d.edge_node1, 3 * model.particle_count],
              outputs=[poses], device=dev)
    return poses.numpy().astype(np.float64)


def _axes(cov6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quaternions ``(x, y, z, w)`` and standard deviations of covariances given as cov6 ``(N, 6)``."""
    cov = np.zeros((len(cov6), 3, 3))
    i, j = np.triu_indices(3)
    cov[:, i, j] = cov6
    cov[:, j, i] = cov6
    w, V = np.linalg.eigh(cov)
    V[:, :, 2] *= np.sign(np.linalg.det(V))[:, None]  # proper rotations
    return Rotation.from_matrix(V).as_quat(), np.sqrt(np.maximum(w, 0.0))


def _local(splats: Splats, edge_length) -> SimpleNamespace:
    """Each splat in its edge's proxy frame (origin at the midpoint, axes ``(m1, m2, t)``) at the rest length."""
    p = splats.numpy()
    position = np.column_stack([p.uv, (p.s - 0.5) * np.asarray(edge_length, dtype=np.float64)[p.edge]])
    return SimpleNamespace(edge=p.edge, position=position, quat=Rotation.from_matrix(p.rotation).as_quat(),
                           scale=np.exp(p.log_scale))


def _appearance(splats: Splats, colours, opacity) -> tuple[np.ndarray, np.ndarray]:
    """Degree-0 SH coefficients ``(S, 3)`` and opacities ``(S,)``."""
    sh = (np.broadcast_to(np.asarray(colours, dtype=np.float64), (len(splats), 3)) - 0.5) / SH_C0
    return sh, np.broadcast_to(np.asarray(opacity, dtype=np.float64), (len(splats),))


def _by_edge(edge: np.ndarray, count: int, *arrays) -> list[tuple[np.ndarray, ...]]:
    """Per edge ``0 .. count - 1``: its rows of each of ``arrays`` (in their order)."""
    order = np.argsort(edge, kind="stable")
    cuts = np.searchsorted(edge[order], np.arange(1, count))
    return list(zip(*(np.split(np.asarray(a)[order], cuts) for a in arrays)))


def edge_gaussians(splats: Splats, edge_length, colours, opacity, **kwargs) -> list[Gaussian]:
    """One :class:`newton.Gaussian` per edge (``edge_length``: the rest lengths), in the frame of the edge's proxy, to
    ride on it: ``builder.add_shape_gaussian(body, gaussian=...)``. Rigid per edge at the rest length: stretch moves
    the splats a little against :func:`skin`.

    Args:
        colours: RGB in ``[0, 1]``, ``(S, 3)`` or one for all.
        opacity: ``(S,)`` or one for all.
        kwargs: For :class:`newton.Gaussian` (``min_response``, ``sorting_mode``).
    """
    g = _local(splats, edge_length)
    sh, opacity = _appearance(splats, colours, opacity)
    return [Gaussian(*part, sh_degree=0, **kwargs)
            for part in _by_edge(g.edge, len(edge_length), g.position, g.quat, g.scale, opacity, sh)]


def _set_splats(field, position, quat, scale, time=None) -> None:
    """Positions, orientations, scales and extent at ``time`` (``None``: the default value, what static readers
    such as Newton's take)."""
    from pxr import Usd, Vt  # noqa: PLC0415

    at = Usd.TimeCode.Default() if time is None else time
    r = 3.0 * np.max(scale, axis=1, keepdims=True)
    field.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(position, dtype=np.float32)), at)
    field.GetOrientationsAttr().Set(Vt.QuatfArray.FromNumpy(np.asarray(quat, dtype=np.float32)), at)  # (x, y, z, w)
    field.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(scale, dtype=np.float32)), at)
    extent = np.stack([(position - r).min(0), (position + r).max(0)]).astype(np.float32)
    field.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(extent), at)


def _set_appearance(field, sh, opacity) -> None:
    from pxr import Vt  # noqa: PLC0415

    field.GetOpacitiesAttr().Set(Vt.FloatArray.FromNumpy(np.asarray(opacity, dtype=np.float32)))
    field.GetRadianceSphericalHarmonicsDegreeAttr().Set(0)
    field.GetRadianceSphericalHarmonicsCoefficientsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(sh, dtype=np.float32)))


def export_usd(path, model: Model, splats: Splats, colours, opacity, frames, fps: float, baked: bool = False,
               root: str = "/rod"):
    """Write a rollout's splats to the USD file ``path`` (``.usda`` or ``.usdc``) as ``ParticleField3DGaussianSplat``
    prims (z up, metres, one time code per frame; the first frame is also the default value). Returns the stage.

    - Rigged (default): one ``Xform`` per edge, ``<root>/edge_<e>``, its translate and orient time-sampled to the
      proxy pose, holding a static field of that edge's splats (as :func:`edge_gaussians`). Compact; rigid per edge.
    - ``baked``: one field ``<root>/splats`` whose positions, orientations and scales are time-sampled from
      :func:`skin` (exact, stretch included).

    Args:
        frames: ``(q, edge_d1)`` per frame: a state's ``dismech.q`` and ``dismech.edge_d1_q`` (NumPy).
        colours, opacity: As :func:`edge_gaussians`.
    """
    from pxr import Gf, Usd, UsdGeom, UsdVol  # noqa: PLC0415

    frames = list(frames)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(frames) - 1)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, root).GetPrim())
    sh, opacity = _appearance(splats, colours, opacity)
    if baked:
        field = UsdVol.ParticleField3DGaussianSplat.Define(stage, f"{root}/splats")
        _set_appearance(field, sh, opacity)
        dev = model.device
        q = wp.empty(len(frames[0][0]), dtype=float, device=dev)
        d1 = wp.empty(len(frames[0][1]), dtype=wp.vec3, device=dev)
        out = skin(model, splats, q, d1)
        for k, (qk, d1k) in enumerate(frames):
            q.assign(np.asarray(qk, dtype=np.float32))
            d1.assign(np.asarray(d1k, dtype=np.float32))
            skin(model, splats, q, d1, out=out)
            means, (quat, scale) = out[0].numpy(), _axes(out[1].numpy().astype(np.float64))
            for time in (None, 0) if k == 0 else (k,):
                _set_splats(field, means, quat, scale, time)
    else:
        g = _local(splats, model.dismech.edge_length.numpy())
        poses = np.stack([proxy_poses(model, q, d1) for q, d1 in frames])  # (frames, edges, 7)
        parts = _by_edge(g.edge, poses.shape[1], g.position, g.quat, g.scale, sh, opacity)
        for e, (position, quat, scale, sh_e, opacity_e) in enumerate(parts):
            xform = UsdGeom.Xform.Define(stage, f"{root}/edge_{e}")
            translate, orient = xform.AddTranslateOp(), xform.AddOrientOp()
            for k, pose in enumerate(poses[:, e]):
                translate.Set(Gf.Vec3d(*pose[:3]), k)
                orient.Set(Gf.Quatf(float(pose[6]), *(float(v) for v in pose[3:6])), k)
            field = UsdVol.ParticleField3DGaussianSplat.Define(stage, f"{root}/edge_{e}/splats")
            _set_splats(field, position, quat, scale)
            _set_appearance(field, sh_e, opacity_e)
    stage.GetRootLayer().Save()
    return stage
