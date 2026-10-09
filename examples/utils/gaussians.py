"""Gaussian splats rendered in torch, for the examples that fit to images (``--extra splat``, CUDA).

- :class:`Camera`: a pinhole camera on gswarp (the Inria 3DGS rasterizer): splats ``(means, cov6)`` with colours
  and opacities -> an image.
- :func:`view`, :func:`backprop`: the Warp -> torch -> Warp handoff of an image loss's gradient.
- :func:`cov6`, :func:`rotation_matrix`: free Gaussians' covariances from log scales and quaternions.

Importing it puts Warp and torch on one stream, :data:`STREAM`, created by Warp so CUDA graphs can be captured on it
(not on torch's legacy default stream); import it before building any solver. gswarp binds Warp to its own wrapper of
torch's stream, on which Warp cannot capture: :class:`Camera` and :func:`backprop` bind :data:`STREAM` back.
"""

import math

import numpy as np
import torch
import warp as wp
from gswarp import GaussianRasterizationSettings, GaussianRasterizer

if not torch.cuda.is_available():
    raise SystemExit("Gaussian splat rendering (gswarp) needs CUDA")
DEVICE = "cuda:0"
STREAM = wp.Stream(DEVICE)
torch.cuda.set_stream(torch.cuda.ExternalStream(STREAM.cuda_stream, device=DEVICE))
wp.set_stream(STREAM, device=DEVICE, sync=True)


def _rebind() -> None:
    """Warp back on :data:`STREAM` after gswarp (the same CUDA stream: no sync)."""
    wp.set_stream(STREAM, device=DEVICE, sync=False)


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World -> camera 4x4, COLMAP axes: +z forward, +x right, +y down."""
    eye, target, up = (np.asarray(a, dtype=float) for a in (eye, target, up))
    f = (target - eye) / np.linalg.norm(target - eye)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    V = np.eye(4)
    V[:3, :3] = np.stack([r, np.cross(f, r), f])
    V[:3, 3] = -V[:3, :3] @ eye
    return V


class Camera:
    """A square pinhole camera at ``eye`` looking at ``target``: ``self(means, cov6, colours, opacity)`` is the
    image ``(3, size, size)``, differentiable in all four; :meth:`project` gives pixel coordinates.

    ``background``: a colour, or an image ``(3, size, size)`` (torch) the splats are composited over."""

    def __init__(self, eye, target, size: int = 256, fov: float = math.radians(45.0),
                 background=(0.05, 0.05, 0.08), near: float = 0.01, far: float = 100.0):
        self.eye, self.size, self.fov = np.asarray(eye, dtype=float), size, fov
        self.plate = background if isinstance(background, torch.Tensor) else None
        if self.plate is not None:
            background = (0.0, 0.0, 0.0)
        self.V = look_at(eye, target)
        tan = math.tan(fov / 2)
        P = torch.zeros(4, 4)  # Inria's projection
        P[0, 0], P[1, 1] = 1 / tan, 1 / tan
        P[3, 2], P[2, 2], P[2, 3] = 1.0, far / (far - near), -far * near / (far - near)
        V = torch.tensor(self.V, dtype=torch.float32)
        settings = GaussianRasterizationSettings(
            image_height=size, image_width=size, tanfovx=tan, tanfovy=tan,
            bg=torch.tensor(background, dtype=torch.float32, device=DEVICE), scale_modifier=1.0,
            viewmatrix=V.T.contiguous().to(DEVICE), projmatrix=(V.T @ P.T).contiguous().to(DEVICE), sh_degree=0,
            campos=torch.tensor(self.eye, dtype=torch.float32, device=DEVICE), prefiltered=False, auto_tune=False,
            auto_tune_verbose=False)
        self.raster = GaussianRasterizer(settings)

    def __call__(self, means, cov6, colours, opacity) -> torch.Tensor:
        return self.layers(means, cov6, colours, opacity)[0]

    def layers(self, means, cov6, colours, opacity) -> tuple[torch.Tensor, torch.Tensor]:
        """The image and the splats' opacity ``(1, size, size)``, both differentiable."""
        image, _, meta = self.raster(means3D=means, means2D=torch.zeros_like(means), cov3D_precomp=cov6,
                                     colors_precomp=colours, opacities=opacity.reshape(-1, 1))
        _rebind()
        if self.plate is not None:
            image = image + (1.0 - meta.alpha) * self.plate
        return image, meta.alpha

    def project(self, x) -> np.ndarray:
        """Pixel coordinates ``(..., 2)`` of world points ``(..., 3)``."""
        c = np.asarray(x, dtype=float) @ self.V[:3, :3].T + self.V[:3, 3]
        f = 0.5 * self.size / math.tan(self.fov / 2)
        return np.stack([f * c[..., 0] / c[..., 2], f * c[..., 1] / c[..., 2]], -1) + 0.5 * self.size

    def depth(self, x) -> np.ndarray:
        return (np.asarray(x, dtype=float) @ self.V[:3, :3].T + self.V[:3, 3])[..., 2]


def to_uint8(image: torch.Tensor) -> np.ndarray:
    """``(3, H, W)`` -> ``(H, W, 3)`` uint8."""
    return (image.detach().clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def view(a: wp.array) -> torch.Tensor:
    """A torch view of a Warp array (zero copy)."""
    return wp.to_torch(a, requires_grad=False)


def backprop(arrays, loss) -> float:
    """``loss(*tensors)`` on torch leaves viewing the Warp ``arrays``, and its backward; the leaves' gradients are
    added into the arrays' ``.grad``, where ``tape.backward`` picks them up."""
    leaves = [view(a).detach().requires_grad_() for a in arrays]
    value = loss(*leaves)
    value.backward()
    _rebind()
    for a, t in zip(arrays, leaves):
        if t.grad is not None:
            view(a.grad).add_(t.grad)
    return float(value.detach())


def _rotation(w, x, y, z, stack):
    return stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """``(N, 3, 3)`` from quaternions ``(N, 4)``, w first (normalised here)."""
    q = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    return _rotation(*q.T, np.stack)


def cov6(log_scale: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """``(xx, xy, xz, yy, yz, zz)`` of Gaussians with log standard deviations ``(N, 3)`` and quaternions
    ``(N, 4)``, w first."""
    q = quat / quat.norm(dim=1, keepdim=True)
    R = _rotation(*q.unbind(1), torch.stack)
    C = (R * torch.exp(2.0 * log_scale)[:, None, :]) @ R.transpose(1, 2)
    i, j = np.triu_indices(3)
    return C[:, i, j]
