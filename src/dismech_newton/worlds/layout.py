"""Multi-world DOF, element, and BVH grid-cell layout descriptors (zero NVRTC kernels)."""

import numpy as np
import warp as wp
from newton import Model


@wp.struct
class Worlds:
    """Each world's step is its own solve (one line search, convergence and failure per world): ``of`` per DOF its
    world; ``active``: the worlds iterating this iteration; per-world sums are ``a[w * stride + k]``."""

    of: wp.array[wp.int32]
    active: wp.array[wp.int32]
    count: int
    stride: int


def model_world_info(model: Model) -> tuple[np.ndarray, bool, int]:
    """Return ``(particle_world, has_global_particles, num_param_worlds)`` for a Newton ``model``."""
    pw = model.particle_world.numpy() if model.particle_count else np.zeros(0, np.int32)
    has_global = bool(model.world_count > 1 and (pw < 0).any())
    nw = 1 if (has_global or not pw.size) else int(np.maximum(pw, 0).max(initial=0)) + 1
    return pw, has_global, nw


def dof_worlds(model: Model, n: int, stencils) -> np.ndarray:
    """Per DOF of ``n`` its world: a node's (global nodes: world 0, and one world if there are any), an element's
    DOFs their first node's (``stencils``: rows of DOFs, each within a world, the first a node's; -1s unused)."""
    nodes = np.maximum(model.particle_world.numpy(), 0) if model.particle_count else np.zeros(0, np.int64)
    if (model.particle_world.numpy() < 0).any() and model.world_count > 1:
        return np.zeros(n, np.int32)
    out = np.zeros(n, np.int64)
    out[: 3 * len(nodes)] = np.repeat(nodes, 3)
    for rows in stencils:
        rows = np.asarray(rows)
        if rows.size:
            w = np.broadcast_to(out[rows[:, :1]], rows.shape)
            out[rows[rows >= 0]] = w[rows >= 0]
    return out.astype(np.int32)


def elem_worlds(model: Model, elem) -> wp.array:
    """Per-element parameter world index ``wp.array[wp.int32]`` (``0`` when global particles exist)."""
    pw, has_global, _ = model_world_info(model)
    n = int(elem.count)
    if has_global or not n:
        ew = np.zeros(n, dtype=np.int32)
    else:
        nodes = elem.dofs()[:, 0] // 3
        ew = np.maximum(pw[nodes], 0).astype(np.int32)
    return wp.array(ew, dtype=wp.int32, device=elem.params.device)


elem_world_indices = elem_worlds


def world_shift(model: Model) -> np.ndarray:
    """Per node a shift of its world's boxes in the trees: the worlds on a grid of cells 4 times the largest world's
    extent (plus 1 m) apart, so a query meets only its own world's (fewer, if a world moves that far: then the world
    filter still holds). None with one world, or with global nodes (they meet every world)."""
    n, W = model.particle_count, max(model.world_count, 1)
    shift = np.zeros((max(n, 1), 3))
    world = model.particle_world.numpy() if n else np.zeros(0, np.int32)
    if W == 1 or not n or (world < 0).any():
        return shift
    x = model.particle_q.numpy()
    extent = max(np.ptp(x[world == w], axis=0).max() for w in np.unique(world))
    side = int(np.ceil(W ** (1.0 / 3.0) - 1.0e-9))
    cell = np.stack([world % side, (world // side) % side, world // (side * side)], axis=1)
    shift[:n] = cell * (4.0 * extent + 1.0)
    return shift
