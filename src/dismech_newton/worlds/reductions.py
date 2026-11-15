"""Per-world tile reductions and masked environment-reset copy kernel."""

from typing import Any

import warp as wp


@wp.kernel(module="unique", enable_backward=False)  # generic: an overload recompiles this, not the module
def masked_copy_kernel(start: int, world: wp.array[wp.int32], mask: wp.array[wp.bool], src: wp.array(dtype=Any),
                       dst: wp.array(dtype=Any)):
    """``dst[i] = src[i]`` for the rows ``i`` (from ``start``) whose world ``mask`` selects."""
    i = start + wp.tid()
    if mask[world[i]]:
        dst[i] = src[i]


@wp.func
def world_sum(a: wp.array[wp.float64], w: int, k: int, stride: int, v: wp.float64):
    """``a[w stride + k] += v`` (``w < 0``: a padding thread, nothing): summed over the block first when its threads
    share ``w`` (all of a single world's), else per thread; every thread of the block calls it with the same ``k``."""
    lo = wp.tile_extract(wp.tile_min(wp.tile(wp.where(w >= 0, w, 1 << 30))), 0)
    hi = wp.tile_extract(wp.tile_max(wp.tile(w)), 0)
    if lo == hi:
        wp.tile_atomic_add(a, wp.tile_sum(wp.tile(v)), offset=lo * stride + k)
    elif w >= 0:
        wp.atomic_add(a, w * stride + k, v)
