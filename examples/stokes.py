"""Regularized Stokeslet segments (Cortez 2018): the viscous drag of the IMC flagella paper.

A port of ``RegularizedStokeslet.cpp`` from https://github.com/StructuresComp/rod-contact-sim
(Tong, Choi et al., arXiv:2205.10309, Appendix A). The fluid velocity at every node is a linear
function of the force densities at every node of every rod, ``U = A f``, with ``A`` dense. At the
start of each step ``A`` is built at the nodes' positions, ``8 pi eta A f = U`` is solved for the
nodes' velocities (no slip), and ``F = -8 pi eta A^-1 U`` is applied explicitly for the step.

As in the reference, ``A`` is factored by Cholesky from its lower triangle (Eigen's ``llt``), in
double precision; here on the GPU with cuSOLVER.
"""

import cupy as cp
import cupyx.lapack
import numpy as np
import warp as wp


@wp.func
def _segment_blocks(x: wp.vec3d, y0: wp.vec3d, y1: wp.vec3d, eps: wp.float64):
    """``(M1, M2)``: velocity at ``x`` per unit force density at ``y0`` and ``y1`` (linear along the segment)."""
    x0 = x - y0
    x1 = x - y1
    v = y0 - y1
    l = wp.length(v)
    e2 = eps * eps
    r0 = wp.sqrt(wp.dot(x0, x0) + e2)
    r1 = wp.sqrt(wp.dot(x1, x1) + e2)
    one = wp.float64(1.0)
    ll = l * l
    x0v = wp.dot(x0, v)
    x1v = wp.dot(x1, v)
    t0_1 = (wp.log(l * r1 + x1v) - wp.log(l * r0 + x0v)) / l
    t0_3 = -(one / (r1 * (l * r1 + x1v)) - one / (r0 * (l * r0 + x0v)))
    t1_1 = (r1 - r0) / ll - t0_1 * x0v / ll
    t1_3 = -(one / (r1 * ll) - one / (r0 * ll)) - t0_3 * x0v / ll
    t2_3 = -(one / (r1 * ll)) + t0_1 / ll - t1_3 * x0v / ll
    t3_3 = -(one / (r1 * ll)) + wp.float64(2.0) * t1_1 / ll - t2_3 * x0v / ll
    xx = wp.outer(x0, x0)
    xv = wp.outer(x0, v) + wp.outer(v, x0)
    vv = wp.outer(v, v)
    eye = wp.identity(3, dtype=wp.float64)
    m2 = (t1_1 + e2 * t1_3) * eye + t1_3 * xx + t2_3 * xv + t3_3 * vv
    m1 = (t0_1 + e2 * t0_3) * eye + t0_3 * xx + t1_3 * xv + t2_3 * vv - m2
    return m1, m2


@wp.kernel
def _assemble_kernel(
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    nv: int,
    eps: wp.float64,
    # outputs
    A: wp.array2d[wp.float64],
    U: wp.array[wp.float64],
):
    """Lower triangle of ``A``, block ``(i, k)`` for evaluation node ``i`` and force node ``k``:
    ``M1`` of the segment starting at ``k`` plus ``M2`` of the segment ending at ``k``."""
    i, k = wp.tid()
    if k > i:
        return
    x = wp.vec3d(pos[i])
    kl = k % nv
    block = wp.mat33d()
    if kl < nv - 1:
        m1, m2 = _segment_blocks(x, wp.vec3d(pos[k]), wp.vec3d(pos[k + 1]), eps)
        block += m1
    if kl > 0:
        m1, m2 = _segment_blocks(x, wp.vec3d(pos[k - 1]), wp.vec3d(pos[k]), eps)
        block += m2
    for a in range(3):
        for b in range(3):
            r = 3 * i + a
            c = 3 * k + b
            if c <= r:
                A[r, c] = block[a, b]
                A[c, r] = block[a, b]  # mirror: Cholesky of the lower triangle
    if k == 0:
        v = vel[i]
        for a in range(3):
            U[3 * i + a] = wp.float64(v[a])


@wp.kernel
def _force_kernel(f: wp.array[wp.float64], scale: wp.float64, force: wp.array[wp.vec3]):
    i = wp.tid()
    force[i] = wp.vec3(wp.vec3d(f[3 * i], f[3 * i + 1], f[3 * i + 2]) * scale)


class Stokeslets:
    """Drag on ``rods * nv`` nodes (rods of ``nv`` nodes, stored rod after rod)."""

    def __init__(self, rods: int, nv: int, viscosity: float, epsilon: float, device):
        self.n, self.nv, self.device = rods * nv, nv, device
        self.viscosity, self.epsilon = viscosity, epsilon
        self.A = wp.zeros((3 * self.n, 3 * self.n), dtype=wp.float64, device=device)
        self.U = wp.zeros(3 * self.n, dtype=wp.float64, device=device)
        self.force = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self._stream = cp.cuda.ExternalStream(wp.get_stream(device).cuda_stream)

    def compute(self, pos: wp.array, vel: wp.array) -> wp.array:
        """Nodal drag ``F = -8 pi eta A(pos)^-1 vel`` into ``self.force``."""
        wp.launch(_assemble_kernel, dim=(self.n, self.n), inputs=[pos, vel, self.nv, wp.float64(self.epsilon)],
                  outputs=[self.A, self.U], device=self.device)
        with self._stream:
            f = cupyx.lapack.posv(cp.asarray(self.A), cp.asarray(self.U))
            f = cp.ascontiguousarray(f.reshape(-1))
            wp.launch(_force_kernel, dim=self.n, inputs=[wp.from_dlpack(f), wp.float64(-8.0 * np.pi * self.viscosity)],
                      outputs=[self.force], device=self.device)
        return self.force
