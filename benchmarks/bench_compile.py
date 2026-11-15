"""NVRTC compiled module count, cold core compile time, and warm solver instantiation benchmarks."""

import os
import subprocess
import sys

import newton
import numpy as np
import pytest
import warp as wp
from conftest import count_dismech_modules

from dismech_newton import SolverDiSMech, flatten_state
from dismech_newton.coupling import SolverCoupledProxy
from dismech_newton.residual import ResidualSpec, SurfaceMaterialSpec


def _build_rod(n_seg: int = 4) -> newton.ModelBuilder:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    b.add_ground_plane()
    rod = newton.Rod.create_straight((0.0, 0.0, 0.02), (1.0, 0.0, 0.0), 0.2, segment_count=n_seg, radius=0.01)
    SolverDiSMech.add_rod(b, rod=rod)
    return b


def _build_shell(nx: int = 4, ny: int = 4) -> newton.ModelBuilder:
    from utils.common import grid  # noqa: PLC0415

    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    b.add_ground_plane()
    v, f = grid(nx, ny, 0.2, 0.2)
    v[:, 2] = 0.02
    SolverDiSMech.add_cloth_mesh(b, vertices=v.tolist(), indices=f.ravel().tolist(), density=1.0,
                                 tri_ke=1.0e3, tri_ka=1.0e3, bend_stiffness=0.05)
    return b


def _build_coupled(device):
    from newton.solvers import SolverSemiImplicit  # noqa: PLC0415

    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverDiSMech.register_custom_attributes(b)
    link = b.add_link(mass=0.25, inertia=wp.mat33(np.diag([0.02, 0.03, 0.04])))
    rigid_sh = b.add_shape_sphere(link, radius=0.02)
    j0 = b.add_joint_revolute(parent=-1, child=link, parent_xform=wp.transform((0.0, 0.0, 0.03), wp.quat_identity()),
                              child_xform=wp.transform((-0.08, 0.0, 0.0), wp.quat_identity()), axis=(0, 1, 0))
    b.add_articulation([j0])
    proxy = b.add_link(is_kinematic=True, mass=0.0, inertia=wp.mat33(np.zeros((3, 3))))
    apsh = b.add_shape_sphere(proxy, radius=0.02)
    p0, s0_i = b.particle_count, b.shape_count
    rod = newton.Rod.create_straight((-0.1, 0.0, 0.0), (1.0, 0.0, 0.0), 0.2, segment_count=4, radius=0.01)
    r_b, _ = SolverDiSMech.add_rod(b, rod=rod, collide=True, proxies=True)
    SolverDiSMech.fix_segment(b, body=r_b[0])
    m = b.finalize(device=device)
    entries = [
        SolverCoupledProxy.Entry(name="rigid", solver=lambda mv: SolverSemiImplicit(mv), bodies=[link], joints=[j0],
                                 shapes=[rigid_sh]),
        SolverCoupledProxy.Entry(name="soft", solver=SolverDiSMech, particles=list(range(p0, b.particle_count)),
                                 bodies=r_b, shapes=[apsh, *range(s0_i, b.shape_count)]),
    ]
    proxies = [SolverCoupledProxy.Proxy(source="rigid", destination="soft", bodies=[link], proxy_bodies=[proxy])]
    return m, entries, proxies


def test_compile_cold_core(benchmark, needs_cuda, tmp_path):
    """Cold NVRTC compile of core rod + shell + contact + solver in a fresh ``WARP_CACHE_PATH``."""
    code = """
import newton, warp as wp, dismech_newton as dn
from dismech_newton import SolverDiSMech, add_cloth_mesh, add_rod
b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
b.add_ground_plane()
r = newton.Rod.create_straight((0.0, 0.0, 0.02), (1.0, 0.0, 0.0), 0.2, segment_count=3, radius=0.01)
add_rod(b, rod=r)
v = [[0.0, 0.1, 0.02], [0.1, 0.1, 0.02], [0.0, 0.2, 0.02]]
add_cloth_mesh(b, vertices=v, indices=[0, 1, 2], density=1.0, tri_ke=1e3, tri_ka=1e3, bend_stiffness=0.05)
m = b.finalize()
s = SolverDiSMech(m)
s0, s1 = m.state(), m.state()
s.step(s0, s1, None, None, 0.01)
wp.synchronize()
mods = sum(
    1 for mod in wp._src.context.user_modules.values()
    if mod.execs and any(
        getattr(getattr(k, "func", None), "__code__", None) is not None
        and "dismech_newton" in k.func.__code__.co_filename
        for k in mod.kernels.values()
    )
)
print(f"COLD_MODULES={mods}")
"""
    env = {**os.environ, "WARP_CACHE_PATH": str(tmp_path)}
    captured = {"modules": 0}

    def _cold_run():
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        for line in out.stdout.splitlines():
            if line.startswith("COLD_MODULES="):
                captured["modules"] = int(line.split("=", 1)[1])

    benchmark.pedantic(_cold_run, rounds=1, iterations=1, warmup_rounds=0)
    benchmark.extra_info["compiled_modules"] = captured["modules"]


@pytest.mark.parametrize("kind", ["rod", "shell", "contact", "coupled", "residual"])
def test_compile_warm(benchmark, device, needs_cuda, kind: str):
    """Warm solver construction, sparse analysis, and first step timing plus active ``wp.Module`` count."""
    if kind == "coupled":
        m, entries, proxies = _build_coupled(device)
        s0, s1, ctrl = m.state(), m.state(), m.control()
        newton.eval_fk(m, s0.joint_q, s0.joint_qd, s0)

        def _instantiate_and_step():
            c = SolverCoupledProxy(m, entries=entries, coupling=SolverCoupledProxy.Config(proxies=proxies, iterations=1))
            c.step(s0, s1, ctrl, None, 2.0e-3)
            wp.synchronize()
    else:
        if kind == "shell":
            m = _build_shell().finalize(device=device)
            res = None
            sc = False
        elif kind == "rod":
            m = _build_rod().finalize(device=device)
            res = None
            sc = False
        elif kind == "contact":
            m = _build_rod().finalize(device=device)
            res = None
            sc = True
        else:
            b = _build_rod()
            from utils.common import grid  # noqa: PLC0415

            v, f = grid(3, 3, 0.2, 0.2)
            v[:, 2] = 0.04
            SolverDiSMech.add_cloth_mesh(b, vertices=v.tolist(), indices=f.ravel().tolist(), density=1.0,
                                         tri_ke=1.0e3, tri_ka=1.0e3, bend_stiffness=0.05)
            m = b.finalize(device=device)
            res = ResidualSpec(
                rod_rff_dim=16,
                shell_rff_dim=16,
                contact=SurfaceMaterialSpec(rff_dim=16, num_materials=2),
            )
            sc = True
        s0, s1 = m.state(), m.state()
        flatten_state(s0)
        flatten_state(s1)

        def _instantiate_and_step():
            s = SolverDiSMech(m, self_contact=sc, residual=res)
            s.step(s0, s1, None, None, 5.0e-3)
            wp.synchronize()

    benchmark.pedantic(_instantiate_and_step, rounds=5, iterations=1, warmup_rounds=1)
    benchmark.extra_info["compiled_modules"] = count_dismech_modules()
