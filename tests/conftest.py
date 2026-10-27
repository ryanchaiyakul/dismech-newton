"""Shared fixtures; ``--device`` picks the tests' Warp device (default ``cuda`` if available, else ``cpu``),
``--ptx-arch`` compiles CUDA kernels as PTX for an older GPU (``75``: Colab's T4, whose NVRTC bugs the local card
never shows; run it before merging kernel changes), ``--seeds N`` runs every seeded test over ``N`` seeds (the
first is the fast loop's): what the tolerances are checked against. ``--cubin-arch``/``--cubin`` build and load the
pushed commits' T4 kernel cache (``scripts/warp_cache.py``)."""

import re
import zlib

import numpy as np
import pytest
import warp as wp

from dismech_newton.linear import cudss


def pytest_addoption(parser):
    parser.addoption("--device", default=None, help="Warp device of the tests (default: cuda if available, else cpu)")
    parser.addoption("--ptx-arch", type=int, default=None,
                     help="compile CUDA kernels to PTX for this arch (e.g. 75: Colab's T4) instead of the device's")
    parser.addoption("--seeds", type=int, default=1, help="run every test taking `rng` over this many seeds")
    parser.addoption("--cubin-arch", type=int, default=None,
                     help="also compile every CUDA module to CUBIN for this arch (the kernel cache's build)")
    parser.addoption("--cubin", action="store_true",
                     help="load CUDA modules as CUBIN (a restored kernel cache; the T4 would JIT PTX at every load)")


def pytest_configure(config):
    """Before anything compiles: the PTX target holds for every module (the kernel cache keys on it)."""
    arch = config.getoption("--ptx-arch")
    if arch:
        wp.config.cuda_output = "ptx"
        wp.config.ptx_target_arch = arch
    if config.getoption("--cubin"):
        wp.config.cuda_output = "cubin"
    if config.getoption("--cubin-arch"):
        _also_compile_cubin(config.getoption("--cubin-arch"))


def _also_compile_cubin(arch):
    """Every CUDA module version Warp compiles is also compiled to CUBIN for ``arch``, as it compiles: modules are
    re-hashed as kernels get added, so the session's last versions are not all of them. (Warp's private
    ``Module._compile``; ``wp.compile_aot_module`` only sees the last version.)"""
    from warp._src.context import Module

    original = Module._compile

    def _compile(self, device=None, output_dir=None, output_name=None, output_arch=None, use_ptx=None, options=None):
        out = original(self, device, output_dir, output_name, output_arch, use_ptx, options)
        if device is not None and device.is_cuda:
            original(self, None, output_dir, None, arch, False, options)
        return out

    Module._compile = _compile


@pytest.fixture(scope="session")
def device(pytestconfig):
    name = pytestconfig.getoption("--device") or ("cuda" if wp.is_cuda_available() else "cpu")
    if name.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return wp.get_device(name)


@pytest.fixture(autouse=True)
def _on_device(device):
    """Models, solvers and arrays on ``--device``."""
    with wp.ScopedDevice(device):
        yield


def pytest_generate_tests(metafunc):
    n = metafunc.config.getoption("--seeds")
    if n > 1 and "rng" in metafunc.fixturenames:
        metafunc.parametrize("seed", range(n), ids=[f"seed{k}" for k in range(n)])


@pytest.fixture
def seed():
    return 0


@pytest.fixture
def rng(request, seed):
    """Seeded from the test id (less ``--seeds``' part) and ``seed``; seed 0 from the id alone."""
    nodeid = re.sub(r"\[seed\d+\]|-seed\d+|seed\d+-", "", request.node.nodeid)
    base = zlib.crc32(nodeid.encode())
    return np.random.default_rng((base, seed) if seed else base)


# -- requirements: skip unless ``--device`` is CUDA (with the library) ----------------------


@pytest.fixture
def needs_cuda(device):
    if not device.is_cuda:
        pytest.skip("needs CUDA")


@pytest.fixture
def needs_cudss(needs_cuda):
    if cudss is None:
        pytest.skip("needs cuDSS (nvmath)")
