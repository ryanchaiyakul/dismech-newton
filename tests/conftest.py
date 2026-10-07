"""Shared fixtures; ``--device`` picks the tests' Warp device (default ``cpu``)."""

import zlib

import numpy as np
import pytest
import warp as wp

from dismech_newton.linear import cudss, cusparse


def pytest_addoption(parser):
    parser.addoption("--device", default="cpu", help="Warp device of the tests (default: cpu)")


@pytest.fixture(scope="session")
def device(pytestconfig):
    name = pytestconfig.getoption("--device")
    if name.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return wp.get_device(name)


@pytest.fixture(autouse=True)
def _on_device(device):
    """Models, solvers and arrays on ``--device``."""
    with wp.ScopedDevice(device):
        yield


@pytest.fixture
def rng(request):
    """Seeded from the test id."""
    return np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))


# -- requirements: skip unless ``--device`` is CUDA (with the library) ----------------------


@pytest.fixture
def needs_cuda(device):
    if not device.is_cuda:
        pytest.skip("needs CUDA")


@pytest.fixture
def needs_cudss(needs_cuda):
    if cudss is None:
        pytest.skip("needs cuDSS (nvmath)")


@pytest.fixture
def needs_cusparse(needs_cuda):
    if cusparse is None:
        pytest.skip("needs cuSPARSE (nvmath)")
