"""Shared fixtures; ``--device`` picks the tests' Warp device (default ``cpu``)."""

import functools
import zlib

import numpy as np
import pytest
import warp as wp
from triplets import TripletConfig, eval_local, eval_triplet


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
    """Models and solvers on ``--device``."""
    with wp.ScopedDevice(device):
        yield


@pytest.fixture
def rng(request):
    """Seeded from the test id."""
    return np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))


# -- triplets -----------------------------------------------------------------------------


def _random_triplet(seed: int) -> TripletConfig:
    rng = np.random.default_rng(seed)
    x0 = 0.1 * rng.normal(size=3)
    x1 = x0 + 0.1 * (np.array([1.0, 0.0, 0.0]) + 0.4 * rng.normal(size=3))
    x2 = x1 + 0.1 * (np.array([1.0, 0.0, 0.0]) + 0.4 * rng.normal(size=3))
    theta_e, theta_f = rng.uniform(-np.pi, np.pi, size=2)
    l0e, l0f = 0.1 * rng.uniform(0.8, 1.2, size=2)
    return TripletConfig.current(x0, x1, x2, theta_e, theta_f, ref_twist=rng.normal(), l0e=l0e, l0f=l0f, seed=seed)


def _bend(angle: float, l: float = 0.1):
    """Two edges of length ``l`` bent by ``angle`` in the xy-plane."""
    return (0.0, 0.0, 0.0), (l, 0.0, 0.0), (l + l * np.cos(angle), l * np.sin(angle), 0.0)


TRIPLETS = {
    "straight": lambda: TripletConfig.current(*_bend(0.0)),
    "bent": lambda: TripletConfig.current(*_bend(np.radians(30.0))),
    "sharp_bend": lambda: TripletConfig.current(*_bend(np.radians(120.0))),
    "twisted": lambda: TripletConfig.current(*_bend(np.radians(30.0)), theta_e=0.4, theta_f=-0.9, ref_twist=0.5),
    "stretched": lambda: TripletConfig.current(*_bend(np.radians(45.0)), l0e=0.08, l0f=0.13),
    **{f"random-{i}": functools.partial(_random_triplet, i) for i in range(3)},
}


@pytest.fixture(params=list(TRIPLETS))
def triplet(request) -> TripletConfig:
    return TRIPLETS[request.param]()


@pytest.fixture(scope="session")
def strains(device):
    """``strains(Q, cfg, sigma=...)`` on ``device``."""
    return functools.partial(eval_triplet, device=device)


@pytest.fixture(scope="session")
def local_strains(device):
    """``local_strains(Z, cfg, sigma=...)`` on ``device``: strains and derivatives in ADMM's ``z``."""
    return functools.partial(eval_local, device=device)
