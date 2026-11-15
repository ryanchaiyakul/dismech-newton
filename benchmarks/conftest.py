"""Shared fixtures and CUDA stream-synchronized runners for ``benchmarks/``."""

import sys
from pathlib import Path

import pytest
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))


def pytest_addoption(parser):
    parser.addoption("--device", default=None, help="Warp device for benchmarks (default: cuda if available)")


@pytest.fixture(scope="session")
def device(pytestconfig):
    name = pytestconfig.getoption("--device") or ("cuda" if wp.is_cuda_available() else "cpu")
    if name.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return wp.get_device(name)


@pytest.fixture(autouse=True)
def _on_device(device):
    with wp.ScopedDevice(device):
        yield


@pytest.fixture
def needs_cuda(device):
    if not device.is_cuda:
        pytest.skip("benchmarks require CUDA")


def count_dismech_modules() -> int:
    """Count Warp modules belonging to ``dismech_newton`` with compiled device executables."""
    count = 0
    for mod in wp._src.context.user_modules.values():
        if not mod.execs:
            continue
        for kernel in mod.kernels.values():
            func = getattr(kernel, "func", None)
            code = getattr(func, "__code__", None)
            if code is not None and "dismech_newton" in code.co_filename:
                count += 1
                break
    return count


def bench_framegraph(benchmark, fg, *, substeps: int = 4, batch: int = 10, warmup: int = 10, rounds: int = 7) -> None:
    """Warm up ``fg``, assert CUDA graph capture succeeded, and benchmark stream-synchronized replay batches."""
    for _ in range(warmup):
        fg()
    assert fg.graph is not None, "expected CUDA FrameGraph capture"
    wp.synchronize()

    def _run():
        for _ in range(batch):
            fg()
        wp.synchronize()

    benchmark.pedantic(_run, rounds=rounds, iterations=1, warmup_rounds=1)
    if benchmark.stats:
        benchmark.extra_info["ms_per_step"] = (benchmark.stats["median"] * 1.0e3) / (batch * substeps)
