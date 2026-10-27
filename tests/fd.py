"""Richardson-extrapolated central finite differences (``jacobian``: batched, ``f`` maps ``(N, n)`` to
``(N, m)``) and the tests' tolerance checks."""

import numpy as np


def _richardson(central, h):
    return (4.0 * central(0.5 * h) - central(h)) / 3.0


def derivative(f, h: float) -> float:
    """``f'(0)``."""
    return _richardson(lambda e: (f(e) - f(-e)) / (2.0 * e), h)


def jacobian(f, x: np.ndarray, h, error: bool = False):
    """Shape ``(m, n)``; with ``error``, also its error estimate ``|D(h) - D(h/2)|``, the two central differences
    apart: their truncation error and round-off, which bounds the extrapolation's."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size

    def central(h):
        D = np.diag(np.broadcast_to(h, n))
        y = f(np.concatenate([x + D, x - D]))
        return ((y[:n] - y[n:]) / (2.0 * np.diag(D))[:, None]).T

    h = np.asarray(h, dtype=np.float64)
    coarse, fine = central(h), central(0.5 * h)
    J = (4.0 * fine - coarse) / 3.0
    return (J, np.abs(coarse - fine)) if error else J


def assert_close(actual: np.ndarray, expected: np.ndarray, rtol: float, name: str = "", scale: float | None = None):
    """``max |actual - expected| <= rtol * scale``, ``scale`` defaulting to ``max |expected|``."""
    if scale is None:
        scale = np.abs(expected).max()
    err = np.abs(actual - expected).max()
    assert err <= rtol * scale, f"{name}: max error {err:.3e} > {rtol:g} * scale {scale:.3e}"


def assert_derivative(actual: float, fd: float, est: float, rtol: float, name: str = "", scale: float | None = None,
                      noise: float = 4.0, resolved: float | None = None):
    """``|actual - fd| <= rtol * scale + noise * est``, ``scale`` defaulting to ``|fd|``: a direction the differences
    resolve poorly (``est``, from :func:`jacobian`; ``noise`` standard deviations) says so instead of needing a looser
    ``rtol``. ``resolved``: also ``est <= resolved |fd|``."""
    err = abs(actual - fd)
    scale = abs(fd) if scale is None else scale
    assert err <= rtol * scale + noise * est, (
        f"{name}: adjoint {actual:.6e}, differences {fd:.6e} (error estimate {est:.1e}): error {err:.2e} > "
        f"{rtol:g} * scale {scale:.2e} + {noise:g} * {est:.1e}")
    if resolved is not None:
        assert est <= resolved * abs(fd), f"{name}: differences {fd:.6e} unresolved: error estimate {est:.1e}"
