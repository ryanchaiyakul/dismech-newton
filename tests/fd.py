"""Batched, Richardson-extrapolated central finite differences; ``f`` maps ``(N, n)`` to ``(N, m)``."""

import numpy as np


def _richardson(central, h):
    return (4.0 * central(0.5 * h) - central(h)) / 3.0


def derivative(f, h: float) -> float:
    """``f'(0)``."""
    return _richardson(lambda e: (f(e) - f(-e)) / (2.0 * e), h)


def jacobian(f, x: np.ndarray, h) -> np.ndarray:
    """Shape ``(m, n)``."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size

    def central(h):
        D = np.diag(np.broadcast_to(h, n))
        y = f(np.concatenate([x + D, x - D]))
        return ((y[:n] - y[n:]) / (2.0 * np.diag(D))[:, None]).T

    return _richardson(central, np.asarray(h, dtype=np.float64))


def hessian(f, x: np.ndarray, h) -> np.ndarray:
    """Shape ``(m, n, n)``."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size

    def central(h):
        D = np.diag(np.broadcast_to(h, n))
        i, j = np.triu_indices(n)
        pts = [x + si * D[i] + sj * D[j] for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
        pp, pm, mp, mm = np.split(f(np.concatenate(pts)), 4)
        d = np.diag(D)
        upper = (pp - pm - mp + mm) / (4.0 * d[i] * d[j])[:, None]
        H = np.zeros((upper.shape[1], n, n))
        H[:, i, j] = upper.T
        H[:, j, i] = upper.T
        return H

    return _richardson(central, np.asarray(h, dtype=np.float64))


def assert_close(actual: np.ndarray, expected: np.ndarray, rtol: float, name: str = "", scale: float | None = None):
    """``max |actual - expected| <= rtol * scale``, ``scale`` defaulting to ``max |expected|``."""
    if scale is None:
        scale = np.abs(expected).max()
    err = np.abs(actual - expected).max()
    assert err <= rtol * scale, f"{name}: max error {err:.3e} > {rtol:g} * scale {scale:.3e}"
