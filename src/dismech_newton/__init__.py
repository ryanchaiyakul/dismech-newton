"""Discrete elastic rods and midedge shells for Newton: one implicit solver, strain-space ADMM elements with barrier
contact, friction and an adjoint (:class:`SolverDiSMech`)."""

from .builder import add_cloth_mesh, add_rod, fix_segment, register_custom_attributes
from .solver import SolverDiSMech
from .state import flatten_state, suspended_tape

__all__ = [
    "SolverDiSMech",
    "add_cloth_mesh",
    "add_rod",
    "fix_segment",
    "flatten_state",
    "register_custom_attributes",
    "suspended_tape",
]

import os as _os  # EXPERIMENT (proxy-impact): DM_MODE / DM_ITER override every SolverCoupledProxy's mode / iterations

if _os.environ.get("DM_MODE") or _os.environ.get("DM_ITER") or _os.environ.get("DM_RELAX"):
    from newton.solvers.experimental.coupled import SolverCoupledProxy as _P

    def _wrap(cls, field, env, cast):
        init = cls.__init__

        def __init__(self, *a, **kw):
            init(self, *a, **kw)
            object.__setattr__(self, field, cast(_os.environ[env]))

        if _os.environ.get(env):
            cls.__init__ = __init__

    _wrap(_P.Proxy, "mode", "DM_MODE", str)
    _wrap(_P.Proxy, "proxy_relaxation_mode", "DM_RELAX", str)
    _wrap(_P.Config, "iterations", "DM_ITER", int)
