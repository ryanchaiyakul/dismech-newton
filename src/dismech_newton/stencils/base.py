"""The stencil interface: a group of DOFs that assembles its own elastic contribution.

Builder side, :meth:`~Stencil.register` declares the class's custom frequency and attributes.
Solver side, per step: :meth:`~Stencil.begin_step`, :meth:`~Stencil.assemble` every Newton
iteration, then :meth:`~Stencil.end_step`. A subclass that sets ``energy`` is concrete: it is
named after its class (``LinearTriplet`` -> ``"linear_triplet"``) unless it sets ``name``, and
is registered in :data:`STENCILS`, where the solver finds the ones a model uses.
"""

import re
from typing import ClassVar

import numpy as np
import warp as wp
from newton import Model, ModelBuilder, State
from warp import Function

from ..system import SymmetricCSR

NAMESPACE = "dismech"

# Concrete stencil classes by name (= custom frequency), in definition order.
STENCILS: dict[str, type["Stencil"]] = {}


class Stencil:
    name: ClassVar[str]
    num_dofs: ClassVar[int]  # DOFs per stencil
    energy: ClassVar[Function | None] = None  # signature fixed by the stencil family
    uses_rate: ClassVar[bool] = False  # whether ``energy`` reads the start-of-step strains

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.energy is not None:
            cls.name = cls.__dict__.get("name") or re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
            STENCILS[cls.name] = cls

    @classmethod
    def frequency(cls) -> str:
        return f"{NAMESPACE}:{cls.name}"

    @classmethod
    def attr(cls, field: str) -> str:
        return f"{cls.name}_{field}"

    @classmethod
    def rows(cls, model: Model) -> int:
        """Stencils of this class in ``model`` (0 if it was never registered)."""
        return model.custom_frequency_counts.get(cls.frequency(), 0)

    @classmethod
    def register(cls, builder: ModelBuilder) -> None:
        """Register this class's custom frequency and attributes on ``builder`` (idempotent)."""
        raise NotImplementedError

    def bind(self, model: Model, fixed: wp.array) -> None:
        """Read model attributes and allocate buffers; never write to rows/columns where ``fixed``."""
        raise NotImplementedError

    def dofs(self) -> np.ndarray:
        """``(count, num_dofs)`` global DOFs of every stencil, for the Hessian pattern."""
        raise NotImplementedError

    def assemble(
        self, state_in: State, state_out: State, residual: wp.array, hessian: SymmetricCSR, dt: float
    ) -> None:
        """Add the gradient and upper-triangle Hessian at ``state_out`` into ``residual`` and ``hessian``."""
        raise NotImplementedError

    def begin_step(self, state_in: State) -> None:
        pass

    def end_step(self, state_in: State, state_out: State) -> None:
        """Advance per-stencil state in ``state_out`` (edge frames are already advanced)."""
