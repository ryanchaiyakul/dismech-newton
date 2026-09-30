"""The rod triplet stencil family: two consecutive edges (DOFs ``[x0, theta_e, x1, theta_f, x2]``)."""

from typing import ClassVar

import numpy as np
import warp as wp
from newton import Model, ModelBuilder, State
from warp import Function

from ...system import SymmetricCSR
from ..base import NAMESPACE, Stencil
from .kernels import advance_ref_twist_kernel, make_assemble_kernel, triplet_strain_kernel
from .strains.helper import vec5f, vec5i


class TripletStencil(Stencil):
    """Stretch of both edges, two bend curvatures and twist, under an energy on those 5 strains.

    Abstract: a concrete subclass sets ``energy``, an inlined ``wp.func``
    ``(eps, eps_prev, rest, params, dt) -> (sigma, C)`` with ``sigma = dE/d eps`` and
    ``C = d^2E/d eps^2`` (``vec5f`` / ``mat55f``), where ``params`` is the triplet's parameter
    vector (its annotated type is what the model stores, see :meth:`params_type`) and
    ``eps_prev`` the start-of-step strains (``eps`` itself unless ``uses_rate``). A concrete
    subclass also sets :meth:`pack`. See :mod:`dismech_newton.stencils.triplet.linear`.
    """

    num_dofs = 11
    energy: ClassVar[Function | None] = None

    @classmethod
    def params_type(cls) -> type:
        """Per-triplet parameter vector type: the annotation of ``energy``'s ``params`` argument."""
        return list(cls.energy.input_types.values())[3]

    @classmethod
    def pack(cls, k: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Parameter rows ``(T, len(params_type()))`` from per-strain stiffness and damping ``(T, 5)``."""
        raise NotImplementedError

    # -- builder side -------------------------------------------------------------------

    @classmethod
    def register(cls, builder: ModelBuilder) -> None:
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name=cls.name, namespace=NAMESPACE))
        # Edge indices stay scalar int32 so that merging builders can offset them.
        attributes = [
            ("edge0", wp.int32, f"{NAMESPACE}:edge"),
            ("edge1", wp.int32, f"{NAMESPACE}:edge"),
            ("params", cls.params_type(), None),
        ]
        for field, dtype, references in attributes:
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name=cls.attr(field), dtype=dtype, frequency=cls.frequency(), namespace=NAMESPACE,
                    references=references,
                )
            )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name=cls.attr("ref_twist_q"),
                dtype=float,
                frequency=cls.frequency(),
                assignment=Model.AttributeAssignment.STATE,
                namespace=NAMESPACE,
            )
        )

    @classmethod
    def add_rows(cls, builder: ModelBuilder, edge0: int, triplets: np.ndarray, k: np.ndarray, c: np.ndarray) -> None:
        """Add one row per triplet; ``triplets`` holds local edge pairs, offset by ``edge0``.

        ``k`` and ``c`` are the per-strain stiffness and damping, ``(T, 5)``; :meth:`pack`
        turns them into parameter rows.
        """
        params = cls.params_type()
        packed = cls.pack(k, c)
        p = f"{NAMESPACE}:"
        for t in range(len(triplets)):
            builder.add_custom_values(
                **{
                    p + cls.attr("edge0"): edge0 + int(triplets[t, 0]),
                    p + cls.attr("edge1"): edge0 + int(triplets[t, 1]),
                    p + cls.attr("params"): params(*packed[t].tolist()),
                    # Zero: the first strain evaluation then yields the reference twist itself.
                    p + cls.attr("ref_twist_q"): 0.0,
                }
            )

    # -- solver side --------------------------------------------------------------------

    def bind(self, model: Model, fixed: wp.array) -> None:
        d = getattr(model, NAMESPACE)
        dev = model.device
        self.model = model
        self.der = d
        self.device = wp.get_device(dev)
        self.count = self.rows(model)
        self.num_node_dofs = 3 * model.particle_count
        self.dof_fixed = fixed
        self._params = getattr(d, self.attr("params"))
        self._ref_twist = self.attr("ref_twist_q")
        self._kernel = make_assemble_kernel(type(self))

        # One connectivity record per triplet, (e, f, n0, n1, n2): no dependent loads in the kernels.
        e, f = getattr(d, self.attr("edge0")).numpy(), getattr(d, self.attr("edge1")).numpy()
        node0, node1 = d.edge_node0.numpy(), d.edge_node1.numpy()
        conn = np.column_stack((e, f, node0[e], node1[e], node1[f])).astype(np.int32)
        self._conn = wp.array(conn, dtype=vec5i, device=dev)

        # Start-of-step strains exist only for energies that read them.
        self._strain_prev = wp.zeros(self.count if self.uses_rate else 0, dtype=vec5f, device=dev)

        # Rest curvatures and twist: the initial configuration is unstressed (stretch rest is exactly zero).
        rest = wp.zeros(self.count, dtype=vec5f, device=dev)
        self._measure(model.state(), rest)
        self._rest = wp.array(rest.numpy()[:, 2:], dtype=wp.vec3, device=dev)

    @property
    def rest(self) -> wp.array:
        """Rest ``[kappa1, kappa2, tau]`` per triplet."""
        return self._rest

    def dofs(self) -> np.ndarray:
        """``(T, 11)`` DOFs ``[x0, theta_e, x1, theta_f, x2]`` of every triplet (host mirror of the kernel)."""
        conn = self._conn.numpy().astype(np.int64)
        e, f, n0, n1, n2 = conn.T
        dofs = np.empty((len(e), self.num_dofs), dtype=np.int64)
        for k in range(3):
            dofs[:, k] = 3 * n0 + k
            dofs[:, 4 + k] = 3 * n1 + k
            dofs[:, 8 + k] = 3 * n2 + k
        dofs[:, 3] = self.num_node_dofs + e
        dofs[:, 7] = self.num_node_dofs + f
        return dofs

    def _measure(self, state: State, out: wp.array) -> None:
        """Strains of ``state`` (its frames are current, so the transport from them is the identity)."""
        d, s = self.der, state.dismech
        wp.launch(
            triplet_strain_kernel,
            dim=self.count,
            inputs=[
                state.particle_q, state.particle_q, s.edge_q, s.edge_d1_q, getattr(s, self._ref_twist),
                self._conn, d.edge_length,
            ],
            outputs=[out],
            device=self.device,
        )

    def begin_step(self, state_in: State) -> None:
        if self.uses_rate:
            self._measure(state_in, self._strain_prev)

    def assemble(
        self, state_in: State, state_out: State, residual: wp.array, hessian: SymmetricCSR, dt: float
    ) -> None:
        d = self.der
        s_in, s_out = state_in.dismech, state_out.dismech
        wp.launch(
            self._kernel,
            dim=self.count,
            inputs=[
                state_out.particle_q, state_in.particle_q, s_out.edge_q, s_in.edge_d1_q,
                getattr(s_in, self._ref_twist), self._conn, d.edge_length, self._params, self._rest,
                self._strain_prev, dt, self.num_node_dofs, self.dof_fixed,
            ],
            outputs=[residual, hessian.indptr, hessian.indices, hessian.vals],
            device=self.device,
        )

    def end_step(self, state_in: State, state_out: State) -> None:
        s_in, s_out = state_in.dismech, state_out.dismech
        wp.launch(
            advance_ref_twist_kernel,
            dim=self.count,
            inputs=[state_out.particle_q, s_out.edge_d1_q, self._conn, getattr(s_in, self._ref_twist)],
            outputs=[getattr(s_out, self._ref_twist)],
            device=self.device,
        )
