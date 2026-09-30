"""Adding DER rods to a ``ModelBuilder`` and the ``dismech`` custom attributes they fill.

Nodes are Newton particles (``particle_q``, ``particle_qd``, ``particle_mass``; a
fixed node has ``ParticleFlags.ACTIVE`` cleared). Edges and triplets are custom
frequencies holding what particles cannot (twist angle, reference director); each concrete
stencil class (e.g. ``linear_triplet``) is a frequency of its own. Unless
``proxies=False``, every edge also gets a kinematic, massless capsule proxy body (rendering
only unless ``collide=True``); the solver poses it, it is not simulated. Each proxy costs
Newton body and shape rows and a ``body_q`` entry in every state, so large rods that are not
drawn should turn them off. Rest strains are not stored: the solver measures them from the
initial configuration, so the rod starts unstressed. Edge tangents are not stored either;
they follow from the node positions.
"""

import copy
from typing import Literal

import numpy as np
import warp as wp
from newton import Model, ModelBuilder, ParticleFlags, Rod
from newton._src.core.types import Quat, Vec3

from .stencils import NAMESPACE, LinearDampedTriplet, LinearTriplet, Stencil, TripletStencil

_P = f"{NAMESPACE}:"

# Edge attributes: (frequency, name, dtype, references). Stencil attributes (triplets, ...)
# are declared by the stencils themselves.
ATTRIBUTES: list[tuple[str, str, type, str | None]] = [
    ("edge", "edge_inertia", float, None),
    ("edge", "edge_fixed", wp.int32, None),
    ("edge", "edge_node0", wp.int32, "particle"),
    ("edge", "edge_node1", wp.int32, "particle"),
    ("edge", "edge_body", wp.int32, "body"),
    ("edge", "edge_length", float, None),
]

# Time-evolving state: (frequency, name, dtype).
STATE_ATTRIBUTES: list[tuple[str, str, type]] = [
    ("edge", "edge_q", float),  # material twist angle theta
    ("edge", "edge_qd", float),
    ("edge", "edge_d1_q", wp.vec3),  # time-parallel reference director
]


def register_custom_attributes(builder: ModelBuilder, *stencils: type[Stencil]) -> None:
    """Register the ``dismech`` attributes (edges, then each given stencil class's) on ``builder``.

    Idempotent. Each concrete stencil class is its own frequency, so rods of different
    classes can share a builder; :func:`add_rod` registers the class it uses.
    """
    builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="edge", namespace=NAMESPACE))
    for stencil in stencils:
        stencil.register(builder)
    for frequency, name, dtype, references in ATTRIBUTES:
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name=name,
                dtype=dtype,
                frequency=_P + frequency,
                namespace=NAMESPACE,
                references=references,
            )
        )
    for frequency, name, dtype in STATE_ATTRIBUTES:
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name=name,
                dtype=dtype,
                frequency=_P + frequency,
                assignment=Model.AttributeAssignment.STATE,
                namespace=NAMESPACE,
            )
        )


def _rows(builder: ModelBuilder, name: str) -> int:
    values = builder.custom_attributes[_P + name].values
    return len(values) if values else 0


def _resolve(explicit, rigidity, length, default, n: int) -> np.ndarray:
    """Stiffness per entity: explicit value, else rigidity / length, else default."""
    if explicit is not None:
        return np.full(n, explicit)
    if rigidity is not None:
        return rigidity / length
    return np.full(n, default)


def fix_segment(builder: ModelBuilder, body: int | None = None, *, edge: int | None = None) -> None:
    """Clamp a segment: both its nodes and its twist.

    Name the segment by its proxy ``body`` or, for rods built with ``proxies=False``, by its
    ``edge`` index (what :func:`add_rod` returns then). Clamping the first segment of a rod
    gives a cantilever.
    """
    if (body is None) == (edge is None):
        raise ValueError("fix_segment: pass exactly one of `body` and `edge`")
    if edge is None:
        edge = builder.custom_attributes[_P + "edge_body"].values.index(body)
    builder.custom_attributes[_P + "edge_fixed"].values[edge] = 1
    for name in ("edge_node0", "edge_node1"):
        node = builder.custom_attributes[_P + name].values[edge]
        builder.particle_flags[node] &= ~ParticleFlags.ACTIVE


def add_rod(
    builder: ModelBuilder,
    rod: Rod,
    *,
    cfg: ModelBuilder.ShapeConfig | None = None,
    stretch_stiffness: float | None = None,
    stretch_damping: float | None = None,
    bend_stiffness: float | None = None,
    bend_damping: float | None = None,
    twist_stiffness: float | None = None,
    twist_damping: float | None = None,
    stencil: type[TripletStencil] | None = None,
    label: str | None = None,
    collide: bool = False,
    proxies: bool = True,
    color: Vec3 | None = None,
) -> list[int]:
    """Add an ordered chain (open or closed) of DER segments.

    Returns the proxy body indices, or, with ``proxies=False``, the (builder-global) edge
    indices; :func:`fix_segment` takes either.

    Stiffness per entity: the explicit argument, else the rod's section
    rigidity divided by the rest length (EA / l0 per edge, EI / L_dual and
    GJ / L_dual per triplet), else a default (stretch 1e5, bend 0, twist =
    bend). Damping is strain-rate viscosity in the same units; twist damping
    defaults to bend damping. Mass comes from ``cfg.density``.

    ``stencil`` is the concrete triplet class (and so the energy) of the rod's triplets (see
    :mod:`dismech_newton.stencils`); by default :class:`LinearDampedTriplet` if any damping is
    given and :class:`LinearTriplet` otherwise, so an undamped rod stores no damping. Rods of
    different classes can share a builder.

    The proxy capsules are render-only unless ``collide`` is set: ``finalize`` enumerates
    every pair of colliding shapes (O(segments^2) host and device memory), and the solver
    ignores contacts anyway.
    """
    points, edges, frames = rod._normalize_and_validate_geometry()
    if not Rod._is_ordered_chain_topology(len(points), edges):
        raise NotImplementedError("add_rod: only ordered chains are supported")
    cfg = copy.copy(cfg or builder.default_shape_cfg)
    if not collide:
        cfg.has_shape_collision = False
    radius = rod._resolve_radius() or 0.1
    points = np.asarray(points, dtype=float)
    frames = np.asarray(frames, dtype=float)

    # -- topology and rest geometry -------------------------------------------------
    n_e = len(points) - 1
    idx = np.arange(n_e)
    nodes = points[:-1] if rod.closed else points
    edge_nodes = np.column_stack((idx, (idx + 1) % n_e if rod.closed else idx + 1))
    first = idx if rod.closed else idx[:-1]
    triplets = np.column_stack((first, (first + 1) % n_e))
    n_t = len(triplets)

    vec = points[1:] - points[:-1]
    length = np.linalg.norm(vec, axis=1)
    tangent = vec / length[:, None]
    # d1: the segment frame's local +X (quaternion rotation of x), made orthogonal to t.
    u, w = frames[:, :3], frames[:, 3:]
    ux = np.cross(u, [1.0, 0.0, 0.0])
    d1 = np.array([1.0, 0.0, 0.0]) + 2.0 * w * ux + 2.0 * np.cross(u, ux)
    d1 -= np.sum(d1 * tangent, axis=1, keepdims=True) * tangent
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
    l_dual = 0.5 * (length[triplets[:, 0]] + length[triplets[:, 1]])

    # -- stiffness, folded per triplet -----------------------------------------------
    rigidities = rod._resolve_section_rigidities()
    ea, _, ei, gj = rigidities if rigidities is not None else (None,) * 4
    stretch_ke = _resolve(stretch_stiffness, ea, length, 1.0e5, n_e)
    bend_ke = _resolve(bend_stiffness, ei, l_dual, 0.0, n_t)
    twist_ke = _resolve(twist_stiffness, gj, l_dual, bend_ke, n_t)
    stretch_kd = np.full(n_e, stretch_damping or 0.0)
    bend_kd = np.full(n_t, bend_damping or 0.0)
    twist_kd = bend_kd if twist_damping is None else np.full(n_t, twist_damping)

    # An interior edge belongs to two triplets and shares its stretch between them.
    share = 1.0 / np.bincount(triplets.ravel(), minlength=n_e)[triplets]  # (T, 2)
    scale = length[triplets] ** 2 * share
    k = np.column_stack((stretch_ke[triplets] * scale, bend_ke, bend_ke, twist_ke))
    c = np.column_stack((stretch_kd[triplets] * scale, bend_kd, bend_kd, twist_kd))
    if stencil is None:
        stencil = LinearDampedTriplet if np.any(c) else LinearTriplet

    # -- kinematic capsule proxies (body frame at the segment midpoint) ---------------
    bodies, shapes = [], []
    for e in range(n_e if proxies else 0):
        mid = 0.5 * (points[e] + points[e + 1])
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*mid.tolist()), wp.quat(*frames[e].tolist())),
            mass=0.0,
            lock_inertia=True,
            is_kinematic=True,
            label=f"{label}_edge_body_{e}" if label else None,
        )
        shapes.append(
            builder.add_shape_capsule(
                body,
                radius=radius,
                half_height=0.5 * float(length[e]),
                cfg=cfg,
                label=f"{label}_edge_capsule_{e}" if label else None,
                color=color if color is not None else ModelBuilder._DEFAULT_ROD_COLOR,
            )
        )
        bodies.append(body)
    for a, b in triplets if proxies else ():  # adjacent proxies overlap at the shared node
        builder.add_shape_collision_filter_pair(shapes[a], shapes[b])

    # -- attribute rows ---------------------------------------------------------------
    register_custom_attributes(builder, stencil)
    node0, edge0 = builder.particle_count, _rows(builder, "edge_length")

    # Lumped masses: each edge splits its mass between its nodes; twist inertia of a solid cylinder.
    edge_mass = cfg.density * np.pi * radius**2 * length
    node_mass = np.zeros(len(nodes))
    np.add.at(node_mass, edge_nodes[:, 0], 0.5 * edge_mass)
    np.add.at(node_mass, edge_nodes[:, 1], 0.5 * edge_mass)

    for x, m in zip(nodes, node_mass):
        builder.add_particle(wp.vec3(*x.tolist()), wp.vec3(0.0, 0.0, 0.0), float(m), radius=radius)
    for e in range(n_e):
        builder.add_custom_values(
            **{
                _P + "edge_inertia": float(0.5 * edge_mass[e] * radius**2),
                _P + "edge_fixed": 0,
                _P + "edge_node0": node0 + int(edge_nodes[e, 0]),
                _P + "edge_node1": node0 + int(edge_nodes[e, 1]),
                _P + "edge_body": bodies[e] if proxies else -1,
                _P + "edge_length": float(length[e]),
                _P + "edge_q": 0.0,
                _P + "edge_qd": 0.0,
                _P + "edge_d1_q": wp.vec3(*d1[e].tolist()),
            }
        )
    stencil.add_rows(builder, edge0, triplets, k, c)
    return bodies if proxies else list(range(edge0, edge0 + n_e))


def add_rod_graph(
    builder: ModelBuilder,
    node_positions: list[Vec3],
    edges: list[tuple[int, int]],
    *,
    radius: float = 0.1,
    cfg: ModelBuilder.ShapeConfig | None = None,
    stretch_stiffness: float | None = None,
    stretch_damping: float | None = None,
    shear_stiffness: float | None = None,
    shear_damping: float | None = None,
    bend_stiffness: float | None = None,
    bend_damping: float | None = None,
    twist_stiffness: float | None = None,
    twist_damping: float | None = None,
    label: str | None = None,
    wrap_in_articulation: bool = True,
    quaternions: list[Quat] | None = None,
    junction_collision_filter: bool = True,
    color: Vec3 | None = None,
    body_frame_origin: Literal["start", "com"] | None = None,
) -> tuple[list[int], list[int]]:
    """DER counterpart of :meth:`newton.ModelBuilder.add_rod_graph`, with the same arguments.

    Builds a :class:`newton.Rod` from ``node_positions`` and ``edges`` and hands it to
    :func:`add_rod`, so only ordered chains are supported; a ring given as the chain edges
    plus a closing ``(n - 1, 0)`` edge becomes a closed rod. Returns ``(body_indices, [])``:
    bodies follow ``edges``, and a DER rod has no joints.

    DER has no shear, so ``shear_stiffness`` and ``shear_damping`` must be None. The proxies
    are kinematic and posed at the segment midpoint, so ``wrap_in_articulation``,
    ``junction_collision_filter`` and ``body_frame_origin`` have no effect.
    """
    if shear_stiffness is not None or shear_damping is not None:
        raise NotImplementedError("add_rod_graph: DER rods have no shear mode")
    points = np.asarray(node_positions, dtype=float)
    edge_array = np.asarray(edges, dtype=int).reshape(-1, 2)
    n = len(points)
    ring = np.vstack((Rod._generate_ordered_chain_edges(n), [[n - 1, 0]]))
    if n >= 3 and np.array_equal(edge_array, ring):
        rod = Rod(np.vstack((points, points[:1])), quaternions=quaternions, closed=True, radius=radius)
    else:
        rod = Rod(points, edges=edge_array, quaternions=quaternions, radius=radius)
    bodies = add_rod(
        builder,
        rod,
        cfg=cfg,
        stretch_stiffness=stretch_stiffness,
        stretch_damping=stretch_damping,
        bend_stiffness=bend_stiffness,
        bend_damping=bend_damping,
        twist_stiffness=twist_stiffness,
        twist_damping=twist_damping,
        label=label,
        color=color,
    )
    return bodies, []
