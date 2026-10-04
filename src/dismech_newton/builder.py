"""Adding DER rods to a ``ModelBuilder``: nodes are particles, edges and triplets ``dismech`` frequencies."""

import copy

import numpy as np
import warp as wp
from newton import Model, ModelBuilder, ParticleFlags, Rod
from newton._src.core.types import Vec3

from .strains import vec5f, vec10f

NAMESPACE = "dismech"
_P = f"{NAMESPACE}:"

# (frequency, name, dtype, references)
ATTRIBUTES = [
    ("edge", "edge_inertia", float, None),
    ("edge", "edge_fixed", wp.int32, None),
    ("edge", "edge_node0", wp.int32, "particle"),
    ("edge", "edge_node1", wp.int32, "particle"),
    ("edge", "edge_body", wp.int32, "body"),
    ("edge", "edge_length", float, None),
    ("triplet", "triplet_edge0", wp.int32, _P + "edge"),  # scalar int32, so merging builders offsets them
    ("triplet", "triplet_edge1", wp.int32, _P + "edge"),
    ("triplet", "triplet_params", vec10f, None),  # [stiffness, damping] per strain
]

# Time-evolving state: (frequency, name, dtype, default).
STATE_ATTRIBUTES = [
    ("edge", "edge_q", float, None),  # material twist angle theta
    ("edge", "edge_qd", float, None),
    ("edge", "edge_d1_q", wp.vec3, None),  # time-parallel reference director
    ("triplet", "triplet_ref_twist_q", float, None),
    # The strains at the end of the last step (strain-rate damping); NaN until a step measures them.
    ("triplet", "triplet_strain_q", vec5f, vec5f(*([float("nan")] * 5))),
]


def register_custom_attributes(builder: ModelBuilder) -> None:
    """Register the ``dismech`` frequencies and attributes on ``builder`` (idempotent)."""
    for name in ("edge", "triplet"):
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name=name, namespace=NAMESPACE))
    for frequency, name, dtype, references in ATTRIBUTES:
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name=name, dtype=dtype, frequency=_P + frequency, namespace=NAMESPACE, references=references
            )
        )
    for frequency, name, dtype, default in STATE_ATTRIBUTES:
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name=name, dtype=dtype, frequency=_P + frequency, assignment=Model.AttributeAssignment.STATE,
                namespace=NAMESPACE, default=default,
            )
        )


def _resolve(explicit, rigidity, length, default, n: int) -> np.ndarray:
    if explicit is not None:
        return np.full(n, explicit)
    if rigidity is not None:
        return rigidity / length
    return np.full(n, default)


def fix_segment(builder: ModelBuilder, body: int | None = None, *, edge: int | None = None) -> None:
    """Clamp a segment (both nodes and its twist), by proxy ``body`` or ``edge`` index."""
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
    label: str | None = None,
    collide: bool = False,
    proxies: bool = True,
    color: Vec3 | None = None,
) -> list[int]:
    """Add an ordered chain of DER segments; returns the proxy bodies (edge indices without proxies).

    Stiffness defaults to section rigidity / rest length, else stretch 1e5, bend 0, twist = bend.
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

    # -- stiffness and damping, folded per triplet -----------------------------------
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
    params = np.column_stack(
        (stretch_ke[triplets] * scale, bend_ke, bend_ke, twist_ke, stretch_kd[triplets] * scale, bend_kd, bend_kd,
         twist_kd)
    )

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
    register_custom_attributes(builder)
    edge_values = builder.custom_attributes[_P + "edge_length"].values
    node0, edge0 = builder.particle_count, len(edge_values) if edge_values else 0

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
    for t in range(n_t):
        builder.add_custom_values(
            **{
                _P + "triplet_edge0": edge0 + int(triplets[t, 0]),
                _P + "triplet_edge1": edge0 + int(triplets[t, 1]),
                _P + "triplet_params": vec10f(*params[t].tolist()),
                _P + "triplet_ref_twist_q": 0.0,  # the first strain evaluation yields the reference twist itself
            }
        )
    return bodies if proxies else list(range(edge0, edge0 + n_e))


def add_colliding_rod(
    builder: ModelBuilder,
    rod: Rod,
    *,
    cfg: ModelBuilder.ShapeConfig | None = None,
    contact_exclusion: float = 1.5,
    contact_gap: float = 0.5,
    **kwargs,
) -> list[int]:
    """:func:`add_rod` with colliding proxies; pairs closer than ``contact_exclusion`` thicknesses along the rod
    are filtered. Without ``cfg.gap``, the capsules report pairs within ``contact_gap`` radii of touching (fast
    motion is covered by the pipeline's speculative contacts); a larger gap adds far contacts that slow ADMM."""
    if not kwargs.get("proxies", True):
        raise ValueError("add_colliding_rod: collision needs the capsule proxies (proxies=True)")
    radius = rod._resolve_radius() or 0.1
    cfg = copy.copy(cfg or builder.default_shape_cfg)
    if cfg.gap is None:
        cfg.gap = contact_gap * radius
    bodies = add_rod(builder, rod, cfg=cfg, collide=True, **kwargs)

    cutoff = contact_exclusion * 2.0 * radius
    points = np.asarray(rod._normalize_and_validate_geometry()[0], dtype=float)
    length = np.linalg.norm(points[1:] - points[:-1], axis=1)
    n_e = len(length)
    shapes = [builder.body_shapes[b][0] for b in bodies]
    for a in range(n_e):
        dist = 0.0
        for k in range(1, n_e if rod.closed else n_e - a):
            if dist >= cutoff:
                break
            b = (a + k) % n_e
            builder.add_shape_collision_filter_pair(shapes[min(a, b)], shapes[max(a, b)])
            dist += length[b]
    return bodies
