"""Adding rods and shells to a ``ModelBuilder``: nodes are particles; a rod's edges and triplets, a shell's hinges
and faces are ``dismech`` frequencies."""

import copy

import numpy as np
import warp as wp
from newton import Model, ModelBuilder, ParticleFlags, Rod
from newton._src.core.types import Vec3

vec5f = wp.types.vector(5, float)  # a triplet's strains [eps_e, eps_f, kappa1, kappa2, tau]
vec10f = wp.types.vector(10, float)  # [stiffness, damping] per triplet strain
vec6f = wp.types.vector(6, float)  # a face's strains (membrane 3, bending 3)
vec12f = wp.types.vector(12, float)

NAMESPACE = "dismech"
_P = f"{NAMESPACE}:"
ROD_GROUP = -0x5DE7  # every colliding rod capsule's collision group: a negative group collides with all but itself

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
    # Midedge shells (shell.py): a hinge per mesh edge, its midedge angle a DOF.
    ("hinge", "hinge_fixed", wp.int32, None),
    *(("face", f"face_node{k}", wp.int32, "particle") for k in range(6)),  # its 3 vertices, then the 3 flaps
    *(("face", f"face_hinge{k}", wp.int32, _P + "hinge") for k in range(3)),  # hinge k: the edge opposite node k
    ("face", "face_sign", wp.vec3, None),  # +-1: the hinge's angle as the face sees it
    ("face", "face_flap", wp.vec3, None),  # 1 where the hinge has a second face
    ("face", "face_strain_map", wp.mat33, None),  # (m00, m01, m11) -> whitened strains in the rest frame
    ("face", "face_params", vec12f, None),  # [stiffness, damping] per strain (membrane 3, bending 3)
]

# Time-evolving state: (frequency, name, dtype, default).
STATE_ATTRIBUTES = [
    ("edge", "edge_q", float, None),  # material twist angle theta
    ("edge", "edge_qd", float, None),
    ("edge", "edge_d1_q", wp.vec3, None),  # time-parallel reference director
    ("triplet", "triplet_ref_twist_q", float, None),
    # The strains at the end of the last step (strain-rate damping); NaN until a step measures them.
    ("triplet", "triplet_strain_q", vec5f, vec5f(*([float("nan")] * 5))),
    ("hinge", "hinge_q", float, None),  # midedge angle
    ("hinge", "hinge_qd", float, None),
    ("face", "face_strain_q", vec6f, vec6f(*([float("nan")] * 6))),
]


def register_custom_attributes(builder: ModelBuilder) -> None:
    """Register the ``dismech`` frequencies and attributes on ``builder`` (idempotent)."""
    for name in ("edge", "triplet", "hinge", "face"):
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


def complete(ns, attributes: list, device) -> None:
    """Empty arrays for the ``attributes`` Newton leaves out of ``ns`` (those of a frequency without rows)."""
    for _, name, dtype, *_ in attributes:
        if getattr(ns, name, None) is None:
            setattr(ns, name, wp.zeros(0, dtype=dtype, device=device))


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

    Stiffness defaults to section rigidity / rest length, else stretch 1e5, bend 0, twist = bend. ``proxies``: a
    kinematic capsule per segment, posed every step (rendering, and what Newton's pipeline sees). ``collide``: the
    rod takes part in the solver's contacts (:mod:`.collide`), its friction ``cfg.mu`` (without ``cfg``, 0.3); its
    capsules are in :data:`ROD_GROUP`, so a pipeline never pairs two of them.
    """
    points, edges, frames = rod._normalize_and_validate_geometry()
    if not Rod._is_ordered_chain_topology(len(points), edges):
        raise NotImplementedError("add_rod: only ordered chains are supported")
    if collide and not proxies:
        raise ValueError("add_rod: a colliding rod needs its capsule proxies (proxies=True)")
    default = cfg is None
    cfg = copy.copy(cfg or builder.default_shape_cfg)
    if collide:
        cfg.collision_group = ROD_GROUP
        if default:
            cfg.mu = 0.3
    else:
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


# -- shells: Newton's cloth mesh --------------------------------------------------------------------------


def add_cloth_mesh(
    builder: ModelBuilder,
    *,
    vertices,
    indices,
    pos=(0.0, 0.0, 0.0),
    rot=None,
    scale: float = 1.0,
    vel=(0.0, 0.0, 0.0),
    bend_stiffness: float | None = None,
    damping: float | None = None,
    **kwargs,
) -> np.ndarray:
    """:meth:`newton.ModelBuilder.add_cloth_mesh` as a midedge shell; returns its vertices' particle indices.

    Newton supplies the particles, their lumped masses (``density`` per area), ``particle_radius`` (the contact
    thickness), the triangles and the bending edges; a hinge is a bending edge. The material comes from Newton's
    parameters as its VBD solver reads them:

    - membrane: ``tri_ke``, ``tri_ka``, the 2D Lame ``mu``, ``lam`` (exact at small strain);
    - bending: the plate's ``D = edge_ke A_e / L_e`` per hinge (``A_e`` the area of its two faces), averaged over a
      face's hinges: Newton's dihedral spring is anisotropic, so this is within about 10% along a regular grid's
      axes and 30% across its diagonals. ``bend_stiffness`` sets ``D`` [N m] instead;
    - damping as a time: ``4 tri_kd / (lam + 2 mu)`` (membrane), ``edge_kd / edge_ke`` (bending), or ``damping`` [s]
      for both.

    The rest shape is the initial one. Fix vertices through their particle flags; a clamped edge needs its
    neighbouring row too (the midedge angles stay free).
    """
    register_custom_attributes(builder)
    v0, t0, e0 = builder.particle_count, len(builder.tri_indices), len(builder.edge_indices)
    builder.add_cloth_mesh(
        pos=pos,
        rot=wp.quat_identity() if rot is None else rot,
        scale=scale,
        vel=vel,
        vertices=vertices,
        indices=indices,
        **kwargs,
    )
    tris = np.asarray(builder.tri_indices[t0:], dtype=np.int64).reshape(-1, 3)
    edges = np.asarray(builder.edge_indices[e0:], dtype=np.int64).reshape(-1, 4)  # (o0, o1, v0, v1)
    ke, ka, kd = np.asarray(builder.tri_materials[t0:], dtype=np.float64).reshape(-1, 5)[:, :3].T
    edge_ke, edge_kd = np.asarray(builder.edge_bending_properties[e0:], dtype=np.float64).reshape(-1, 2).T
    x = np.asarray(builder.particle_q, dtype=np.float32).astype(np.float64)  # as the model stores them

    # -- topology: hinge i of a face is its edge opposite node i, the flap the other face's opposite node -------
    hinge_of = {(min(a, b), max(a, b)): h for h, (_, _, a, b) in enumerate(edges)}
    face_nodes = np.concatenate([tris, tris], axis=1)
    face_hinges, sign, flap = np.zeros((len(tris), 3), np.int64), np.zeros((len(tris), 3)), np.zeros((len(tris), 3))
    for f, tri in enumerate(tris):
        for i in range(3):
            a, b = tri[(i + 1) % 3], tri[(i + 2) % 3]
            h = hinge_of[(min(a, b), max(a, b))]
            o0, o1, h0, _ = edges[h]
            face_hinges[f, i] = h
            sign[f, i] = 1.0 if h0 == a else -1.0
            other = o1 if o0 == tri[i] else o0
            if other >= 0:
                face_nodes[f, 3 + i], flap[f, i] = other, 1.0
    if not np.allclose(np.bincount(face_hinges.ravel(), (sign * flap).ravel(), minlength=len(edges)), 0.0):
        raise ValueError("add_cloth_mesh: inconsistently oriented faces")

    # -- rest frame and material --------------------------------------------------------------------------------
    p = x[tris]
    eu, ev = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    metric = np.stack(
        [np.stack([(eu * eu).sum(1), (eu * ev).sum(1)], -1), np.stack([(eu * ev).sum(1), (ev * ev).sum(1)], -1)], -2
    )
    area = 0.5 * np.sqrt(np.linalg.det(metric))
    L = np.linalg.inv(np.linalg.cholesky(metric))  # L a L^T = I
    lam_2mu = ka + 2.0 * ke  # 2D Lame: lam = tri_ka, mu = tri_ke
    if bend_stiffness is None:
        hinge_area = np.bincount(face_hinges.ravel(), np.repeat(area, 3), minlength=len(edges))
        hinge_len = np.linalg.norm(x[edges[:, 3]] - x[edges[:, 2]], axis=1)
        bend_stiffness = (edge_ke * hinge_area / hinge_len)[face_hinges].mean(1)
    gamma = (4.0 * kd / lam_2mu, (edge_kd / np.where(edge_ke > 0, edge_ke, np.inf))[face_hinges].mean(1))
    K = np.repeat(np.column_stack([0.25 * area * lam_2mu, area * bend_stiffness]), 3, axis=1)
    C = K * (damping if damping is not None else np.repeat(np.column_stack(gamma), 3, axis=1))
    params, nu = np.concatenate([K, C], axis=1), ka / lam_2mu
    # W = R T: T takes (m00, m01, m11) to (S11, S22, S12) of S = L M L^T, R whitens (module docstring)
    l00, l01, l10, l11 = L[:, 0, 0], L[:, 0, 1], L[:, 1, 0], L[:, 1, 1]
    T = np.stack(
        [
            np.stack([l00 * l00, 2 * l00 * l01, l01 * l01], -1),
            np.stack([l10 * l10, 2 * l10 * l11, l11 * l11], -1),
            np.stack([l00 * l10, l00 * l11 + l01 * l10, l01 * l11], -1),
        ],
        -2,
    )
    R = np.zeros((len(tris), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 1], R[:, 2, 2] = 1.0, nu, np.sqrt(1.0 - nu**2), np.sqrt(2.0 * (1.0 - nu))
    W = R @ T

    for _ in edges:
        builder.add_custom_values(
            **{
                _P + "hinge_fixed": 0,
                _P + "hinge_q": 0.0,
                _P + "hinge_qd": 0.0,
            }
        )
    hinge0 = len(builder.custom_attributes[_P + "hinge_fixed"].values) - len(edges)
    for f in range(len(tris)):
        builder.add_custom_values(
            **{_P + f"face_node{k}": int(face_nodes[f, k]) for k in range(6)},
            **{_P + f"face_hinge{k}": hinge0 + int(face_hinges[f, k]) for k in range(3)},
            **{
                _P + "face_sign": wp.vec3(*sign[f]),
                _P + "face_flap": wp.vec3(*flap[f]),
                _P + "face_strain_map": wp.mat33(*W[f].ravel().tolist()),
                _P + "face_params": vec12f(*params[f].tolist()),
            },
        )
    return np.arange(v0, builder.particle_count)
