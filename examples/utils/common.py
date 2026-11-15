"""What the examples share, none of it part of the ``dismech_newton`` package.

- :class:`CableExample`: the frame loop in Newton's example format (drive, step, CUDA graph replay).
- :class:`FrameGraph`, :class:`TapedGraph`: device work run eagerly once, then replayed as CUDA graphs.
- :class:`Drive`, :func:`segment_dofs`: move clamped segments along a prescribed path.
- :func:`frame_box`: point the viewer's camera at a box.
- :func:`default_frames`: the ``--num-frames`` default from a simulated duration.
- :func:`smoothstep`: a smooth ramp from 0 to 1.
- :func:`grid`: a triangulated plate.
- :func:`close_pairs`, :func:`segment_distance`, :func:`capsules`, :class:`Crossings`, :func:`newton_grid`: geometry
  checks & grids; :func:`capsule_poses`: capsules along a polyline, for drawing recorded node positions.
- :data:`TCP`, :func:`add_franka`, :func:`set_franka_targets`: shared Franka URDF & IK target helpers.
- :func:`inset`: the theory plots drawn in the viewer.
- :func:`cached`: results that take minutes, computed once per configuration.
"""

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial import cKDTree

from dismech_newton import flatten_state

wp.set_module_options({"enable_backward": False})


def smoothstep(t, t0: float, t1: float):
    """0 before ``t0``, 1 after ``t1``, cubic between (``t`` a scalar or an array)."""
    s = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


# -- results computed once ---------------------------------------------------------------

CACHE = Path(__file__).resolve().parents[2] / ".cache" / "examples"


def cached(name: str, config: dict, compute, fresh: bool = False) -> dict[str, np.ndarray]:
    """``compute()`` (a dict of arrays), stored in ``.cache/examples/<name>-<hash of config>.npz``.

    ``config`` (JSON-able) holds everything the result depends on: a changed value computes and stores a new
    file, and the old ones stay. Bump a version in it when the code computing the result changes. ``fresh``
    recomputes regardless. ``rm -rf .cache`` clears every cached result.
    """
    text = json.dumps(config, sort_keys=True)
    path = CACHE / f"{name}-{hashlib.sha256(text.encode()).hexdigest()[:12]}.npz"
    if path.exists() and not fresh:
        with np.load(path) as f:
            return {k: f[k] for k in f.files if k != "config"}
    out = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, config=text, **out)
    tmp.replace(path)  # never a half-written result
    print(f"cached {path.relative_to(CACHE.parents[1])}")
    return out


# -- prescribed motion of clamped segments ------------------------------------------------


PAD_KH = 1.0e8  # a grip pad's pressure per depth [N/m^3] (an elastomer's E over its depth; Drake's 1e5-1e7 Pa / mm)


def pads(model, bodies, kh: float = PAD_KH, dissipation: float = 0.0) -> int:
    """Make the shapes of ``bodies`` (indices) hydroelastic grip pads in our contact (``SolverDiSMech``): Newton's
    ``HYDROELASTIC`` flag and ``kh``, set on the finalized model (Newton's builder would ask the shapes for SDFs,
    which our contact does not use), and Hunt-Crossley's ``dissipation`` [s/m]. Before the solver is built; returns
    the shapes padded."""
    from newton import ShapeFlags  # noqa: PLC0415

    on = np.isin(model.shape_body.numpy(), list(bodies))
    flags, k = model.shape_flags.numpy(), model.shape_material_kh.numpy()
    model.shape_flags.assign(np.where(on, flags | int(ShapeFlags.HYDROELASTIC), flags).astype(flags.dtype))
    model.shape_material_kh.assign(np.where(on, kh, k).astype(k.dtype))
    model.shape_pad_dissipation = np.where(on, dissipation, 0.0)
    return int(on.sum())


TCP = 0.107  # hand origin to the point between the fingertips [m]


def add_franka(builder, base_z: float) -> None:
    """Add the Franka FR3 + hand URDF at ``base_z``, initialised at ``FRANKA_Q``."""
    import newton.utils  # noqa: PLC0415
    from newton.examples.multiphysics.example_franka_cable_ik_pick_place import FRANKA_Q  # noqa: PLC0415

    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        xform=wp.transform(wp.vec3(0.0, 0.0, base_z), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
        joint_damping_mode="drive",  # Newton 1.6's mapping (the URDF damping drives the joints)
    )
    n = len(FRANKA_Q)
    builder.joint_q[-n:] = FRANKA_Q
    builder.joint_target_q[-n:] = FRANKA_Q


@wp.kernel
def set_franka_targets(ik_q: wp.array2d[float], finger: wp.array[float], target: wp.array2d[float]):
    """The IK's arm joints, the drive's finger opening."""
    k = wp.tid()
    if k < ik_q.shape[1] - 2:
        target[0, k] = ik_q[0, k]
    else:
        target[0, k] = finger[0]


def segment_dofs(model, body: int, twist_only: bool = False) -> list[int]:
    """DOFs ``[x0, x1, theta]`` (or ``[theta]``) of the segment whose proxy is ``body``."""
    der = model.dismech
    e = int(np.flatnonzero(der.edge_body.numpy() == body)[0])
    n0, n1 = int(der.edge_node0.numpy()[e]), int(der.edge_node1.numpy()[e])
    theta = [3 * model.particle_count + e]
    return theta if twist_only else [3 * n0, 3 * n0 + 1, 3 * n0 + 2, 3 * n1, 3 * n1 + 1, 3 * n1 + 2, *theta]


@wp.kernel
def _drive_kernel(dofs: wp.array[wp.int32], start: wp.array[float], end: wp.array[float], s: float,
                  q: wp.array[float]):
    i = wp.tid()
    q[dofs[i]] = start[i] + s * (end[i] - start[i])


class Drive:
    """Prescribes fixed DOFs, ``state.dismech.q[dofs] = rest + delta``, interpolated over a frame.

    :meth:`set` (host, once per frame) takes the frame's start and end ``delta``; calling the drive
    with the substep's fraction ``s`` of the frame launches one kernel, so it can be graph-captured.
    """

    def __init__(self, model, dofs: list[int]):
        q0 = model.particle_q.numpy().ravel()
        if max(dofs) >= len(q0):  # a twist DOF: the model has rods
            q0 = np.concatenate([q0, model.dismech.edge_q.numpy()])
        self.rest = q0[dofs].astype(np.float64)
        self.dofs = wp.array(dofs, dtype=wp.int32, device=model.device)
        self.start = wp.array(self.rest, dtype=float, device=model.device)
        self.end = wp.array(self.rest, dtype=float, device=model.device)

    def set(self, start, end) -> None:
        self.start.assign((self.rest + start).astype(np.float32))
        self.end.assign((self.rest + end).astype(np.float32))

    def __call__(self, state, s: float) -> None:
        wp.launch(_drive_kernel, dim=len(self.rest), inputs=[self.dofs, self.start, self.end, s],
                  outputs=[state.dismech.q])


# -- CUDA graphs ---------------------------------------------------------------------------


def capturable(solver) -> bool:
    """Whether work stepping ``solver`` can be captured: on CUDA, and as the solver says (DER solvers decide on their
    first step; Newton's are taken as capturable)."""
    return wp.get_device(solver.model.device).is_cuda and getattr(solver, "graph_capturable", True)


class FrameGraph:
    """``self()`` runs ``work`` (device work only): eagerly until the solver can be captured (it sets itself up on
    its first step), then captured once and replayed. ``work`` must leave the arrays it reads where they were (e.g.
    an even count of state swaps); ``enabled=False`` keeps it eager."""

    def __init__(self, work, solver, enabled: bool = True):
        self.work, self.solver, self.enabled = work, solver, enabled
        self.graph = None

    def __call__(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
            return
        self.work()
        if self.enabled and capturable(self.solver):
            was_gc = gc.isenabled()
            gc.disable()
            try:
                with wp.ScopedCapture() as capture:
                    self.work()
            finally:
                if was_gc:
                    gc.enable()
            self.graph = capture.graph


class TapedGraph:
    """A taped ``run()`` and its backward: eagerly until the solver can be captured, then two CUDA graphs.

    :meth:`forward` returns ``run``'s output (the same arrays on every replay); seed the outputs' ``.grad`` between it
    and :meth:`backward`, which leaves the inputs' gradients. Every :meth:`forward` zeroes the tape's gradients."""

    def __init__(self, run, solver):
        self.run, self.solver = run, solver
        self.tape, self.output, self.graphs = None, None, None

    def forward(self):
        if self.tape is not None:
            self.tape.zero()
        if self.graphs is not None:
            wp.capture_launch(self.graphs[0])
        else:
            self.tape = wp.Tape()
            with self.tape:
                self.output = self.run()
        return self.output

    def backward(self) -> None:
        if self.graphs is not None:
            wp.capture_launch(self.graphs[1])
            return
        self.tape.backward()
        if capturable(self.solver):  # a capture runs nothing: the gradients just computed stay
            tape = wp.Tape()
            was_gc = gc.isenabled()
            gc.disable()
            try:
                with wp.ScopedCapture() as forward:
                    with tape:
                        output = self.run()
                with wp.ScopedCapture() as backward:
                    tape.backward()
            finally:
                if was_gc:
                    gc.enable()
            self.tape, self.output, self.graphs = tape, output, (forward.graph, backward.graph)


# -- the frame loop -----------------------------------------------------------------------


class CableExample:
    """Frame loop of the examples, in Newton's example format.

    A subclass builds its model and solver and calls :meth:`start`; :meth:`drive` sets the drives
    for each frame. The first frame runs eagerly (the solver sets itself up), later frames replay
    one CUDA graph of the frame's substeps (an even count keeps ``state_0`` in place). With
    ``plot_every``, the viewer shows :meth:`image` (the theory plot, titled ``plot_name``) every that many frames.
    """

    fps, substeps, capture = 60, 8, True
    plot_every, plot_name = 0, "theory"
    drives: tuple = ()
    control = None  # passed to every solver step

    def start(self, viewer, model, solver):
        self.viewer, self.model, self.solver = viewer, model, solver
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.substeps
        self.sim_time = 0.0
        self.frame = 0
        self.state_0, self.state_1 = model.state(), model.state()
        flatten_state(self.state_0)
        flatten_state(self.state_1)
        self.graph = FrameGraph(self.simulate, solver, enabled=self.capture)
        if viewer is not None:  # None: a simulation another example draws
            viewer.set_model(model)

    def drive(self, t0: float, t1: float) -> None:
        """Set the drives for the frame from ``t0`` to ``t1`` (host)."""

    def simulate(self):
        for k in range(self.substeps):
            for drive in self.drives:
                drive(self.state_0, (k + 1) / self.substeps)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.post_substep()

    def post_substep(self) -> None:
        """After each substep, inside the frame's graph: device work only."""

    def step(self):
        self.drive(self.sim_time, self.sim_time + self.frame_dt)
        self.graph()
        self.sim_time += self.frame_dt
        self.frame += 1

    def log_plot(self):
        """Every ``plot_every`` frames, show :meth:`image` in the viewer."""
        if self.plot_every and self.frame % self.plot_every == 0:
            self.viewer.log_image(self.plot_name, self.image())

    def render(self):
        self.log_plot()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


def default_frames(parser, seconds) -> None:
    """Without ``--num-frames``, run for ``seconds(known_args)`` of simulated time."""
    parser.set_defaults(num_frames=0)
    known = parser.parse_known_args()[0]
    if known.num_frames <= 0:
        parser.set_defaults(num_frames=int(np.ceil(seconds(known) * CableExample.fps)))


def frame_box(viewer, left: float, right: float, bottom: float, top: float, *, y: float = 0.0,
              pitch: float = 0.0) -> None:
    """Look along ``+y`` (tilted down by ``-pitch`` degrees) at the box ``[left, right] x [bottom, top]`` in
    the ``xz`` plane at ``y``, filling the view; nothing without a camera (the null viewer)."""
    camera = getattr(viewer, "camera", None)
    if camera is None:
        return
    aspect = camera.width / camera.height if camera.height else 16.0 / 9.0
    half = max(0.5 * (top - bottom), 0.5 * (right - left) / aspect)
    dist = half / np.tan(np.deg2rad(0.5 * camera.fov))
    p = np.deg2rad(pitch)
    center = np.array([0.5 * (left + right), y, 0.5 * (top + bottom)])
    pos = center - dist * np.array([0.0, np.cos(p), np.sin(p)])
    viewer.set_camera(pos=wp.vec3(*pos), pitch=float(pitch), yaw=90.0)


# -- geometry checks ----------------------------------------------------------------------


@wp.kernel
def _boxes_kernel(x: wp.array[wp.vec3], tris: wp.array[wp.vec3i], lo: wp.array[wp.vec3], hi: wp.array[wp.vec3]):
    t = wp.tid()
    a, b, c = x[tris[t][0]], x[tris[t][1]], x[tris[t][2]]
    lo[t] = wp.min(wp.min(a, b), c)
    hi[t] = wp.max(wp.max(a, b), c)


@wp.kernel
def _crossings_kernel(bvh: wp.uint64, x: wp.array[wp.vec3], edges: wp.array[wp.vec2i], tris: wp.array[wp.vec3i],
                      count: wp.array[wp.int32], hits: wp.array[wp.vec2i]):
    """Per edge: the faces it shares no vertex with that it passes through (its ends on either side of the face's
    plane, where it meets the plane inside the face)."""
    e = wp.tid()
    i, j = edges[e][0], edges[e][1]
    a, b = x[i], x[j]
    query = wp.bvh_query_aabb(bvh, wp.min(a, b), wp.max(a, b))
    t = int(0)
    while wp.bvh_query_next(query, t):
        f = tris[t]
        if f[0] == i or f[1] == i or f[2] == i or f[0] == j or f[1] == j or f[2] == j:
            continue
        v0, v1, v2 = x[f[0]], x[f[1]], x[f[2]]
        n = wp.cross(v1 - v0, v2 - v0)
        da, db = wp.dot(a - v0, n), wp.dot(b - v0, n)
        if da * db >= 0.0:
            continue
        y = a + da / (da - db) * (b - a)
        if (wp.dot(wp.cross(v1 - v0, y - v0), n) >= 0.0 and wp.dot(wp.cross(v2 - v1, y - v1), n) >= 0.0
                and wp.dot(wp.cross(v0 - v2, y - v2), n) >= 0.0):
            k = wp.atomic_add(count, 0, 1)
            if k < hits.shape[0]:
                hits[k] = wp.vec2i(e, t)


class Crossings:
    """How many mesh edges pass through a face they share no vertex with, on the device; ``hits``: the first
    (edge, face) pairs found."""

    def __init__(self, edges: np.ndarray, tris: np.ndarray, device):
        self.edges = wp.array(edges.astype(np.int32), dtype=wp.vec2i, device=device)
        self.tris = wp.array(tris.astype(np.int32), dtype=wp.vec3i, device=device)
        self.lo, self.hi = (wp.zeros(len(tris), dtype=wp.vec3, device=device) for _ in range(2))
        self.bvh = None
        self.count = wp.zeros(1, dtype=wp.int32, device=device)
        self.hits = wp.zeros(64, dtype=wp.vec2i, device=device)

    def __call__(self, x: wp.array) -> int:
        wp.launch(_boxes_kernel, dim=self.tris.shape[0], inputs=[x, self.tris, self.lo, self.hi])
        if self.bvh is None:
            self.bvh = wp.Bvh(self.lo, self.hi)
        else:
            self.bvh.refit()
        self.count.zero_()
        wp.launch(_crossings_kernel, dim=self.edges.shape[0],
                  inputs=[self.bvh.id, x, self.edges, self.tris, self.count, self.hits])
        return int(self.count.numpy()[0])


def newton_grid(**kwargs):
    """Vertices (world), faces, fixed vertices and panel coordinates (the grid's x, y) of Newton's
    ``add_cloth_grid``."""
    import newton  # noqa: PLC0415

    b = newton.ModelBuilder()
    b.add_cloth_grid(**kwargs)
    fixed = (np.asarray(b.particle_flags) & int(newton.ParticleFlags.ACTIVE)) == 0
    x = np.asarray(b.particle_q)
    panel = (x - np.asarray(kwargs["pos"])) @ np.asarray(wp.quat_to_matrix(kwargs["rot"])).reshape(3, 3)
    return x, np.asarray(b.tri_indices).reshape(-1, 3), np.flatnonzero(fixed), panel[:, :2]


def grid(nx: int, ny: int, lx: float, ly: float, rng=None, jitter: float = 0.0):
    """An ``nx x ny``-node plate in the xy plane, faces counter-clockwise; ``jitter`` moves interior nodes."""
    xs, ys = np.linspace(0.0, lx, nx), np.linspace(0.0, ly, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    v = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    if jitter:
        inner = (X.ravel() > 0) & (X.ravel() < lx) & (Y.ravel() > 0) & (Y.ravel() < ly)
        v[inner, :2] += jitter * rng.uniform(-1.0, 1.0, (inner.sum(), 2)) * [lx / (nx - 1), ly / (ny - 1)]
    f = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
            f += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    return v, np.array(f, dtype=np.int32)


def segment_distance(p0, p1, q0, q1) -> np.ndarray:
    """Closest distance between segments ``p0 p1`` and ``q0 q1`` (row-wise)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e = np.sum(d1 * d1, 1), np.sum(d2 * d2, 1)
    b, c, f = np.sum(d1 * d2, 1), np.sum(d1 * r, 1), np.sum(d2 * r, 1)
    den = a * e - b * b
    s = np.where(den > 1e-12, np.clip((b * f - c * e) / np.maximum(den, 1e-30), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p0 + s[:, None] * d1 - q0 - t[:, None] * d2, axis=1)


def close_pairs(x: np.ndarray, radius: float, exclude: int, reach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segments ``(i, j)`` of the polyline ``x`` at least ``exclude`` apart along it whose midpoints are within
    ``reach``, and their gaps in diameters (``< 1``: the rope passes into itself)."""
    mid = 0.5 * (x[1:] + x[:-1])
    pairs = cKDTree(mid).query_pairs(reach, output_type="ndarray")
    i, j = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) >= exclude].T
    return i, j, segment_distance(x[i], x[i + 1], x[j], x[j + 1]) / (2.0 * radius)


def capsule_poses(x: np.ndarray) -> np.ndarray:
    """Capsule poses (segments, 7) of a polyline of nodes: at the segments' midpoints, local z along them."""
    d = x[1:] - x[:-1]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    q = np.concatenate([np.cross([0.0, 0.0, 1.0], d), 1.0 + d[:, 2:3]], axis=1)  # (x, y, z, w): z onto d
    return np.concatenate([0.5 * (x[1:] + x[:-1]), q / np.linalg.norm(q, axis=1, keepdims=True)], axis=1)


def capsules(model, state, rods: list[list[int]]):
    """Endpoints ``(a, b)`` and radius of every capsule of every rod (each rod a list of bodies)."""
    body_q = state.body_q.numpy()
    shape_body = model.shape_body.numpy()
    half = model.shape_scale.numpy()[:, 1]
    radius = model.shape_scale.numpy()[:, 0]
    shape_of = {int(b): s for s, b in enumerate(shape_body) if b >= 0}
    out = []
    for bodies in rods:
        s = np.array([shape_of[b] for b in bodies])
        q = body_q[bodies]
        p, rot = q[:, :3], q[:, 3:]  # quaternion (x, y, z, w)
        u, w = rot[:, :3], rot[:, 3:]
        z = np.array([0.0, 0.0, 1.0])
        uz = np.cross(u, z)
        axis = z + 2.0 * w * uz + 2.0 * np.cross(u, uz)
        d = axis * half[s, None]
        out.append((p - d, p + d, radius[s]))
    return out


# -- inset plots: the theory panels drawn over the 3D view (and the viewer's image window) --------

SIM, THEORY, MUTED = "#5aaaff", "#ebebeb", "#8a8f99"


def inset_scale(size: tuple[int, int], ncols: int = 1) -> float:
    """Text and line scale of an inset of ``size`` (width, height) px: 1 for a 400 px square plot."""
    return min(size[0] / ncols, 1.25 * size[1]) / 400.0


def inset(draw, size: tuple[int, int] = (400, 400), ncols: int = 1) -> np.ndarray:
    """RGBA image of ``ncols`` plots side by side, ``size`` (width, height) px in all, on a translucent
    rounded panel.

    ``draw(axes)`` fills the axes (one, or a list for ``ncols > 1``). Labels take mathtext (LaTeX
    without a LaTeX install), in Computer Modern; sizes scale with :func:`inset_scale`.
    """
    import matplotlib  # noqa: PLC0415
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415
    from matplotlib.patches import FancyBboxPatch  # noqa: PLC0415
    from matplotlib.transforms import IdentityTransform  # noqa: PLC0415

    pt = inset_scale(size, ncols)  # 1 pt = 1 px at 72 dpi
    style = {
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
        "font.size": 18 * pt, "axes.labelsize": 21 * pt, "xtick.labelsize": 17 * pt, "ytick.labelsize": 17 * pt,
        "legend.fontsize": 16 * pt, "legend.frameon": False, "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
        "text.color": THEORY, "axes.labelcolor": THEORY, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": THEORY, "ytick.labelcolor": THEORY, "axes.edgecolor": MUTED,
        "axes.linewidth": 1.2 * pt, "xtick.major.width": 1.2 * pt, "ytick.major.width": 1.2 * pt,
        "xtick.minor.width": 0.8 * pt, "ytick.minor.width": 0.8 * pt,
        "xtick.major.size": 5 * pt, "ytick.major.size": 5 * pt, "xtick.minor.size": 3 * pt,
        "ytick.minor.size": 3 * pt, "xtick.direction": "in", "ytick.direction": "in",
        "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": "none",
        "lines.linewidth": 2.2 * pt, "lines.markersize": 4.5 * pt,
    }
    with matplotlib.rc_context(style):
        w, h, e = *size, 1.5 * pt  # the panel in pixels, inside its edge
        fig = Figure(figsize=(w / 72.0, h / 72.0), dpi=72.0)
        fig.patch.set_alpha(0.0)
        fig.patches.append(FancyBboxPatch((e, e), w - 2 * e, h - 2 * e, boxstyle=f"round,pad=0,rounding_size={12 * pt}",
                                          transform=IdentityTransform(), figure=fig, linewidth=e, zorder=-1,
                                          facecolor=(0.07, 0.08, 0.1, 0.82), edgecolor=(1.0, 1.0, 1.0, 0.14)))
        axes = fig.subplots(1, ncols, squeeze=False)[0]
        draw(axes[0] if ncols == 1 else list(axes))
        fig.tight_layout(pad=0.7, w_pad=1.5)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba()).copy()
