"""What the examples share, none of it part of the ``dismech_newton`` package.

- :class:`CableExample`: the frame loop in Newton's example format (drive, collide, step, CUDA graph replay).
- :class:`Drive`, :func:`segment_dofs`: move clamped segments along a prescribed path.
- :func:`frame_box`: point the viewer's camera at a box.
- :func:`default_frames`: the ``--num-frames`` default from a simulated duration.
- :func:`close_pairs`, :func:`segment_distance`, :func:`capsules`: geometry checks.
- :func:`inset`: the theory plots drawn in the viewer.
"""

import newton
import numpy as np
import warp as wp
from scipy.spatial import cKDTree

from dismech_newton import flatten_state


def smoothstep(t: float, t0: float, t1: float) -> float:
    s = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


# -- prescribed motion of clamped segments ------------------------------------------------


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
        q0 = np.concatenate([model.particle_q.numpy().ravel(), model.dismech.edge_q.numpy()])
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
    pipeline_options: dict = {}
    drives: tuple = ()
    control = None  # passed to every solver step

    def start(self, viewer, model, solver, radius: float, **pipeline):
        self.viewer, self.model, self.solver = viewer, model, solver
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.substeps
        self.sim_time = 0.0
        self.frame = 0
        # The rod nodes are particles, but the solver reads rigid contacts only: skip the soft ones.
        options = {"contact_matching": "latest", **self.pipeline_options, **pipeline}
        self.pipeline = newton.CollisionPipeline(model, soft_contact_max=0, verify_buffers=False,
                                                 speculative_contact_gap_max=2.0 * radius, **options)
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = model.state(), model.state()
        flatten_state(self.state_0)
        flatten_state(self.state_1)
        self.graph = None
        if viewer is not None:  # None: a simulation another example draws
            viewer.set_model(model)

    def drive(self, t0: float, t1: float) -> None:
        """Set the drives for the frame from ``t0`` to ``t1`` (host)."""

    def simulate(self):
        for k in range(self.substeps):
            for drive in self.drives:
                drive(self.state_0, (k + 1) / self.substeps)
            self.pipeline.collide(self.state_0, self.contacts, dt=2.0 * self.sim_dt)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.post_substep()

    def post_substep(self) -> None:
        """After each substep, inside the frame's graph: device work only."""

    def step(self):
        self.drive(self.sim_time, self.sim_time + self.frame_dt)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
            if self.capture and self.solver.graph_capturable:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph
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
        self.viewer.log_contacts(self.contacts, self.state_0)
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
