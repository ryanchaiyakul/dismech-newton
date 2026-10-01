"""Helpers shared by the examples and ``scripts/compare.py``."""

import newton
import numpy as np
import warp as wp
from scipy.spatial import cKDTree

from dismech_newton import flatten_state


def smoothstep(t: float, t0: float, t1: float) -> float:
    s = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


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


# -- metrics on the capsules, identical for Newton's cables (rigid bodies) and ours (proxies) --


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


def metrics(model, state, rods: list[list[int]]) -> dict[str, float]:
    """Accuracy of the capsule chain: joint gap (relative to the segment length), ground
    penetration and self-penetration (relative to the diameter)."""
    caps = capsules(model, state, rods)
    gap = max(float(np.max(np.linalg.norm(a[1:] - b[:-1], axis=1) / (2.0 * np.linalg.norm(b - a, axis=1)[1:])))
              if len(a) > 1 else 0.0 for a, b, _ in caps)
    ground = max(float(np.max(r - np.minimum(a[:, 2], b[:, 2]))) / (2.0 * r.max()) for a, b, r in caps)
    a = np.concatenate([c[0] for c in caps])
    b = np.concatenate([c[1] for c in caps])
    r = np.concatenate([c[2] for c in caps])
    rod = np.concatenate([np.full(len(c[0]), k) for k, c in enumerate(caps)])
    idx = np.concatenate([np.arange(len(c[0])) for c in caps])
    pen = 0.0
    mid = 0.5 * (a + b)
    reach = float(np.max(np.linalg.norm(b - a, axis=1)) + 2.0 * r.max())
    pairs = cKDTree(mid).query_pairs(reach, output_type="ndarray")
    if len(pairs):
        i, j = pairs.T
        keep = (rod[i] != rod[j]) | (np.abs(idx[i] - idx[j]) > 2)
        i, j = i[keep], j[keep]
        dist = segment_distance(a[i], b[i], a[j], b[j])
        pen = float(max(0.0, np.max((r[i] + r[j] - dist) / (r[i] + r[j])))) if len(i) else 0.0
    return {"joint_gap": gap, "ground_pen": max(ground, 0.0), "self_pen": pen}


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


class CableExample:
    """Frame loop of the examples, in Newton's example format.

    A subclass builds its model and solver and calls :meth:`start`; :meth:`drive` sets the drives
    for each frame. The first frame runs eagerly (the solver sets itself up), later frames replay
    one CUDA graph of the frame's substeps (an even count keeps ``state_0`` in place).
    """

    fps, substeps, capture = 60, 4, True
    pipeline_options: dict = {}
    drives: tuple = ()

    def start(self, viewer, model, solver, radius: float, **pipeline):
        self.viewer, self.model, self.solver = viewer, model, solver
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.substeps
        self.sim_time = 0.0
        # The rod nodes are particles, but the solver reads rigid contacts only: skip the soft ones.
        options = {"contact_matching": "latest", **self.pipeline_options, **pipeline}
        self.pipeline = newton.CollisionPipeline(model, soft_contact_max=0, verify_buffers=False,
                                                 speculative_contact_gap_max=2.0 * radius, **options)
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = model.state(), model.state()
        flatten_state(self.state_0)
        flatten_state(self.state_1)
        self.graph = None
        viewer.set_model(model)

    def drive(self, t0: float, t1: float) -> None:
        """Set the drives for the frame from ``t0`` to ``t1`` (host)."""

    def simulate(self):
        for k in range(self.substeps):
            for drive in self.drives:
                drive(self.state_0, (k + 1) / self.substeps)
            self.pipeline.collide(self.state_0, self.contacts, dt=2.0 * self.sim_dt)
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.drive(self.sim_time, self.sim_time + self.frame_dt)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
            if self.capture and self.model.device.is_cuda:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()
