"""Recover a rod from still photos as free 3D Gaussians, then fit its bending stiffness and damping to a video.

Shows: :mod:`dismech_newton.splat`, the rod as Gaussian splats rigid in each edge's material frame; a rod model built
from data instead of given (3D Gaussian splatting with no rod in it, the rod's length, radius and pose read off the
Gaussians, every Gaussian bound to the rod: the inverse of :func:`~dismech_newton.splat.skin`); gradients from pixels,
the image loss of a differentiable rasterizer (gswarp, in torch) handed back to Warp and taken through the skin and
every solver step by one ``tape.backward``.

A painted rod, clamped at one end, is held still in a curved pose: 24 cameras photograph it. It is released and one
fixed camera films it swing for 1 s at 30 fps. The fit knows the clamp's pose (a calibrated fixture: where, which
axis), the rod's weight, the number of segments to model it with, the calibrated cameras and its stretch and twist
stiffness. Everything else it reads off the data:

1. Free Gaussians (positions, shapes, colours, opacities) fitted to the 24 photos from a visual hull, by Adam.
2. The rod from the Gaussians. A first centreline: circles fitted to the outer boundary of the Gaussians' density
   in cross-sections (plain averages lean to where 3DGS happened to put its Gaussians). Its ends: where the density
   falls to half its level just inside them. Its radius: half the width it renders in the photos (the density's raw
   values depend on the Gaussians' layout; what renders does not). Its pose: equal chords along that centreline
   from the clamp, then fitted to the rod's outline in all 24 photos (a plain tube on the nodes, edge lengths
   fixed): the density's centreline carries millimetre kinks, the outline does not. Each Gaussian is then bound to
   its nearest edge, in that edge's material frame at twist 0: the paint's orientation comes with the Gaussians,
   so the clamp's angle is no unknown.
3. Bend stiffness and damping by L-BFGS from the video, from a guess off by 2.5-3x: the bound Gaussians, skinned to
   a rod released from the recovered pose, rendered and compared pixel by pixel. The fit simulates with
   :class:`~dismech_newton.ADMMDiSMechSolver` at its defaults (at most 50 iterations a step), restarted for every
   rollout, and takes the gradient through its steps; the video is simulated with Newton.

The start pose limits the fit: a node off by a millimetre on 5 cm edges is a kink as curved as the rod, the released
rod rings with it, and the damping absorbs that (from the density's centreline alone, EI and c came out 5% and 15%
off, varying run to run). Hence the outline fit, and photos of 512 px.

The viewer plays the three acts: the Gaussians training, the rod they give (grey: the true pose; green: the
recovered), then every L-BFGS step replayed (green) beside the truth (grey). The Gaussians, the rod and the fit are
cached in ``.cache/examples`` (``--fresh`` recomputes them; a fresh run takes ~2 min). It needs CUDA and the
``splat`` extra (torch, gswarp):

    uv run --extra splat examples/fit_capture.py
    uv run --extra splat examples/fit_capture.py --viewer null --test
"""

import math

import newton
import newton.examples
import numpy as np
import torch
import warp as wp
from PIL import Image, ImageDraw
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import minimize
from utils.common import TapedGraph, cached, capsule_poses, inset, inset_scale
from utils.gaussians import DEVICE, Camera, backprop, cov6, rotation_matrix, to_uint8, view

from dismech_newton import ADMMDiSMechSolver, DiSMechSolver, add_rod, fix_segment, flatten_state
from dismech_newton.solver import advance_frames_kernel
from dismech_newton.splat import Splats, skin, tube
from dismech_newton.triplet import advance_ref_twist_kernel

# -- the rod and the truth -----------------------------------------------------------------
LENGTH, SEGMENTS, RADIUS = 0.5, 10, 0.02
CLAMP, CLAMP_AXIS = np.array([0.0, 0.0, 1.0]), (1.0, 0.0, 0.0)  # the fixture: the clamped end and its axis (known)
TRUE = dict(EI=8.0, c=0.1)  # bend stiffness and damping (both planes)
GUESS = dict(EI=0.4, c=3.0)  # x the truth
KINDS = list(TRUE)
GJ, EA = 8.0, 1.0e4  # twist and stretch stiffness, known
BEND, DAMP = [2, 3], [7, 8]  # triplet_params columns of EI and c
TWIST = 0.8  # the held pose's uniform twist [rad]: where the paint faces, never an unknown

# -- the data: 24 photos of the held rod, then 1 s of video from one camera -------------------------
VIEWS, PHOTO, PHOTO_NOISE = 24, 512, 0.01  # cameras, px, pixel noise std
DT, EVERY, FRAMES = 1.0 / 240.0, 8, 30  # 30 fps for 1 s
STEPS = EVERY * FRAMES
IMAGE, NOISE = 256, 0.05  # the video: px, pixel noise std
VIDEO_EYE, VIDEO_LOOK_AT = (0.15, -0.95, 1.35), (0.28, 0.05, 0.95)
SEED = 0

# -- the truth's look: rings of splats on every edge, grey with a red stripe along m1 and a blue one along -m1 --
RINGS, PER_RING = 3, 8
SIGMA = (0.003, 0.008, 0.010)  # the splats' standard deviations (radial, tangential, along the edge) [m]
PAINT, RED, BLUE, BACKGROUND = (0.75, 0.75, 0.72), (0.9, 0.15, 0.1), (0.1, 0.3, 0.95), (0.05, 0.05, 0.08)
OPACITY = 0.9

# -- the free Gaussians -----------------------------------------------------------------------
POINTS, ITERATIONS, BATCH = 8000, 3000, 4  # initial Gaussians (from the visual hull), Adam steps, photos per step
LR = dict(means=3e-4, log_scale=5e-3, quat=2e-3, colour=2e-2, opacity=5e-2)
SNAPSHOTS = [0, 25, 50, 100, 200, 400, 800, 1500, 3000]  # the steps the viewer shows

# -- the rod from the Gaussians ----------------------------------------------------------------
OPAQUE = 0.2  # Gaussians at least this opaque make the rod's shape
END_WINDOW = 0.06  # [m] inside each end: the level whose half maximum is the end


def held_pose() -> np.ndarray:
    """Nodes (SEGMENTS + 1, 3) of the held pose: from the clamp along +x, turning up and sideways."""
    u = np.arange(SEGMENTS) / (SEGMENTS - 1)
    t = np.stack([np.ones(SEGMENTS), 0.9 * u, 0.6 * u], 1)  # the first edge along the clamp's axis, +x
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return CLAMP + np.concatenate([[np.zeros(3)], np.cumsum(LENGTH / SEGMENTS * t, 0)])


def build(length=LENGTH, radius=RADIUS, mass=None, admm: bool = False):
    """A rod straight at rest from the clamp along +x, its first edge clamped; ``mass``: the total (else the
    default density's). ``admm``: the ADMM solver at its defaults (else Newton)."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    rod = newton.Rod.create_straight(tuple(CLAMP), (1.0, 0.0, 0.0), length, segment_count=SEGMENTS, radius=radius)
    ids = add_rod(builder, rod, stretch_stiffness=EA, bend_stiffness=TRUE["EI"], twist_stiffness=GJ,
                  bend_damping=TRUE["c"], proxies=False)
    fix_segment(builder, edge=ids[0])
    model = builder.finalize(device=DEVICE)
    if mass is not None:
        m = model.particle_mass
        assert m is not None
        m.assign(m.numpy() * (mass / m.numpy().sum()))
    solver = ADMMDiSMechSolver(model) if admm else DiSMechSolver(model)
    solver.refresh_mass()
    return model, solver


def states(model, count: int, requires_grad: bool = False) -> list:
    out = [model.state(requires_grad=requires_grad) for _ in range(count)]
    for s in out:
        flatten_state(s)
    return out


def start(solver, rest, st, q):
    """``st`` = ``rest`` moved to the DOFs ``q``, still: the edge frames transported from rest, reference twists
    and strains measured there. The solver starts its next step from ``st`` (no ADMM warm start from the last
    rollout)."""
    for k in ("q", "qd", "edge_d1_q", "triplet_ref_twist_q", "triplet_strain_q"):
        getattr(st.dismech, k).assign(getattr(rest.dismech, k))
    st.dismech.q.assign(np.asarray(q, dtype=np.float32))
    st.dismech.qd.zero_()
    d, tr = solver.der, solver.triplets
    wp.launch(advance_frames_kernel, dim=d.edge_length.shape[0],
              inputs=[rest.particle_q, st.particle_q, d.edge_node0, d.edge_node1, rest.dismech.edge_d1_q],
              outputs=[st.dismech.edge_d1_q])
    wp.launch(advance_ref_twist_kernel, dim=tr.count,
              inputs=[st.dismech.q, st.dismech.edge_d1_q, tr.conn, rest.dismech.triplet_ref_twist_q],
              outputs=[st.dismech.triplet_ref_twist_q])
    tr.measure(st, st.dismech.triplet_strain_q)
    solver.reset(st)


def held_q(nodes, twist: float) -> np.ndarray:
    return np.concatenate([np.asarray(nodes, dtype=float).ravel(), np.full(SEGMENTS, twist)])


def photo_cameras(centre=(0.25, 0.1, 1.05), distance=0.9) -> list[Camera]:
    """VIEWS cameras on a sphere about ``centre`` (a golden spiral, elevations -30 to 70 degrees)."""
    out = []
    for k in range(VIEWS):
        el = np.radians(-30.0 + 100.0 * (k + 0.5) / VIEWS)
        az = k * np.pi * (3.0 - np.sqrt(5.0))
        eye = np.asarray(centre) + distance * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        out.append(Camera(eye, centre, PHOTO, background=BACKGROUND))
    return out


def video_camera(size: int = IMAGE) -> Camera:
    return Camera(VIDEO_EYE, VIDEO_LOOK_AT, size, background=BACKGROUND)


def paint(model):
    """The truth's look: its splats, colours and opacities."""
    splats, k = tube(model, RINGS, PER_RING, RADIUS, SIGMA)
    colours = np.tile(PAINT, (len(splats), 1))
    colours[k == 0] = RED
    colours[k == PER_RING // 2] = BLUE
    return (splats, torch.tensor(colours, dtype=torch.float32, device=DEVICE),
            torch.full((len(splats),), OPACITY, device=DEVICE))


def record(video_size: int = IMAGE) -> dict:
    """The data: the photos ``(VIEWS, 3, PHOTO, PHOTO)`` and the video ``(FRAMES + 1, 3, H, W)`` (torch, with
    noise); for checking only: the true DOFs per frame, the truth's splats at the held pose, the rod's weight."""
    model, solver = build()
    rest, *sts = states(model, STEPS + 2)
    start(solver, rest, sts[0], held_q(held_pose(), TWIST))
    splats, colours, opacity = paint(model)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)

    def shot(cam, s, noise):
        with torch.no_grad():
            img = cam(*map(view, skin(model, splats, s.dismech.q, s.dismech.edge_d1_q)), colours, opacity)
            return img + noise * torch.randn(img.shape, generator=gen, device=DEVICE)

    photos = torch.stack([shot(cam, sts[0], PHOTO_NOISE) for cam in photo_cameras()])
    held = [a.numpy() for a in skin(model, splats, sts[0].dismech.q, sts[0].dismech.edge_d1_q)]
    cam = video_camera(video_size)
    video, q = [], []
    for i in range(STEPS + 1):
        if i % EVERY == 0:
            video.append(shot(cam, sts[i], NOISE))
            q.append(sts[i].dismech.q.numpy())
        if i < STEPS:
            solver.step(sts[i], sts[i + 1], None, None, DT)
    return dict(photos=photos, video=torch.stack(video), q=np.array(q), held_means=held[0], held_cov6=held[1],
                mass=float(model.particle_mass.numpy().sum()))


# -- 1. free Gaussians from the photos --------------------------------------------------------------


def visual_hull(cams, photos, rng, count) -> np.ndarray:
    """``count`` random points inside every photo's foreground."""
    bg = torch.tensor(BACKGROUND, device=DEVICE)[:, None, None]
    masks = [((p - bg).abs().sum(0) > 0.15).cpu().numpy() for p in photos]
    lo, hi = np.array([-0.2, -0.4, 0.6]), np.array([0.8, 0.6, 1.5])
    keep = []
    while sum(len(k) for k in keep) < count:
        x = lo + (hi - lo) * rng.random((200000, 3))
        inside = np.ones(len(x), bool)
        for cam, mask in zip(cams, masks):
            px = np.round(cam.project(x)).astype(int)
            ok = (px >= 0).all(1) & (px < cam.size).all(1)
            hit = np.zeros(len(x), bool)
            hit[ok] = mask[px[ok, 1], px[ok, 0]]
            inside &= hit
        keep.append(x[inside])
    return np.concatenate(keep)[:count]


def train(photos) -> dict:
    """Free Gaussians: Adam on positions, log scales, quaternions, colour and opacity logits; L1 on BATCH random
    photos per step; those fading below 1% opacity pruned. Snapshots of photo 0 and the PSNR at SNAPSHOTS."""
    cams = photo_cameras()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    x = visual_hull(cams, photos, rng, POINTS)
    P = {"means": torch.tensor(x, dtype=torch.float32, device=DEVICE),
         "log_scale": torch.full((len(x), 3), math.log(0.004), device=DEVICE),
         "quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE).repeat(len(x), 1),
         "colour": torch.zeros(len(x), 3, device=DEVICE),  # logits: grey
         "opacity": torch.zeros(len(x), device=DEVICE)}  # logit: half opaque
    for p in P.values():
        p.requires_grad_()
    opt = torch.optim.Adam([{"params": [P[k]], "lr": LR[k], "name": k} for k in P], eps=1e-15)

    def render(cam):
        return cam(P["means"], cov6(P["log_scale"], P["quat"]), torch.sigmoid(P["colour"]), torch.sigmoid(P["opacity"]))

    snaps, history = [], []
    for it in range(ITERATIONS + 1):
        if it in SNAPSHOTS:
            with torch.no_grad():
                snaps.append(to_uint8(render(cams[0])))
                mse = np.mean([float(((render(c) - p) ** 2).mean()) for c, p in zip(cams, photos)])
            history.append((it, -10 * math.log10(mse), len(P["means"])))
            print(f"  Gaussians, step {it:4d}: PSNR {history[-1][1]:.2f} dB, {len(P['means'])} Gaussians", flush=True)
        if it == ITERATIONS:
            break
        opt.zero_grad()
        loss = sum((render(cams[k]) - photos[k]).abs().mean() for k in rng.choice(VIEWS, BATCH, replace=False))
        loss.backward()
        opt.step()
        if it in (500, 1500, 2500):
            keep = torch.sigmoid(P["opacity"]) > 0.01
            for g in opt.param_groups:
                old = g["params"][0]
                new = old.detach()[keep].clone().requires_grad_()
                st = opt.state.pop(old)
                st["exp_avg"], st["exp_avg_sq"] = st["exp_avg"][keep], st["exp_avg_sq"][keep]
                opt.state[new] = st
                g["params"][0] = new
                P[g["name"]] = new
    out = {k: v.detach().cpu().numpy() for k, v in P.items()}
    return dict(out, history=np.array(history), snaps=np.array(snaps))


# -- 2. the rod from the Gaussians --------------------------------------------------------------


class Curve:
    """A dense polyline with its arc length, extended straight past both ends."""

    def __init__(self, pts, extend=0.08):
        t0, t1 = pts[1] - pts[0], pts[-1] - pts[-2]
        t0, t1 = t0 / np.linalg.norm(t0), t1 / np.linalg.norm(t1)
        ext = np.linspace(extend, 0, 40, endpoint=False)[:, None]
        self.pts = np.concatenate([pts[0] - ext * t0, pts, pts[-1] + ext[::-1] * t1])
        self.s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(self.pts, axis=0), axis=1))])

    def at(self, s):
        return np.stack([np.interp(s, self.s, self.pts[:, i]) for i in range(3)], -1)

    def tangent(self, s, h=1e-3):
        t = self.at(s + h) - self.at(s - h)
        return t / np.linalg.norm(t, axis=-1, keepdims=True)

    def project(self, p, chunk=20000) -> np.ndarray:
        """Arc coordinates of points ``(N, 3)``."""
        a, ab = (torch.tensor(v, device=DEVICE) for v in (self.pts[:-1], np.diff(self.pts, axis=0)))
        s0 = torch.tensor(self.s[:-1], device=DEVICE)
        out = []
        for k in range(0, len(p), chunk):
            q = torch.tensor(p[k : k + chunk], device=DEVICE)
            u = (((q[:, None] - a[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1)[None]).clamp(0, 1)
            j = (a[None] + u[..., None] * ab[None] - q[:, None]).norm(dim=-1).argmin(1)
            out.append((s0[j] + u[torch.arange(len(j)), j] * ab[j].norm(dim=-1)).cpu().numpy())
        return np.concatenate(out)


def sections(curve, s):
    """Unit vectors across the curve at ``s``: ``(e1, e2)``, each ``(len(s), 3)``."""
    t = curve.tangent(s)
    e1 = np.cross(t, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(t, e1)


def density(g, x, chunk=4096) -> np.ndarray:
    """The Gaussians' density, sum of opacity x exp(-d^2 / 2) (d: the Mahalanobis distance), at points ``x``."""
    M, IC, O = (torch.tensor(v, dtype=torch.float32, device=DEVICE) for v in (g["means"], g["icov"], g["opacity"]))
    flat = torch.tensor(x.reshape(-1, 3), dtype=torch.float32, device=DEVICE)
    out = []
    for k in range(0, len(flat), chunk):
        d = flat[k : k + chunk, None] - M[None]
        out.append((O[None] * torch.exp(-0.5 * torch.einsum("pgi,gij,pgj->pg", d, IC, d))).sum(1))
    return torch.cat(out).cpu().numpy().reshape(x.shape[:-1])


def rough_centreline(g, rng, bins=40) -> Curve:
    """Opacity-weighted means of points sampled from the Gaussians, in bins along the rod (two passes)."""
    L = np.linalg.cholesky(g["cov"] + 1e-12 * np.eye(3))
    p = (g["means"][:, None] + np.einsum("gij,gsj->gsi", L, rng.standard_normal((len(L), 32, 3)))).reshape(-1, 3)
    w = np.repeat(g["opacity"], 32)
    c = np.average(p, 0, w)
    u = (p - c) @ np.linalg.svd((p - c) * np.sqrt(w)[:, None], full_matrices=False)[2][0]
    for _ in range(2):
        edges = np.quantile(u, np.linspace(0.01, 0.99, bins + 1))
        idx = np.clip(np.searchsorted(edges, u) - 1, 0, bins - 1)
        mids = np.array([np.average(p[idx == b], 0, w[idx == b]) for b in range(bins)])
        curve = spline(mids, lam=1e-4)
        u = curve.project(p)
    return curve


def spline(points, lam=None) -> Curve:
    """A smoothing spline through ``points`` in their chord length (``lam=None``: generalised cross-validation)."""
    t = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    dense = np.linspace(0, t[-1], 400)
    return Curve(np.stack([make_smoothing_spline(t, points[:, i], lam=lam)(dense) for i in range(3)], -1))


def ends(curve, g) -> tuple[float, float]:
    """Arc coordinates of the rod's ends: where the density's maximum over a cross-section falls to half its level
    just inside the end (the density is uneven along the rod: 3DGS stacks Gaussians where the render saturates)."""
    s = np.arange(curve.s[0], curve.s[-1], 0.001)
    e1, e2 = sections(curve, s)
    r = np.arange(0.0, 0.045, 0.0005)[None, :, None, None]
    phi = np.linspace(0, 2 * np.pi, 24, endpoint=False)[None, None, :, None]
    disc = r * (np.cos(phi) * e1[:, None, None] + np.sin(phi) * e2[:, None, None])
    h = density(g, curve.at(s)[:, None, None] + disc).max(axis=(1, 2))
    mid = (s > s[0] + 0.25 * (s[-1] - s[0])) & (s < s[-1] - 0.25 * (s[-1] - s[0]))
    above = np.flatnonzero(h >= 0.1 * np.median(h[mid]))
    out = []
    for i, inward in ((above[0], 1), (above[-1], -1)):
        inside = (s - s[i]) * inward
        level = np.median(h[(inside > 0.02) & (inside < END_WINDOW)])
        j = i
        while h[j] < 0.5 * level:
            j += inward
        out.append(float(np.interp(0.5 * level, [h[j - inward], h[j]], [s[j - inward], s[j]])))
    return out[0], out[1]


def circle_centres(curve, g, s0, s1, step=0.01, rays=72) -> np.ndarray:
    """Centres of circles fitted to the rod's outer boundary in cross-sections every ``step``: along each ray from
    the section's centre, the outermost point where the density is half the ray's maximum (two passes)."""
    s = np.arange(s0 + 0.004, s1 - 0.003, step)
    c = curve.at(s)
    e1, e2 = sections(curve, s)
    r = np.arange(0.0, 0.045, 0.0003)
    phi = np.linspace(0, 2 * np.pi, rays, endpoint=False)
    dirs = np.cos(phi)[None, :, None] * e1[:, None] + np.sin(phi)[None, :, None] * e2[:, None]
    for _ in range(2):
        D = density(g, c[:, None, None] + r[None, None, :, None] * dirs[:, :, None])  # (sections, rays, r)
        half = 0.5 * D.max(2, keepdims=True)
        m = D.shape[2] - 1 - np.argmax((D >= half)[..., ::-1], axis=2)  # the outermost crossing
        m = np.minimum(m, D.shape[2] - 2)
        Dm, Dn = np.take_along_axis(D, m[..., None], 2)[..., 0], np.take_along_axis(D, m[..., None] + 1, 2)[..., 0]
        edge = r[m] + (Dm - half[..., 0]) / np.maximum(Dm - Dn, 1e-12) * (r[1] - r[0])
        x, y = edge * np.cos(phi), edge * np.sin(phi)
        A = np.stack([x, y, np.ones_like(x)], -1)  # Kasa: x^2 + y^2 = 2 a x + 2 b y + const
        sol = np.stack([np.linalg.lstsq(A[i], x[i] ** 2 + y[i] ** 2, rcond=None)[0] for i in range(len(s))])
        c = c + 0.5 * sol[:, :1] * e1 + 0.5 * sol[:, 1:2] * e2
    return c


def silhouette_radius(curve, s0, s1, means, cov, opacity, step=0.01) -> float:
    """Half the width the rod renders in the photos: across the projected centreline (middle 80%, every ``step``),
    the half-maximum width of the opacity image, in metres at that depth; the median."""
    i, j = np.triu_indices(3)
    M, C, O = (torch.tensor(v, dtype=torch.float32, device=DEVICE) for v in (means, cov[:, i, j], opacity))
    white = torch.ones(len(means), 3, device=DEVICE)
    s = np.arange(s0 + 0.1 * (s1 - s0), s1 - 0.1 * (s1 - s0), step)
    c, ahead = curve.at(s), curve.at(s + 1e-3)
    u = np.arange(-80, 80.25, 0.25)
    out = []
    for cam in photo_cameras():
        black = Camera(cam.eye, cam.eye + np.linalg.inv(cam.V)[:3, 2], cam.size, background=(0.0, 0.0, 0.0))
        with torch.no_grad():
            alpha = black(M, C, white, O)[0].cpu().numpy()
        f = 0.5 * cam.size / math.tan(cam.fov / 2)
        p, depth = cam.project(c), cam.depth(c)
        tan = cam.project(ahead) - p
        n = np.stack([-tan[:, 1], tan[:, 0]], 1) / np.linalg.norm(tan, axis=1, keepdims=True)
        for k in np.flatnonzero(np.linalg.norm(tan, axis=1) > 0.7e-3 * f / depth):  # not foreshortened
            xy = p[k] + u[:, None] * n[k]
            if (xy < 1).any() or (xy > cam.size - 2).any():
                continue
            x0, y0 = np.floor(xy).astype(int).T
            fx, fy = xy[:, 0] - x0, xy[:, 1] - y0
            prof = ((1 - fx) * (1 - fy) * alpha[y0, x0] + fx * (1 - fy) * alpha[y0, x0 + 1]
                    + (1 - fx) * fy * alpha[y0 + 1, x0] + fx * fy * alpha[y0 + 1, x0 + 1])
            half = 0.5 * prof.max()
            above = np.flatnonzero(prof >= half)
            a, b = above[0], above[-1]
            if 0 < a and b < len(u) - 1:
                lo = np.interp(half, [prof[a - 1], prof[a]], [u[a - 1], u[a]])
                hi = np.interp(half, [prof[b + 1], prof[b]], [u[b + 1], u[b]])
                out.append(0.5 * (hi - lo) * depth[k] / f)
    return float(np.median(out))


def nodes_along(curve, s_start, direction, length) -> np.ndarray:
    """SEGMENTS + 1 nodes from ``s_start`` (direction +-1), consecutive chords ``length / SEGMENTS``."""
    l = length / SEGMENTS
    out, s = [curve.at(s_start)], s_start
    for _ in range(SEGMENTS):
        lo, hi = s, s + direction * 2 * l
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if np.linalg.norm(curve.at(mid) - out[-1]) < l else (lo, mid)
        s = 0.5 * (lo + hi)
        out.append(curve.at(s))
    return np.array(out)


def refine_pose(gaussians: dict, nodes, length, radius):
    """The held pose fitted to the rod's outline in every photo: a plain tube of splats (its width a free scale)
    skinned on the nodes, against the learned Gaussians' opacity image in each photo; L-BFGS on the free edges'
    directions (lengths fixed, the clamped edge from the fixture). The density's centreline carries millimetre
    kinks; the outline in 24 views does not."""
    g = free(gaussians, 0.01)
    i, j = np.triu_indices(3)
    M, C, O = (torch.tensor(v, dtype=torch.float32, device=DEVICE) for v in (g["means"], g["cov"][:, i, j], g["opacity"]))
    cams = [Camera(c.eye, c.eye + np.linalg.inv(c.V)[:3, 2], c.size, background=(0.0, 0.0, 0.0))
            for c in photo_cameras()]
    with torch.no_grad():
        targets = [cam(M, C, torch.ones(len(M), 3, device=DEVICE), O)[0] for cam in cams]
    model, solver = build(length, radius)
    rest, st = states(model, 2)
    l = length / SEGMENTS
    plain, _ = tube(model, 3, 12, 0.6 * radius, (0.45 * radius, 0.35 * radius, 0.4 * l), requires_grad=True)
    white = torch.ones(len(plain), 3, device=DEVICE)
    opaque = torch.full((len(plain),), 0.95, device=DEVICE)
    uv0, ls0 = plain.uv.numpy().copy(), plain.log_scale.numpy().copy()
    clamp = np.array([CLAMP, CLAMP + l * np.asarray(CLAMP_AXIS)])
    v0 = np.diff(np.concatenate([clamp[1:], nodes[2:]]), axis=0)
    x0 = np.concatenate([(v0 / np.linalg.norm(v0, axis=1, keepdims=True)).ravel(), [0.0]])
    q = wp.zeros(3 * (SEGMENTS + 1) + SEGMENTS, dtype=float, requires_grad=True)
    d1 = wp.zeros(SEGMENTS, dtype=wp.vec3)
    scale = np.concatenate([np.full(len(x0) - 1, 0.02), [0.1]])  # the optimiser's units: direction, log width

    def pose(x):
        v = x[:-1].reshape(-1, 3)
        u = v / np.linalg.norm(v, axis=1, keepdims=True)
        return np.concatenate([clamp, clamp[1] + l * np.cumsum(u, 0)]), u, np.linalg.norm(v, axis=1, keepdims=True)

    def fun(z):
        x = x0 + scale * z
        n, u, nv = pose(x)
        plain.uv.assign(uv0 * math.exp(x[-1]))
        ls = ls0.copy()
        ls[:, :2] += x[-1]
        plain.log_scale.assign(ls)
        for a in (plain.uv.grad, plain.log_scale.grad, q.grad):
            a.zero_()
        start(solver, rest, st, held_q(n, 0.0))  # the frames: data here (the tube is round)
        d1.assign(st.dismech.edge_d1_q)
        q.assign(st.dismech.q)
        tape = wp.Tape()
        with tape:
            frame = skin(model, plain, q, d1)
        loss = sum(backprop(frame, lambda m, c, cam=cam, t=t: ((cam(m, c, white, opaque)[0] - t) ** 2).sum())
                   for cam, t in zip(cams, targets))
        tape.backward()
        gn = q.grad.numpy()[: 3 * (SEGMENTS + 1)].reshape(-1, 3).astype(np.float64)
        later = np.cumsum(gn[2:][::-1], 0)[::-1]  # an edge's direction moves every node after it
        gv = l * (later - np.sum(later * u, 1, keepdims=True) * u) / nv
        gw = np.sum(plain.uv.grad.numpy() * plain.uv.numpy()) + np.sum(plain.log_scale.grad.numpy()[:, :2])
        return loss, np.concatenate([gv.ravel(), [gw]]) * scale

    res = minimize(fun, np.zeros_like(x0), jac=True, method="L-BFGS-B", options=dict(maxiter=300, ftol=1e-12))
    return pose(x0 + scale * res.x)[0]


def free(gaussians: dict, opaque: float = 0.0) -> dict:
    """The trained Gaussians' world values (opacity above ``opaque``)."""
    opacity = 1 / (1 + np.exp(-gaussians["opacity"]))
    R = rotation_matrix(gaussians["quat"])
    cov = (R * np.exp(2 * gaussians["log_scale"])[:, None, :]) @ R.transpose(0, 2, 1)
    k = opacity > opaque
    return dict(means=gaussians["means"][k].astype(np.float64), cov=cov[k], icov=np.linalg.inv(cov[k]),
                opacity=opacity[k], R=R[k], log_scale=gaussians["log_scale"][k],
                colour=1 / (1 + np.exp(-gaussians["colour"][k])))


def recover(gaussians: dict, mass: float) -> dict:
    """The rod (length, radius, held pose) from the Gaussians, and each Gaussian bound to an edge."""
    rng = np.random.default_rng(SEED)
    g = free(gaussians, OPAQUE)
    curve = rough_centreline(g, rng)
    s0, s1 = ends(curve, g)
    curve = spline(circle_centres(curve, g, s0, s1))
    s0, s1 = ends(curve, g)
    length = s1 - s0
    radius = silhouette_radius(curve, s0, s1, g["means"], g["cov"], g["opacity"])
    tips = curve.at(np.array([s0, s1]))
    first = np.linalg.norm(tips[0] - CLAMP) < np.linalg.norm(tips[1] - CLAMP)
    nodes = refine_pose(gaussians, nodes_along(curve, s0 if first else s1, 1 if first else -1, length), length,
                        radius)

    # bind: each Gaussian to its nearest edge, in that edge's material frame at the held pose, twist 0
    model, solver = build(length, radius, mass)
    rest, st = states(model, 2)
    start(solver, rest, st, held_q(nodes, 0.0))
    d1 = st.dismech.edge_d1_q.numpy().astype(np.float64)
    b = free(gaussians, 0.01)
    a, ab = nodes[:-1], np.diff(nodes, axis=0)
    u = np.clip(np.einsum("gei,ei->ge", b["means"][:, None] - a[None], ab) / np.sum(ab * ab, 1), 0, 1)
    edge = np.linalg.norm(a[None] + u[..., None] * ab[None] - b["means"][:, None], axis=2).argmin(1)
    l = np.linalg.norm(ab, axis=1)
    t = ab / l[:, None]
    frame = np.stack([d1, np.cross(t, d1), t], 2)[edge]  # columns (m1, m2, t)
    rel = np.einsum("gji,gj->gi", frame, b["means"] - a[edge])
    return dict(length=length, radius=radius, nodes=nodes, curve=curve.pts, edge=edge, s=rel[:, 2] / l[edge],
                uv=rel[:, :2], rotation=frame.transpose(0, 2, 1) @ b["R"], log_scale=b["log_scale"],
                colour=b["colour"], opacity=b["opacity"])


# -- 3. physics from the video ---------------------------------------------------------------


class VideoFit:
    """``self(theta)``: the squared pixel error (averaged over the frames) and its gradient in
    ``theta = log (EI, c) / truth``, the bound Gaussians skinned to a rod released from the recovered pose."""

    def __init__(self, rod: dict, mass: float, video):
        self.model, self.solver = build(float(rod["length"]), float(rod["radius"]), mass, admm=True)
        d = self.model.dismech
        self.params = d.triplet_params.numpy().copy()
        d.triplet_params.requires_grad = True
        self.states = states(self.model, STEPS + 1, requires_grad=True)
        self.rest = states(self.model, 1)[0]
        self.q0 = held_q(rod["nodes"], 0.0)
        self.splats = Splats.from_numpy(rod["edge"], rod["s"], rod["uv"], rod["rotation"], rod["log_scale"],
                                        device=DEVICE)
        self.colours, self.opacity = (torch.tensor(rod[k], dtype=torch.float32, device=DEVICE)
                                      for k in ("colour", "opacity"))
        self.camera = video_camera()
        self.video = video
        self.taped = TapedGraph(self.run, self.solver)

    def render(self, means, cov, camera=None):
        return (camera or self.camera)(means, cov, self.colours, self.opacity)

    def set(self, theta):
        p = self.params.copy()
        p[:, BEND] *= math.exp(theta[0])
        p[:, DAMP] *= math.exp(theta[1])
        self.model.dismech.triplet_params.assign(p)
        start(self.solver, self.rest, self.states[0], self.q0)

    def run(self) -> list:
        frames = []
        for i in range(STEPS + 1):
            if i % EVERY == 0:
                s = self.states[i].dismech
                frames.append(skin(self.model, self.splats, s.q, s.edge_d1_q))
            if i < STEPS:
                self.solver.step(self.states[i], self.states[i + 1], None, None, DT)
        return frames

    def __call__(self, theta) -> tuple[float, np.ndarray]:
        """The taped rollout and its backward replay as CUDA graphs after the first call (:class:`TapedGraph`; they
        hold the solver's factorization, which :meth:`set` leaves alone); the rendering (gswarp, sizes that change
        per frame) runs eagerly between them."""
        self.set(theta)
        self.model.dismech.triplet_params.grad.zero_()
        frames = self.taped.forward()
        n = len(frames)
        loss = sum(backprop(f, lambda m, c, t=t: ((self.render(m, c) - t) ** 2).sum() / n)
                   for f, t in zip(frames, self.video))
        self.taped.backward()
        d = self.model.dismech
        p, gp = d.triplet_params.numpy(), d.triplet_params.grad.numpy()
        return loss, np.array([np.sum(gp[:, BEND] * p[:, BEND]), np.sum(gp[:, DAMP] * p[:, DAMP])])

    def replay(self, theta) -> tuple[np.ndarray, np.ndarray]:
        """``q`` and ``edge_d1_q`` at every recorded frame."""
        self.set(theta)
        self.taped.forward()
        recorded = self.states[::EVERY]
        return (np.stack([s.dismech.q.numpy() for s in recorded]),
                np.stack([s.dismech.edge_d1_q.numpy() for s in recorded]))


def fit(f: VideoFit) -> dict:
    """L-BFGS from the guess: every accepted iterate, its loss and replay."""
    theta0 = np.log([GUESS[k] for k in KINDS])
    iterates, losses, best = [theta0], [f(theta0)[0]], {}

    def fun(theta):
        loss, g = f(theta)
        if np.isfinite(loss) and np.all(np.isfinite(g)):
            if not best or loss < best["loss"]:
                best.update(loss=loss, theta=theta.copy())
            return loss, g
        # a far step blew the solver up: a smooth wall back to the best point, so the line search backtracks
        dx = theta - best["theta"]
        return best["loss"] + 1.0 + 100.0 * dx @ dx, 200.0 * dx

    def accepted(intermediate_result):  # scipy passes the OptimizeResult to a parameter of this name
        r = intermediate_result
        iterates.append(r.x.copy())
        losses.append(r.fun)
        print(f"  step {len(iterates) - 1}: loss {r.fun:.2f}  "
              + "  ".join(f"{k} {v:.4f}" for k, v in zip(KINDS, np.exp(r.x))), flush=True)

    minimize(fun, theta0, jac=True, method="L-BFGS-B", callback=accepted, bounds=[(-math.log(30), math.log(30))] * 2,
             options=dict(maxiter=50, ftol=1e-7))
    q, d1 = zip(*(f.replay(t) for t in iterates))
    return dict(theta=np.array(iterates), loss=np.array(losses), loss_true=f(np.zeros(2))[0], q=np.array(q),
                d1=np.array(d1))


def results(data: dict, fresh: bool = False) -> dict:
    """The Gaussians, the rod and the fit, each computed once per configuration (``.cache/examples``)."""
    scene = dict(version=1, rod=[LENGTH, SEGMENTS, RADIUS, list(CLAMP), GJ, EA, TWIST], true=TRUE,
                 photos=[VIEWS, PHOTO, PHOTO_NOISE], video=[DT, EVERY, FRAMES, IMAGE, NOISE, VIDEO_EYE, VIDEO_LOOK_AT],
                 look=[RINGS, PER_RING, SIGMA, PAINT, RED, BLUE, BACKGROUND, OPACITY], seed=SEED)
    gs = dict(scene=scene, train=[POINTS, ITERATIONS, BATCH, LR, SNAPSHOTS], version=1)
    gaussians = cached("fit_capture-gaussians", gs, lambda: train(data["photos"]), fresh)
    rod_config = dict(gaussians=gs, rod=[OPAQUE, END_WINDOW, list(CLAMP_AXIS)], version=2)
    rod = cached("fit_capture-rod", rod_config, lambda: recover(gaussians, data["mass"]), fresh)
    out = cached("fit_capture-fit", dict(rod=rod_config, guess=GUESS, version=2),
                 lambda: fit(VideoFit(rod, data["mass"], data["video"])), fresh)
    return dict(gaussians=gaussians, rod=rod, **out)


# -- the viewer --------------------------------------------------------------------------------

GREY, GREEN = (0.82, 0.82, 0.82), (0.3, 0.8, 0.5)
BLUE_LINE, PURPLE = "#5aaaff", "#c08cff"


def psnr_plot(ax, history, k, scale):
    it = np.maximum(history[:, 0], 1)
    ax.plot(it, history[:, 1], color=BLUE_LINE, alpha=0.25)
    ax.plot(it[: k + 1], history[: k + 1, 1], color=BLUE_LINE, label=f"PSNR {history[k, 1]:.1f} dB")
    ax.plot(it[k], history[k, 1], "o", color=BLUE_LINE, ms=7 * scale, mec="white", mew=1.2 * scale)
    ax.axhline(-10 * math.log10(PHOTO_NOISE**2), color="white", lw=scale, ls=":", alpha=0.6)
    ax.set(xscale="log", xlim=(1, it[-1]), ylim=(24, 42))
    ax.set_xlabel("free Gaussians: Adam step")
    ax.set_ylabel("PSNR [dB]")
    ax.legend(loc="lower right")


def rod_text(ax, lines, scale):
    ax.set_axis_off()
    ax.text(0.03, 0.9, "the rod, from the Gaussians", color="white", fontsize=22 * scale, transform=ax.transAxes,
            va="top")
    for i, line in enumerate(lines):
        ax.text(0.05, 0.68 - 0.17 * i, line, color="#ebebeb", fontsize=18 * scale, transform=ax.transAxes, va="top",
                family="monospace")


def convergence(ax, ratios, k, scale):
    it = np.arange(len(ratios))
    ax.axhline(1.0, color="white", lw=scale, ls=":", alpha=0.6)
    for j, (name, colour) in enumerate(((r"$EI$", BLUE_LINE), (r"$c$", PURPLE))):
        ax.plot(it, ratios[:, j], color=colour, alpha=0.25)
        ax.plot(it[: k + 1], ratios[: k + 1, j], color=colour, label=f"{name}  {ratios[k, j]:.3f}")
        ax.plot(k, ratios[k, j], "o", color=colour, ms=7 * scale, mec="white", mew=1.2 * scale)
    ax.set(yscale="log", ylim=(0.025, 8.0), xlim=(0, len(ratios) - 1))
    ax.set_yticks([1 / 30, 0.1, 0.3, 1, 3], labels=["1/30", "0.1", "0.3", "1", "3"])
    ax.minorticks_off()
    ax.set_xlabel("L-BFGS step (one camera, 1 s)")
    ax.set_ylabel("fit / truth")
    ax.legend(loc="upper right")


class Example:
    """Act 1: the Gaussians training (photo 0 beside the Gaussians' render). Act 2: the rod they give, the
    Gaussians coloured by edge with the recovered nodes. Act 3: each L-BFGS step replays the video."""

    hold, act2, pause = 10, 60, 20  # frames per snapshot, act 2's frames, frames held after each replay

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.data = record()
        r = results(self.data, bool(args is not None and args.fresh))
        self.r, self.rod = r, r["rod"]
        self.ratios = np.exp(r["theta"])
        self.fit = VideoFit(self.rod, self.data["mass"], self.data["video"])
        true_nodes = self.data["q"][0, : 3 * (SEGMENTS + 1)].reshape(-1, 3)
        self.pose_error = np.sqrt(np.mean(np.sum((self.rod["nodes"] - true_nodes) ** 2, 1)))
        g = free(r["gaussians"], OPAQUE)
        curve = Curve(self.rod["curve"])
        s = curve.project(self.rod["nodes"][[0, -1]])
        i, j = np.triu_indices(3)
        held_cov = np.zeros((len(self.data["held_cov6"]), 3, 3))
        held_cov[:, i, j] = held_cov[:, j, i] = self.data["held_cov6"]
        # the truth's own splats through the same estimator: the radius they render
        self.radius_truth = silhouette_radius(curve, s.min(), s.max(), self.data["held_means"], held_cov,
                                              np.full(len(held_cov), OPACITY))
        print(f"Gaussians: PSNR {r['gaussians']['history'][-1, 1]:.2f} dB over the {VIEWS} photos "
              f"({len(g['means'])} at least {OPAQUE:.0%} opaque)")
        print(f"rod: length {1e3 * float(self.rod['length']):.1f} mm (truth {1e3 * LENGTH:.0f}), radius "
              f"{1e3 * float(self.rod['radius']):.1f} mm (the truth's splats render {1e3 * self.radius_truth:.1f}), "
              f"held pose {1e3 * self.pose_error:.2f} mm rms")
        for k, ratio in enumerate(self.ratios):
            print(f"step {k:2d}: loss {r['loss'][k]:8.2f}  " + "  ".join(f"{u} {v:.4f}" for u, v in zip(KINDS, ratio)))
        print(f"loss at the true EI and c {float(r['loss_true']):.2f} (the pixel noise alone "
              f"~{3 * IMAGE**2 * NOISE**2:.0f})")

        self.photo = to_uint8(self.data["photos"][0])
        self.recording = [to_uint8(t) for t in self.data["video"]]
        self.truth_q = self.data["q"]
        self.true_nodes = true_nodes
        self.q_wp = wp.zeros_like(self.fit.rest.dismech.q)
        self.d1_wp = wp.zeros_like(self.fit.rest.dismech.edge_d1_q)
        start(self.fit.solver, self.fit.rest, self.fit.states[0], self.fit.q0)
        s0 = self.fit.states[0].dismech
        means, cov = (view(a).clone() for a in skin(self.fit.model, self.fit.splats, s0.q, s0.edge_d1_q))
        import matplotlib  # noqa: PLC0415

        cmap = matplotlib.colormaps["viridis"]
        by_edge = torch.tensor(cmap(0.1 + 0.8 * self.rod["edge"] / (SEGMENTS - 1))[:, :3], dtype=torch.float32,
                               device=DEVICE)
        cam = photo_cameras()[0]
        with torch.no_grad():
            own = to_uint8(self.fit.render(means, cov, cam))
            edges = Image.fromarray(to_uint8(cam(means, cov, by_edge, self.fit.opacity * 0.35)))
        d = ImageDraw.Draw(edges)
        px = cam.project(self.rod["nodes"])
        d.line([tuple(p) for p in px], fill=(255, 255, 255), width=2)
        for p in px:
            d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=(255, 255, 255), outline=(20, 20, 20))
        self.bound_image = np.concatenate([own, np.asarray(edges)], 1)
        self.rod_lines = [f"length {1e3 * float(self.rod['length']):.1f} mm   (truth {1e3 * LENGTH:.0f})",
                          f"radius {1e3 * float(self.rod['radius']):.1f} mm   (truth renders {1e3 * self.radius_truth:.1f})",
                          f"held pose {1e3 * self.pose_error:.1f} mm rms over {SEGMENTS + 1} nodes",
                          f"{len(self.rod['edge'])} Gaussians bound to {SEGMENTS} edges"]

        builder = newton.ModelBuilder()
        for color in (GREY, GREEN):
            rod = newton.Rod.create_straight(tuple(CLAMP), (1.0, 0.0, 0.0), LENGTH, segment_count=SEGMENTS,
                                             radius=RADIUS)
            add_rod(builder, rod, bend_stiffness=1.0, color=color)
        self.model = builder.finalize()
        self.state = self.model.state()
        assert self.state.body_q is not None
        self.body_q = self.state.body_q
        self.frame, self.sim_time = 0, 0.0
        self.act1 = len(SNAPSHOTS) * self.hold
        self.act3 = FRAMES + 1 + self.pause
        viewer.set_model(self.model)
        viewer.set_camera(pos=wp.vec3(0.25, -1.2, 1.15), pitch=-5.0, yaw=90.0)

    def where(self):
        """(act, index within it, frame of the video)."""
        f = self.frame % (self.act1 + self.act2 + len(self.ratios) * self.act3)
        if f < self.act1:
            return 1, f // self.hold, 0
        f -= self.act1
        if f < self.act2:
            return 2, 0, 0
        f -= self.act2
        return 3, f // self.act3, min(f % self.act3, FRAMES)

    def step(self):
        self.frame += 1
        self.sim_time += EVERY * DT

    def render(self):
        act, k, t = self.where()
        size = (500, 300)
        s = inset_scale(size)
        if act == 1:
            self.viewer.log_image("images", np.concatenate([self.photo, self.r["gaussians"]["snaps"][k]], 1))
            self.viewer.log_image("fit", inset(lambda ax: psnr_plot(ax, self.r["gaussians"]["history"], k, s), size))
            x = [self.true_nodes, self.true_nodes]
        elif act == 2:
            self.viewer.log_image("images", self.bound_image)
            self.viewer.log_image("fit", inset(lambda ax: rod_text(ax, self.rod_lines, s), size))
            x = [self.true_nodes, self.rod["nodes"]]
        else:
            self.q_wp.assign(self.r["q"][k, t])
            self.d1_wp.assign(self.r["d1"][k, t])
            with torch.no_grad():
                step = to_uint8(self.fit.render(*map(view, skin(self.fit.model, self.fit.splats, self.q_wp,
                                                                 self.d1_wp))))
            self.viewer.log_image("images", np.concatenate([self.recording[t], step], 1))
            self.viewer.log_image("fit", inset(lambda ax: convergence(ax, self.ratios, k, s), size))
            x = [self.truth_q[t, : 3 * (SEGMENTS + 1)].reshape(-1, 3),
                 self.r["q"][k, t, : 3 * (SEGMENTS + 1)].reshape(-1, 3)]
        aside = np.array([0.0, -0.06, 0.0])  # the recovered / fitted rod just in front of the truth
        self.body_q.assign(np.concatenate([capsule_poses(x[0]), capsule_poses(x[1] + aside)]).astype(np.float32))
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def test_final(self):
        assert abs(float(self.rod["length"]) / LENGTH - 1.0) < 0.01, f"length {float(self.rod['length']):.4f} m"
        assert abs(float(self.rod["radius"]) / self.radius_truth - 1.0) < 0.05, f"radius {float(self.rod['radius']):.4f} m"
        assert self.pose_error < 3e-3, f"held pose {1e3 * self.pose_error:.2f} mm rms"
        for u, r, tol in zip(KINDS, self.ratios[-1], (0.05, 0.1)):
            assert abs(r - 1.0) < tol, f"{u}: fit / truth {r:.3f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--fresh", action="store_true", help="recompute the cached Gaussians, rod and fit")
    parser.set_defaults(num_frames=600)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
