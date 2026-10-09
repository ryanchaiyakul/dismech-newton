"""Recover a rod from one RGB-D image, then fit its bending stiffness and damping to a video from the same camera.

Shows: a rod model built from data instead of given (its length, radius and held pose fitted to one colour image
and one depth image, as a chain of plain cylinders); :mod:`dismech_newton.splat`, the rod painted with Gaussian
splats rigid in each edge's material frame; gradients from pixels, the image loss of a differentiable rasterizer
(gswarp, in torch) handed back to Warp and taken through the skin and every solver step by one ``tape.backward``.

A painted rod (a solid tube, ray traced), clamped at one end, is held still in a curved pose. One RGB-D camera
takes a still (colour and depth), then films the rod, released, swinging for 1 s at 30 fps. The camera is simulated
with the flaws of a real one (:data:`SENSOR`; ``level`` scales them): lighting with a highlight, the background
known only from a noisy clean plate (plain by default: the rod masked out of the scene), focus blur, read and shot noise, an exposure drift, 8-bit colour; depth at half the resolution, its noise growing
with the square of the distance and correlated across pixels, a bias and a scale error, millimetre steps, pixels
mixing rod and wall at the edges, dropouts, the depth camera misregistered to the colour one; the calibration
itself off (the focal length, the clamp's pose). The fit knows the clamp's pose as calibrated (a fixture: where,
which axis), the rod's weight, the number of segments to model it with, the calibrated camera and its stretch and
twist stiffness. Everything else it reads off the data:

1. The rod as cylinders. A chain of straight cylinders of one radius with flat ends, rendered analytically along
   pixel: its opacity soft in the distance from its outline in the image (each edge a band, each flat end the half
   ellipse its disc shows; a free blur), its depth where the pixel's ray enters it. A first chain from the depth
   image's points (a 3D chamfer, bending penalised), then all of it by Adam against the still's opacity (a soft
   mask where the colour leaves the clean plate) and the depth (a robust loss with a free offset): the clamp's pose
   (free, near the calibrated one), each edge's direction (kinks penalised), one edge length, the radius, the blur.
2. Its paint. Rings of splats on the recovered rod, in each edge's material frame at twist 0, their colours (and
   one opacity and width) fitted to the still: the paint's orientation comes with them, so the clamp's angle is no
   unknown. The side the camera never sees keeps the rod's median colour.
3. Bend stiffness and damping by L-BFGS from the video: the painted rod, skinned to a rod released from the
   recovered pose, rendered over the clean plate and compared pixel by pixel; from the best of a coarse scan over
   1/30-30x the truth (its basin is narrow in EI: a rod that swings at another rate barely overlaps the recorded
   one), the frames blurred by 4, then 2, then 0 px. The fit
   simulates with :class:`~dismech_newton.ADMMDiSMechSolver` at its defaults (at most 50 iterations a step),
   restarted for every rollout, and takes the gradient through its steps; the video is simulated with Newton.

The start pose limits the fit: a node off by a millimetre on 5 cm edges is a kink as curved as the rod, the released
rod rings with it, and the damping absorbs that. One colour view alone leaves depth to the outline's fine detail;
the depth image pins it. With the default flaws the held pose comes out ~6 mm rms (mostly along the camera's
axis), EI within 2%, the damping 20% low; a perfect camera gives 1.2 mm, EI 2% and the damping 19% high (the paint
and the straight edges only approximate the truth).

The viewer plays the three acts: the cylinders fitted (the chain over the still, beside the depth image; grey: the
true pose, green: the recovered), the painted rod beside the still, then every L-BFGS step replayed (green) beside
the truth (grey). The rod and the fit are cached in ``.cache/examples`` (``--fresh`` recomputes them). It needs
CUDA and the ``splat`` extra (torch, gswarp):

    uv run --extra splat examples/fit_capture.py
    uv run --extra splat examples/fit_capture.py --viewer null --test
"""

import math

import newton
import newton.examples
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp
from PIL import Image, ImageDraw
from scipy.optimize import minimize
from utils.common import TapedGraph, cached, capsule_poses, inset, inset_scale
from utils.gaussians import DEVICE, Camera, backprop, to_uint8, view

from dismech_newton import ADMMDiSMechSolver, DiSMechSolver, add_rod, fix_segment, flatten_state
from dismech_newton.solver import advance_frames_kernel
from dismech_newton.splat import Splats, skin
from dismech_newton.triplet import advance_ref_twist_kernel

# -- the rod and the truth -----------------------------------------------------------------
LENGTH, SEGMENTS, RADIUS = 0.5, 10, 0.02
CLAMP, CLAMP_AXIS = np.array([0.0, 0.0, 1.0]), (1.0, 0.0, 0.0)  # the clamped end and its axis (truth)
TRUE = dict(EI=8.0, c=0.1)  # bend stiffness and damping (both planes)
BOUND, SCAN = math.log(30.0), (14, 4)  # the fit's bounds (log of x the truth); the scan's points in each
KINDS = list(TRUE)
GJ, EA = 8.0, 1.0e4  # twist and stretch stiffness, known
BEND, DAMP = [2, 3], [7, 8]  # triplet_params columns of EI and c
TWIST = 0.8  # the held pose's uniform twist [rad]: where the paint faces, never an unknown

# -- the camera: one RGB-D camera; a still of the held rod, then 1 s of video ---------------------------
EYE, LOOK_AT, FOV = (0.15, -0.95, 1.35), (0.28, 0.05, 0.95), math.radians(45.0)
STILL, DEPTH, IMAGE = 512, 256, 256  # px: the still's colour, its depth, the video
DT, EVERY, FRAMES = 1.0 / 240.0, 8, 30  # 30 fps for 1 s
STEPS = EVERY * FRAMES
WALL = 1.8  # [m] the wall behind the rod, square to the camera
SEED = 0

# its flaws; ``level`` scales them all (0: a perfect camera, the scene lit flat on a plain background)
SENSOR = dict(
    level=1.0,
    read=0.01, shot=0.03, video_read=0.05,  # colour noise std: sqrt(read^2 + shot^2 x value); the video's read
    blur=0.8,  # [px of the still] focus, a Gaussian
    gain=0.03,  # the still's and the video's exposure over the clean plate's
    plate_frames=10,  # frames averaged into the clean plate (not scaled)
    texture=0.0,  # the background's texture (0: plain, as the rod masked out of the scene)
    ambient=0.35, highlight=0.3,  # lighting: Lambert with this ambient, a Blinn-Phong highlight (level <= 1)
    depth_noise=2.5e-3, depth_corr=1.5,  # [m] std at 1 m (grows as z^2); its correlation length [depth px]
    depth_bias=5e-3, depth_scale=3e-3,  # [m], relative
    depth_step=1e-3,  # [m] quantisation
    dropout=0.03, edge_loss=0.5,  # depth pixels lost at random; of those mixing rod and wall
    misregister=(1e-3, 0.1),  # the depth camera off the colour one: [m], [deg]
    clamp=(2e-3, 1.0),  # the clamp's pose as calibrated, off the truth: [m], [deg]
    focal=3e-3,  # the calibrated focal length's relative error
)
LIGHT = np.array([-0.3, -0.8, 1.0]) / np.linalg.norm([-0.3, -0.8, 1.0])

# -- the truth's look: a solid tube (a smooth spline through the nodes, flat ends), ray traced, grey with a red
# stripe along m1 and a blue one along -m1 ------------------------------------------------------------------
PIECES, SUPER = 6, 2  # spline pieces per edge; rays per pixel, each way
PAINT, RED, BLUE, BACKGROUND = (0.75, 0.75, 0.72), (0.9, 0.15, 0.1), (0.1, 0.3, 0.95), (0.2, 0.2, 0.22)
STRIPE = 0.35  # [rad] the stripes' half width

# -- the fit's rod: the cylinders, then the paint -----------------------------------------------------
DEPTH_SIGMA = 3e-3  # [m] the depth noise the fit assumes (a robust loss beyond 2.5 of it)
ALPHA_FLOOR = 0.15  # the opacity's error the fit assumes (the mask's and the model's)
FIXTURE = 0.04  # [m] about the clamp the fixture hides the rod: its opacity unused
PRIOR = (3e-3, math.radians(2.0))  # the clamp's pose about the calibrated one: [m], [rad]
SMOOTH = 0.02  # [rad] the change of an edge's turn from the last's
INIT_STEPS, FIT_STEPS, SNAPSHOT = 600, 3000, 100  # Adam steps: the chamfer start, the cylinders; snapshot every
PAINT_RINGS, PAINT_PER_RING, PAINT_STEPS = 4, 16, 400
COARSE = (4.0, 2.0, 0.0)  # [px] the video fit's blurs, coarse to fine


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


# -- the data: a simulated RGB-D camera ----------------------------------------------------------


def flaws() -> dict:
    """The calibration as the fit knows it (the clamp's pose, the field of view) and the true depth camera's pose,
    each off the truth by :data:`SENSOR`'s amounts in random directions."""
    lv, rng = SENSOR["level"], np.random.default_rng(SEED + 1)

    def unit():
        v = rng.standard_normal(3)
        return v / np.linalg.norm(v)

    def tilt(v, deg):  # v turned by deg about a random axis square to it
        v = np.asarray(v, dtype=float)
        a = np.cross(v, unit())
        a /= np.linalg.norm(a)
        th = math.radians(deg)
        return v * math.cos(th) + np.cross(a, v) * math.sin(th) + a * (a @ v) * (1 - math.cos(th))

    shift = SENSOR["misregister"][0] * lv * unit()
    eye, look = np.asarray(EYE), np.asarray(LOOK_AT)
    return dict(clamp=CLAMP + SENSOR["clamp"][0] * lv * unit(), axis=tilt(CLAMP_AXIS, SENSOR["clamp"][1] * lv),
                fov=2 * math.atan(math.tan(FOV / 2) / (1 + SENSOR["focal"] * lv)),
                depth_eye=eye + shift, depth_look=eye + shift + tilt(look - eye, SENSOR["misregister"][1] * lv))


def blur(image, sigma):
    """A Gaussian blur of ``(C, H, W)``, ``sigma`` in pixels."""
    if sigma <= 0:
        return image
    r = int(3 * sigma + 1)
    x = torch.arange(-r, r + 1, device=image.device, dtype=image.dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    c = image.shape[0]
    out = F.conv2d(F.pad(image[None], (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    out = F.conv2d(F.pad(out, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return out[0]


def background(gen) -> torch.Tensor:
    """The wall as the still sees it ``(3, STILL, STILL)``: a smooth texture, mostly in brightness."""
    def noise(n):
        x = torch.randn(1, 3, n, n, generator=gen, device=DEVICE)
        return F.interpolate(x, size=(STILL, STILL), mode="bicubic", align_corners=False)[0]

    tex = noise(6) + 0.5 * noise(24) + 0.25 * noise(96)
    tex = tex / tex.std()
    base = torch.tensor(BACKGROUND, device=DEVICE)[:, None, None]
    amount = SENSOR["level"] * SENSOR["texture"]
    return (base + amount * (0.8 * tex.mean(0, keepdim=True) + 0.2 * tex)).clamp(0.02, 0.98)


def rays(cam, size, box=None):
    """Unit world directions ``(N, 3)`` through pixel centres of a ``size`` image from ``cam``'s pose and field of
    view (``box``: rows and columns ``(y0, y1, x0, x1)``), their pixel indices ``(N,)`` and the camera's forward axis."""
    y0, y1, x0, x1 = box or (0, size, 0, size)
    f = 0.5 * size / math.tan(cam.fov / 2)
    yy, xx = torch.meshgrid(torch.arange(y0, y1, device=DEVICE), torch.arange(x0, x1, device=DEVICE), indexing="ij")
    c = torch.stack([(xx + 0.5 - size / 2) / f, (yy + 0.5 - size / 2) / f, torch.ones_like(xx, dtype=torch.float32)],
                    -1).reshape(-1, 3).float()
    R = torch.tensor(cam.V[:3, :3], dtype=torch.float32, device=DEVICE)
    w = c @ R  # R^T c, row-wise
    return w / w.norm(dim=1, keepdim=True), (yy * size + xx).reshape(-1), R[2]


def centreline(nodes):
    """A Catmull-Rom spline through ``nodes`` ``(N, 3)``: ``PIECES`` straight pieces per edge ``(E x PIECES + 1, 3)``
    and the edge each piece belongs to."""
    P = torch.cat([2 * nodes[:1] - nodes[1:2], nodes, 2 * nodes[-1:] - nodes[-2:-1]])
    p0, p1, p2, p3 = (x[:, None] for x in (P[:-3], P[1:-2], P[2:-1], P[3:]))
    t = (torch.arange(PIECES, device=DEVICE, dtype=nodes.dtype) / PIECES)[None, :, None]
    pts = 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 + (3 * p1 - p0 - 3 * p2 + p3) * t**3)
    owner = torch.arange(len(nodes) - 1, device=DEVICE).repeat_interleave(PIECES)
    return torch.cat([pts.reshape(-1, 3), nodes[-1:]]), owner


def cast(eye, w, P, r):
    """Rays from ``eye`` (unit directions ``w (N, 3)``) into the solid tube of radius ``r`` on the pieces ``P``:
    whether each hits, the ray parameter, the surface normal and the piece hit."""
    a, e = P[:-1], P[1:] - P[:-1]
    ln = e.norm(dim=1)
    t = e / ln[:, None]
    m = eye[None] - a
    wt, mw, mt, mm = w @ t.T, w @ m.T, (m * t).sum(1)[None], (m * m).sum(1)[None]
    A = (1 - wt**2).clamp_min(1e-9)
    B, C = 2 * (mw - mt * wt), mm - mt**2 - r**2
    disc = B**2 - 4 * A * C
    s = (-B - torch.sqrt(disc.clamp_min(0))) / (2 * A)
    u = mt + s * wt
    inf = torch.full_like(s, math.inf)
    s = torch.where((disc > 0) & (s > 0) & (u >= 0) & (u <= ln[None]), s, inf)
    caps = []
    for k, at in ((0, 0.0), (-1, ln[-1])):  # the flat ends
        wk = wt[:, k]
        sc = (at - mt[0, k]) / torch.where(wk.abs() > 1e-9, wk, torch.full_like(wk, 1e-9))
        q = m[k] + sc[:, None] * w
        radial = q - (q @ t[k])[:, None] * t[k]
        caps.append(torch.where((sc > 0) & (radial.norm(dim=1) <= r), sc, inf[:, 0]))
    mj = eye[None] - P[1:-1]  # spheres at the joints: no gaps outside the bends
    bj = w @ mj.T
    dj = bj**2 - (mj * mj).sum(1)[None] + r**2
    sj = -bj - torch.sqrt(dj.clamp_min(0))
    xj = eye[None, None] + sj[..., None] * w[:, None]  # within the flat ends' planes
    inside = (((xj - P[0]) @ t[0]) >= 0) & (((xj - P[-1]) @ t[-1]) <= 0)
    sj = torch.where((dj > 0) & (sj > 0) & inside, sj, torch.full_like(sj, math.inf))
    best, k = torch.cat([s, torch.stack(caps, 1), sj], 1).min(1)
    hit = torch.isfinite(best)
    E = len(a)
    joint = k >= E + 2
    piece = torch.where(k < E, k, torch.where(k == E, 0, torch.where(k == E + 1, E - 1, k - E - 2)))
    x = eye + torch.where(hit, best, torch.zeros_like(best))[:, None] * w
    q = x - a[piece]
    n = q - (q * t[piece]).sum(1, keepdim=True) * t[piece]
    n = torch.where(joint[:, None], x - P[(k - E - 1).clamp(0, E - 1)], n)
    n = torch.where((k == E)[:, None], -t[0], torch.where((k == E + 1)[:, None], t[-1], n))
    return hit, best, n / n.norm(dim=1, keepdim=True).clamp_min(1e-12), piece


def trace(cam, size, q, d1, chunk=1 << 16):
    """The truth through ``size`` x ``size`` pixel centres of ``cam``: its colour (lit, where a ray hits), whether
    it hits, the camera z of the hit; each ``(..., size, size)``. ``q``: the DOFs, ``d1``: the reference
    directors."""
    E = SEGMENTS
    nodes, theta = q[: 3 * (E + 1)].reshape(-1, 3), q[3 * (E + 1) :]
    P, owner = centreline(nodes)
    te = nodes[1:] - nodes[:-1]
    te = te / te.norm(dim=1, keepdim=True)
    m1e = torch.cos(theta)[:, None] * d1 + torch.sin(theta)[:, None] * torch.cross(te, d1, dim=1)
    tp = P[1:] - P[:-1]
    tp = tp / tp.norm(dim=1, keepdim=True)
    m1 = m1e[owner] - (m1e[owner] * tp).sum(1, keepdim=True) * tp
    m1 = m1 / m1.norm(dim=1, keepdim=True)
    m2 = torch.cross(tp, m1, dim=1)
    w, _, fwd = rays(cam, size)
    eye = torch.tensor(cam.eye, dtype=torch.float32, device=DEVICE)
    light = torch.tensor(LIGHT, dtype=torch.float32, device=DEVICE)
    paint = torch.tensor([PAINT, RED, BLUE], dtype=torch.float32, device=DEVICE)
    k = min(SENSOR["level"], 1.0)
    colour, hits, z = [], [], []
    for c in range(0, len(w), chunk):
        wc = w[c : c + chunk]
        hit, s, n, piece = cast(eye, wc, P, RADIUS)
        phi = torch.atan2((n * m2[piece]).sum(1), (n * m1[piece]).sum(1))
        side = (n * tp[piece]).sum(1).abs() < 0.5  # not an end
        which = torch.where(side & (phi.abs() < STRIPE), 1, torch.where(side & (phi.abs() > math.pi - STRIPE), 2, 0))
        h = light - wc
        h = h / h.norm(dim=1, keepdim=True)
        diffuse = SENSOR["ambient"] + (1 - SENSOR["ambient"]) * (n @ light).clamp_min(0)
        spot = SENSOR["highlight"] * (n * h).sum(1).clamp_min(0) ** 40
        col = paint[which] * (1 - k + k * diffuse[:, None]) + k * spot[:, None]
        colour.append(torch.where(hit[:, None], col.clamp(0, 1), torch.zeros_like(col)))
        hits.append(hit)
        z.append(torch.where(hit, s * (wc @ fwd), torch.zeros_like(s)))
    return (torch.cat(colour).T.reshape(3, size, size), torch.cat(hits).reshape(size, size),
            torch.cat(z).reshape(size, size))


def photograph(cam, size, q, d1, wall, gen, read, blur_px) -> torch.Tensor:
    """What the colour camera records ``(3, size, size)``: the truth over the wall (``SUPER`` x ``SUPER`` rays a
    pixel), blurred, exposed, noisy, 8-bit."""
    lv = SENSOR["level"]
    colour, hit, _ = trace(cam, SUPER * size, q, d1)
    up = wall.repeat_interleave(SUPER, 1).repeat_interleave(SUPER, 2)
    image = F.avg_pool2d(torch.where(hit, colour, up)[None], SUPER)[0]
    image = blur(image, lv * blur_px) * (1 + lv * SENSOR["gain"])
    std = torch.sqrt((lv * read) ** 2 + (lv * SENSOR["shot"]) ** 2 * image.clamp_min(0))
    image = image + std * torch.randn(image.shape, generator=gen, device=DEVICE)
    return torch.round(image.clamp(0, 1) * 255) / 255


def depth_image(cam, q, d1, gen) -> torch.Tensor:
    """What the depth camera records ``(DEPTH, DEPTH)``, 0 where it has no reading: at twice its resolution the
    tube's depth or the wall's, averaged 2x2 (pixels on the edge mix the two), scaled and biased, with correlated
    noise growing as z^2, in steps; some pixels lost."""
    lv = SENSOR["level"]
    _, hit, z = trace(cam, 2 * DEPTH, q, d1)
    depth = F.avg_pool2d(torch.where(hit, z, torch.full_like(z, WALL))[None, None], 2)[0, 0]
    mixed = F.avg_pool2d(hit[None, None].float(), 2)[0, 0]
    mixed = (mixed > 0) & (mixed < 1)
    depth = depth * (1 + lv * SENSOR["depth_scale"]) + lv * SENSOR["depth_bias"]
    n = torch.randn((1, DEPTH, DEPTH), generator=gen, device=DEVICE)
    n = blur(n, lv * SENSOR["depth_corr"])[0]
    depth = depth + lv * SENSOR["depth_noise"] * depth**2 * n / n.std()
    if lv > 0:
        depth = torch.round(depth / (lv * SENSOR["depth_step"])) * (lv * SENSOR["depth_step"])
    lost = (torch.rand((DEPTH, DEPTH), generator=gen, device=DEVICE) < lv * SENSOR["dropout"]) | (
        mixed & (torch.rand((DEPTH, DEPTH), generator=gen, device=DEVICE) < min(1.0, lv * SENSOR["edge_loss"])))
    return torch.where(lost, torch.zeros_like(depth), depth)


def record() -> dict:
    """The data, all from the one camera (torch): the still ``(3, STILL, STILL)``, its depth ``(DEPTH, DEPTH)``, the
    clean plate (the empty scene, averaged) at the still's and the video's size, the video ``(FRAMES + 1, 3, H, W)``;
    the calibration as the fit knows it; for checking only: the true DOFs per frame, the rod's weight."""
    model, solver = build()
    rest, *sts = states(model, STEPS + 2)
    start(solver, rest, sts[0], held_q(held_pose(), TWIST))
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    cal = flaws()
    cam = Camera(EYE, LOOK_AT, STILL, FOV)
    depth_cam = Camera(cal["depth_eye"], cal["depth_look"], DEPTH, FOV)
    wall = background(gen)
    wall_video = F.avg_pool2d(wall[None], STILL // IMAGE)[0]
    lv = SENSOR["level"]
    with torch.no_grad():
        q0, d0 = view(sts[0].dismech.q), view(sts[0].dismech.edge_d1_q)
        still = photograph(cam, STILL, q0, d0, wall, gen, SENSOR["read"], SENSOR["blur"])
        depth = depth_image(depth_cam, q0, d0, gen)
        plate = blur(wall, lv * SENSOR["blur"]) + lv * SENSOR["read"] / math.sqrt(SENSOR["plate_frames"]) * torch.randn(
            wall.shape, generator=gen, device=DEVICE)
        video, q = [], []
        for i in range(STEPS + 1):
            if i % EVERY == 0:
                s = sts[i].dismech
                video.append(photograph(cam, IMAGE, view(s.q), view(s.edge_d1_q), wall_video, gen,
                                        SENSOR["video_read"], SENSOR["blur"] * IMAGE / STILL))
                q.append(s.q.numpy())
            if i < STEPS:
                solver.step(sts[i], sts[i + 1], None, None, DT)
    return dict(still=still, depth=depth, plate=plate, plate_video=F.avg_pool2d(plate[None], STILL // IMAGE)[0],
                video=torch.stack(video), calibration=cal, q=np.array(q), mass=float(model.particle_mass.numpy().sum()))


# -- 1. the rod as cylinders ---------------------------------------------------------------------


def cameras(cal: dict, background=(0.0, 0.0, 0.0)):
    """The still's and the video's camera as calibrated (the depth camera taken to be the colour one)."""
    return (Camera(EYE, LOOK_AT, STILL, cal["fov"], background), Camera(EYE, LOOK_AT, IMAGE, cal["fov"], background))


def outline(cam, size, xy, nodes, radius):
    """The signed distance [px] of pixel points ``xy (N, 2)`` (pixel centres at index + 0.5) from the outline of the
    chain of cylinders on ``nodes`` in a ``size`` image from ``cam``: each edge a band tapering with the depth,
    round where edges meet; each flat end the half ellipse its disc shows, as wide as the band and as deep as the
    disc is turned to the camera."""
    R = torch.tensor(cam.V[:3, :3], dtype=torch.float32, device=DEVICE)
    c = nodes @ R.T + torch.tensor(cam.V[:3, 3], dtype=torch.float32, device=DEVICE)
    f = 0.5 * size / math.tan(cam.fov / 2)
    p = f * c[:, :2] / c[:, 2:] + 0.5 * size
    rho = f * radius / c[:, 2]  # the band's half width at each node [px]
    p0, d = p[:-1], p[1:] - p[:-1]
    x = xy[:, None] - p0[None]  # (N, E, 2)
    h_raw = (x * d[None]).sum(-1) / (d * d).sum(-1)[None]
    h = h_raw.clamp(0, 1)
    sd = (x - h[..., None] * d[None]).norm(dim=-1) - (rho[:-1] + h * (rho[1:] - rho[:-1]))
    eye = torch.tensor(cam.eye, dtype=torch.float32, device=DEVICE)
    for k, j, out in ((0, 0, -1.0), (-1, -1, 1.0)):  # the ends: node k, edge j, outward along the edge's image
        t = nodes[1] - nodes[0] if k == 0 else nodes[-1] - nodes[-2]
        view = nodes[k] - eye
        turned = ((t @ view) / (t.norm() * view.norm())).abs()
        a = rho[k]
        b = (a * turned).clamp_min(0.05 * a)
        dk = out * d[j] / d[j].norm()
        u, v = ((xy - p[k]) @ dk), ((xy - p[k]) @ torch.stack([-dk[1], dk[0]]))
        k0 = torch.sqrt((u / b) ** 2 + (v / a) ** 2)
        k1 = torch.sqrt((u / b**2) ** 2 + (v / a**2) ** 2).clamp_min(1e-9)
        cap = torch.where(u > 0, k0 * (k0 - 1) / k1, v.abs() - a)  # an ellipse's distance (approximate)
        beyond = (h_raw[:, j] < 0) if k == 0 else (h_raw[:, j] > 1)
        sd = sd.clone()
        sd[:, j] = torch.where(beyond, cap, sd[:, j])
    return sd.min(1)[0]


def surface(eye, w, nodes, radius):
    """The ray parameter where rays from ``eye`` (unit directions ``w (N, 3)``) enter the chain of cylinders on
    ``nodes``: the side of the edge they pass closest to, or its flat end's disc beyond the chain's ends (where they
    pass closest if they miss)."""
    a, e = nodes[:-1], nodes[1:] - nodes[:-1]
    ln = e.norm(dim=1)
    t = e / ln[:, None]
    m = eye[None] - a  # (E, 3)
    wt, mw, mt = w @ t.T, w @ m.T, (m * t).sum(1)[None]  # (N, E)
    mm = (m * m).sum(1)[None]
    den = (1 - wt**2).clamp_min(1e-6)
    s = (mt * wt - mw) / den  # closest approach of the ray to each axis line
    u = mt + s * wt
    rho2 = (mm + s**2 + u**2 + 2 * s * mw - 2 * u * mt - 2 * s * u * wt).clamp_min(0)
    over = (-u).clamp_min(0) + (u - ln[None]).clamp_min(0)
    k = (rho2 + over**2).argmin(1)
    pick = lambda x: x.gather(1, k[:, None])[:, 0]  # noqa: E731
    A, B, C = pick(den), 2 * pick(mw - mt * wt), pick((mm - mt**2).expand_as(den)) - radius**2
    hit = (-B - torch.sqrt((B**2 - 4 * A * C).clamp_min(0) + 1e-12)) / (2 * A)
    E = len(a)
    for j, at in ((0, 0.0), (E - 1, ln[-1])):  # rays entering through an end's disc
        wj = wt[:, j]
        sc = (at - mt[0, j]) / torch.where(wj.abs() > 1e-6, wj, torch.full_like(wj, 1e-6))
        uh = mt[0, j] + hit * wj  # where the side entry would be, along the edge
        through = (k == j) & ((uh < 0) if j == 0 else (uh > ln[-1]))
        hit = torch.where(through, sc, hit)
    return hit


class Chain:
    """The fit's rod: the clamped node, every edge's direction, one edge length, the radius, the blur [px], the
    depth's offset."""

    def __init__(self, cal):
        self.clamp = torch.tensor(cal["clamp"], dtype=torch.float32, device=DEVICE)
        self.axis = torch.tensor(cal["axis"], dtype=torch.float32, device=DEVICE)
        self.p = dict(x0=self.clamp.clone(), v=self.axis.repeat(SEGMENTS, 1).clone(),
                      log_l=torch.tensor(math.log(0.03), device=DEVICE),  # a 0.3 m guess
                      log_r=torch.tensor(math.log(0.02), device=DEVICE), log_blur=torch.tensor(0.0, device=DEVICE),
                      offset=torch.tensor(0.0, device=DEVICE))
        for x in self.p.values():
            x.requires_grad_()

    def nodes(self):
        u = self.p["v"] / self.p["v"].norm(dim=1, keepdim=True)
        return torch.cat([self.p["x0"][None], self.p["x0"][None] + torch.exp(self.p["log_l"]) * torch.cumsum(u, 0)]), u

    def prior(self):
        """The clamp's pose near the calibrated one; the curvature changing little from edge to edge (no kinks)."""
        u = self.nodes()[1]
        kink = u[2:] - 2 * u[1:-1] + u[:-2]
        return ((((self.p["x0"] - self.clamp) / PRIOR[0]) ** 2).sum() + ((u[0] - self.axis) ** 2).sum() / PRIOR[1] ** 2
                + (kink**2).sum() / SMOOTH**2)


def recover(data: dict) -> dict:
    """The rod (length, radius, held pose) from the still and its depth, as :class:`Chain`; ``snapshots``: the
    nodes every :data:`SNAPSHOT` steps."""
    cal = data["calibration"]
    cam, _ = cameras(cal)
    eye = torch.tensor(cam.eye, dtype=torch.float32, device=DEVICE)

    # the observed opacity: a soft mask, half where the colour's difference from the plate crosses a threshold well
    # above the noise (low: the rod's shaded edges are dim)
    d = (data["still"] - data["plate"]).abs().amax(0)
    threshold = max(float(d.median() + 6 * 1.4826 * (d - d.median()).abs().median()), 0.04)
    alpha = ((d - 0.5 * threshold) / threshold).clamp(0, 1)
    ys, xs = torch.nonzero(alpha > 0.25, as_tuple=True)
    box = (max(int(ys.min()) - 16, 0), min(int(ys.max()) + 17, STILL), max(int(xs.min()) - 16, 0),
           min(int(xs.max()) + 17, STILL))
    w_still, pix, fwd = rays(cam, STILL, box)
    target = alpha.reshape(-1)[pix]
    # the fixture hides the rod's clamped end: no opacity within FIXTURE of the clamp
    clamp = torch.tensor(cal["clamp"], dtype=torch.float32, device=DEVICE)
    m = clamp - eye
    seen = (m - (w_still @ m)[:, None] * w_still).norm(dim=1) > FIXTURE
    target = target[seen]
    xy = torch.stack([pix % STILL, pix // STILL], 1)[seen].float() + 0.5

    # the depth on the rod: valid, well in front of the wall, not next to a pixel that is not
    z = data["depth"]
    valid = z > 0
    near = valid & (z < z[valid].median() - 0.1)
    near = -F.max_pool2d(-near[None, None].float(), 3, 1, 1)[0, 0] > 0.5  # eroded by a pixel
    w_all, dpix, _ = rays(cam, DEPTH)
    pick = near.reshape(-1)[dpix]
    w_depth, z_depth = w_all[pick], z.reshape(-1)[dpix][pick]
    points = eye + w_depth * (z_depth / (w_depth @ fwd) + 0.02)[:, None]  # the surface's points, 2 cm deeper

    rod = Chain(cal)
    snapshots = []

    # a start: the chain to the depth's points (both ways), bending penalised
    opt = torch.optim.Adam([rod.p["x0"], rod.p["v"], rod.p["log_l"]], lr=0.01)
    along = torch.linspace(0, 1, 11, device=DEVICE)[:-1]
    for it in range(INIT_STEPS):
        if it == INIT_STEPS * 2 // 3:
            for g in opt.param_groups:
                g["lr"] = 0.003
        nodes, u = rod.nodes()
        samples = (nodes[:-1, None] + along[None, :, None] * (nodes[1:] - nodes[:-1])[:, None]).reshape(-1, 3)
        D = torch.cdist(samples, points)
        loss = (D.min(1)[0] ** 2).mean() + (D.min(0)[0] ** 2).mean() + 1e-4 * ((u[1:] - u[:-1]) ** 2).sum()
        loss = loss / 1e-4 + rod.prior()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % SNAPSHOT == 0:
            snapshots.append(nodes.detach().cpu().numpy())

    # the cylinders against the opacity and the depth, the depth's offset from where the start leaves it
    with torch.no_grad():
        hit_d = surface(eye, w_depth, rod.nodes()[0], torch.exp(rod.p["log_r"]))
        rod.p["offset"].copy_((z_depth - hit_d * (w_depth @ fwd)).median())
    lr = dict(x0=2e-4, v=2e-3, log_l=1e-3, log_r=2e-3, log_blur=1e-2, offset=2e-4)
    opt = torch.optim.Adam([{"params": [rod.p[k]], "lr": lr[k]} for k in rod.p])
    warm = FIT_STEPS // 20
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / warm) * (
        0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * i / FIT_STEPS))))  # warm up, then a cosine
    for it in range(FIT_STEPS + 1):
        nodes, _ = rod.nodes()
        r = torch.exp(rod.p["log_r"])
        a = torch.sigmoid(-outline(cam, STILL, xy, nodes, r) / torch.exp(rod.p["log_blur"]))
        hit_d = surface(eye, w_depth, nodes, r)
        res = (hit_d * (w_depth @ fwd) + rod.p["offset"] - z_depth) / DEPTH_SIGMA
        huber = torch.where(res.abs() < 2.5, 0.5 * res**2, 2.5 * (res.abs() - 1.25))
        loss = (((a - target) / ALPHA_FLOOR) ** 2).sum() + huber.sum() + rod.prior()
        if it % SNAPSHOT == 0:
            snapshots.append(nodes.detach().cpu().numpy())
        if it == FIT_STEPS:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    nodes, _ = rod.nodes()
    nodes = nodes.detach().cpu().numpy().astype(np.float64)
    p = {k: float(v.detach()) for k, v in rod.p.items() if v.numel() == 1}
    return dict(nodes=nodes, length=SEGMENTS * math.exp(p["log_l"]), radius=math.exp(p["log_r"]),
                blur=math.exp(p["log_blur"]), offset=p["offset"], snapshots=np.array(snapshots), depth_pixels=int(pick.sum()))


# -- 2. its paint ---------------------------------------------------------------------------------


def frames_at(nodes, length, radius):
    """Each edge's material frame ``(E, 3, 3)`` (columns m1, m2, t) at the held pose, twist 0."""
    model, solver = build(length, radius)
    rest, st = states(model, 2)
    start(solver, rest, st, held_q(nodes, 0.0))
    d1 = st.dismech.edge_d1_q.numpy().astype(np.float64)
    t = np.diff(nodes, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.stack([d1, np.cross(t, d1), t], 2)


def painted(rod: dict, data: dict) -> dict:
    """Rings of splats on the recovered rod, bound to its edges in their material frames, their colours (and one
    opacity and width) fitted to the still over the clean plate."""
    nodes, radius = rod["nodes"], rod["radius"]
    E, R, P = SEGMENTS, PAINT_RINGS, PAINT_PER_RING
    l = rod["length"] / E
    edge, s, k = (a.ravel() for a in np.meshgrid(np.arange(E), (np.arange(R) + 0.5) / R, np.arange(P), indexing="ij"))
    phi = 2 * np.pi * k / P
    c, sn = np.cos(phi), np.sin(phi)
    rotation = np.zeros((len(edge), 3, 3))
    rotation[:, :2, 0] = np.stack([c, sn], 1)
    rotation[:, :2, 1] = np.stack([-sn, c], 1)
    rotation[:, 2, 2] = 1.0
    uv = 0.75 * radius * np.stack([c, sn], 1)
    log_scale = np.log(np.tile([0.3 * radius, 0.6 * 2 * np.pi * 0.75 * radius / P, 0.6 * l / R], (len(edge), 1)))

    frame = frames_at(nodes, rod["length"], radius)
    T = lambda x: torch.tensor(x, dtype=torch.float32, device=DEVICE)  # noqa: E731
    Fe = T(frame[edge])
    base = T(nodes[edge] + s[:, None] * (nodes[edge + 1] - nodes[edge]))
    uv_t, Rl, ls = T(uv), T(rotation), T(log_scale)
    i, j = np.triu_indices(3)

    cam, _ = cameras(data["calibration"], data["plate"])
    target = data["still"]
    d = (target - data["plate"]).abs().amax(0)
    on_rod = d > d.median() + 0.5 * (d.max() - d.median())
    median = target[:, on_rod].median(1).values.clamp(0.02, 0.98)
    logit = torch.log(median / (1 - median))
    colour = logit.repeat(len(edge), 1).clone().requires_grad_()
    opacity = torch.tensor(2.0, device=DEVICE, requires_grad=True)
    width = torch.tensor(0.0, device=DEVICE, requires_grad=True)

    def splats():
        uvw = uv_t * torch.exp(width)
        means = base + Fe[:, :, 0] * uvw[:, :1] + Fe[:, :, 1] * uvw[:, 1:]
        Rw = Fe @ Rl
        sig = torch.exp(ls + torch.stack([width, width, torch.zeros_like(width)]))
        C = (Rw * sig[:, None, :] ** 2) @ Rw.transpose(1, 2)
        return means, C[:, i, j]

    opt = torch.optim.Adam([{"params": [colour], "lr": 0.05}, {"params": [opacity, width], "lr": 0.02}])
    for _ in range(PAINT_STEPS):
        m, c6 = splats()
        op = torch.sigmoid(opacity).expand(len(edge))
        loss = ((cam(m, c6, torch.sigmoid(colour), op) - target) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        m, c6 = splats()
        psnr = -10 * math.log10(float(((cam(m, c6, torch.sigmoid(colour), torch.sigmoid(opacity).expand(len(edge)))
                                         - target) ** 2).mean()))
    w = math.exp(float(width))
    log_scale[:, :2] += math.log(w)
    return dict(edge=edge, s=s, uv=uv * w, rotation=rotation, log_scale=log_scale,
                colour=torch.sigmoid(colour).detach().cpu().numpy(),
                opacity=np.full(len(edge), float(torch.sigmoid(opacity))), psnr=psnr)


# -- 3. physics from the video ---------------------------------------------------------------


class VideoFit:
    """``self(theta)``: the squared pixel error (averaged over the frames) and its gradient in
    ``theta = log (EI, c) / truth``, the painted rod skinned to a rod released from the recovered pose."""

    def __init__(self, rod: dict, mass: float, data: dict):
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
        self.camera = cameras(data["calibration"], data["plate_video"])[1]
        self.recorded = data["video"]
        self.focus(0.0)
        self.taped = TapedGraph(self.run, self.solver)

    def focus(self, sigma: float):
        """Compare the frames blurred by ``sigma`` [px] (0: as recorded)."""
        self.sigma = sigma
        self.video = torch.stack([blur(t, sigma) for t in self.recorded])

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
        loss = sum(backprop(f, lambda m, c, t=t: ((blur(self.render(m, c), self.sigma) - t) ** 2).sum() / n)
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
    """L-BFGS from the best of a scan over the bounds (the truth's basin is narrow in EI), coarse to fine (the frames blurred by each of :data:`COARSE` in turn: far from the
    truth the rendered rod barely overlaps the recorded one, and sharp frames give no direction): every accepted
    iterate, its loss (at its blur) and replay."""
    f.focus(COARSE[0])
    grid = [np.array([a, b]) for a in np.linspace(-BOUND, BOUND, SCAN[0]) for b in np.linspace(-BOUND, BOUND, SCAN[1])]
    scan = [f(theta)[0] for theta in grid]
    theta = grid[int(np.nanargmin(scan))]
    print("  scan: best " + "  ".join(f"{k} {v:.4f}" for k, v in zip(KINDS, np.exp(theta))), flush=True)
    iterates, losses, best = [theta], [float(np.nanmin(scan))], {}

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

    for sigma in COARSE:
        f.focus(sigma)
        best.clear()
        print(f"  blur {sigma:g} px", flush=True)
        theta = minimize(fun, theta, jac=True, method="L-BFGS-B", callback=accepted,
                         bounds=[(-BOUND, BOUND)] * 2, options=dict(maxiter=50, ftol=1e-7)).x
    q, d1 = zip(*(f.replay(t) for t in iterates))
    return dict(theta=np.array(iterates), loss=np.array(losses), loss_true=f(np.zeros(2))[0], q=np.array(q),
                d1=np.array(d1))


def results(data: dict, fresh: bool = False) -> dict:
    """The rod and the fit, each computed once per configuration (``.cache/examples``)."""
    scene = dict(version=2, rod=[LENGTH, SEGMENTS, RADIUS, list(CLAMP), GJ, EA, TWIST], true=TRUE,
                 camera=[EYE, LOOK_AT, FOV, STILL, DEPTH, IMAGE, WALL], video=[DT, EVERY, FRAMES], sensor=SENSOR,
                 look=[PIECES, SUPER, PAINT, RED, BLUE, BACKGROUND, STRIPE, list(LIGHT)], seed=SEED)
    rod_config = dict(scene=scene, fit=[DEPTH_SIGMA, ALPHA_FLOOR, FIXTURE, PRIOR, SMOOTH, INIT_STEPS, FIT_STEPS, SNAPSHOT],
                      paint=[PAINT_RINGS, PAINT_PER_RING, PAINT_STEPS], version=1)

    def rod():
        r = recover(data)
        return dict(r, **painted(r, data))

    rod = cached("fit_capture-rgbd-rod", rod_config, rod, fresh)
    out = cached("fit_capture-rgbd-fit", dict(rod=rod_config, bound=BOUND, scan=SCAN, coarse=COARSE, version=1),
                 lambda: fit(VideoFit(rod, data["mass"], data)), fresh)
    return dict(rod=rod, **out)


# -- the viewer --------------------------------------------------------------------------------

GREY, GREEN = (0.82, 0.82, 0.82), (0.3, 0.8, 0.5)
BLUE_LINE, PURPLE = "#5aaaff", "#c08cff"


def rod_text(ax, title, lines, scale):
    ax.set_axis_off()
    ax.text(0.03, 0.92, title, color="white", fontsize=22 * scale, transform=ax.transAxes, va="top")
    for i, line in enumerate(lines):
        ax.text(0.05, 0.74 - 0.14 * i, line, color="#ebebeb", fontsize=17 * scale, transform=ax.transAxes, va="top",
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
    ax.set_xlabel("step")
    ax.set_ylabel("fit / truth")
    ax.legend(loc="upper right")


def chain_over(image, cam, nodes, colour=(255, 255, 255)) -> np.ndarray:
    out = Image.fromarray(image)
    d = ImageDraw.Draw(out)
    px = cam.project(nodes)
    d.line([tuple(p) for p in px], fill=colour, width=2)
    for p in px:
        d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=colour, outline=(20, 20, 20))
    return np.asarray(out)


class Example:
    """Act 1: the cylinders fitted (the chain over the still, beside the depth). Act 2: the painted rod beside the
    still. Act 3: each L-BFGS step replays the video."""

    hold, act2, pause = 8, 60, 20  # frames per snapshot, act 2's frames, frames held after each replay

    def __init__(self, viewer, args=None):
        import matplotlib  # noqa: PLC0415

        self.viewer = viewer
        self.data = record()
        r = results(self.data, bool(args is not None and args.fresh))
        self.r, self.rod = r, r["rod"]
        self.ratios = np.exp(r["theta"])
        self.fit = VideoFit(self.rod, self.data["mass"], self.data)
        true_nodes = self.data["q"][0, : 3 * (SEGMENTS + 1)].reshape(-1, 3)
        self.pose_error = np.sqrt(np.mean(np.sum((self.rod["nodes"] - true_nodes) ** 2, 1)))
        still_cam, _ = cameras(self.data["calibration"])
        true_cam = Camera(EYE, LOOK_AT, STILL, FOV)
        dz = still_cam.depth(self.rod["nodes"]) - true_cam.depth(true_nodes)
        self.depth_error = float(np.sqrt(np.mean(dz**2)))
        print(f"rod: length {1e3 * float(self.rod['length']):.1f} mm (truth {1e3 * LENGTH:.0f}), radius "
              f"{1e3 * float(self.rod['radius']):.1f} mm (truth {1e3 * RADIUS:.0f}), "
              f"held pose {1e3 * self.pose_error:.2f} mm rms ({1e3 * self.depth_error:.2f} along the camera's axis), "
              f"blur {float(self.rod['blur']):.2f} px, depth offset {1e3 * float(self.rod['offset']):.1f} mm, "
              f"paint PSNR {float(self.rod['psnr']):.1f} dB")
        for k, ratio in enumerate(self.ratios):
            print(f"step {k:2d}: loss {r['loss'][k]:8.2f}  " + "  ".join(f"{u} {v:.4f}" for u, v in zip(KINDS, ratio)))
        print(f"loss at the true EI and c {float(r['loss_true']):.2f}")

        self.cam = still_cam
        self.still = to_uint8(self.data["still"])
        z = self.data["depth"].cpu().numpy()
        valid = z > 0
        lo, hi = np.percentile(z[valid], [1, 99])
        colours = matplotlib.colormaps["turbo"](np.clip((z - lo) / (hi - lo), 0, 1))[..., :3]
        colours[~valid] = 0
        depth = (np.repeat(np.repeat(colours, STILL // DEPTH, 0), STILL // DEPTH, 1) * 255).astype(np.uint8)
        self.depth = depth
        self.recording = [to_uint8(t) for t in self.data["video"]]
        self.truth_q = self.data["q"]
        self.true_nodes = true_nodes
        self.q_wp = wp.zeros_like(self.fit.rest.dismech.q)
        self.d1_wp = wp.zeros_like(self.fit.rest.dismech.edge_d1_q)
        start(self.fit.solver, self.fit.rest, self.fit.states[0], self.fit.q0)
        s0 = self.fit.states[0].dismech
        means, cov = (view(a).clone() for a in skin(self.fit.model, self.fit.splats, s0.q, s0.edge_d1_q))
        paint_cam = cameras(self.data["calibration"], self.data["plate"])[0]
        with torch.no_grad():
            own = to_uint8(self.fit.render(means, cov, paint_cam))
        self.painted_image = np.concatenate([self.still, own], 1)
        self.rod_lines = [f"length {1e3 * float(self.rod['length']):.1f} mm   (truth {1e3 * LENGTH:.0f})",
                          f"radius {1e3 * float(self.rod['radius']):.1f} mm   (truth {1e3 * RADIUS:.0f})",
                          f"held pose {1e3 * self.pose_error:.1f} mm rms over {SEGMENTS + 1} nodes",
                          f"depth offset {1e3 * float(self.rod['offset']):.1f} mm, blur {float(self.rod['blur']):.2f} px"]
        self.paint_lines = [f"{len(self.rod['edge'])} splats on {SEGMENTS} edges",
                            f"colours fitted to the still: {float(self.rod['psnr']):.1f} dB",
                            "the unseen side: the median colour"]

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
        self.act1 = len(self.rod["snapshots"]) * self.hold
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
            nodes = self.rod["snapshots"][k]
            self.viewer.log_image("images", np.concatenate([chain_over(self.still, self.cam, nodes), self.depth], 1))
            self.viewer.log_image("fit", inset(lambda ax: rod_text(ax, "the rod as cylinders", self.rod_lines, s),
                                               size))
            x = [self.true_nodes, nodes]
        elif act == 2:
            self.viewer.log_image("images", self.painted_image)
            self.viewer.log_image("fit", inset(lambda ax: rod_text(ax, "its paint", self.paint_lines, s), size))
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
        assert abs(float(self.rod["length"]) / LENGTH - 1.0) < 0.02, f"length {float(self.rod['length']):.4f} m"
        assert abs(float(self.rod["radius"]) / RADIUS - 1.0) < 0.15, f"radius {float(self.rod['radius']):.4f} m"
        assert self.pose_error < 1e-2, f"held pose {1e3 * self.pose_error:.2f} mm rms"
        for u, r, tol in zip(KINDS, self.ratios[-1], (0.05, 0.3)):
            assert abs(r - 1.0) < tol, f"{u}: fit / truth {r:.3f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--fresh", action="store_true", help="recompute the cached rod and fit")
    parser.set_defaults(num_frames=600)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
