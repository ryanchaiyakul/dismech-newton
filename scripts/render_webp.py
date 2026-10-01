"""Render the cable examples headless to animated WebP, for the README.

    uv run scripts/render_webp.py
    uv run scripts/render_webp.py --scenes knot --stride 1
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))  # the example scenes

import cable_plectoneme
import cantilever
import newton.viewer
import numpy as np
import overhand_knot
import warp as wp
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

# name: (example, frames, camera elevation [deg], azimuth [deg] (None: across the end-to-end line),
#        zoom, track only the self-contact core (else the whole rope)); pitch None keeps the example's camera
SCENES = {
    "cantilever": (cantilever.Example, 600, None, None, None, False),  # render at --width 1000 --height 688 (its framing)
    "plectoneme": (cable_plectoneme.Example, 780, -5.0, 90.0, 0.75, False),
    "knot": (overhand_knot.Example, 420, -20.0, None, 1.8, True),
}


def core(x: np.ndarray, radius: float, exclude: int = 6) -> np.ndarray:
    """Nodes within two diameters of a part of the rope more than ``exclude`` nodes away."""
    i, j = cKDTree(x).query_pairs(4.0 * radius, output_type="ndarray").T
    far = np.abs(i - j) > exclude
    idx = np.unique(np.concatenate((i[far], j[far])))
    return x[idx] if len(idx) else x


class Camera:
    """Looks along (pitch, yaw) at the bounding sphere of the tracked points, smoothed over frames;
    the view only ever tightens, it never zooms back out."""

    def __init__(self, viewer, pitch: float, yaw: float, zoom: float, smoothing: float = 0.08):
        self.viewer, self.pitch, self.yaw, self.zoom, self.smoothing = viewer, pitch, yaw, zoom, smoothing
        p, y = np.deg2rad(pitch), np.deg2rad(yaw)
        self.front = np.array([np.cos(y) * np.cos(p), np.sin(y) * np.cos(p), np.sin(p)])
        self.center = self.radius = None

    def track(self, points: np.ndarray, min_radius: float = 0.0):
        center = 0.5 * (points.min(0) + points.max(0))
        radius = max(np.linalg.norm(points - center, axis=1).max(), min_radius)
        if self.center is None:
            self.center, self.radius = center, radius
        else:
            self.center += self.smoothing * (center - self.center)
            self.radius += self.smoothing * min(radius - self.radius, 0.0)  # zoom in only
        dist = self.zoom * self.radius / np.tan(np.deg2rad(0.5 * self.viewer.camera.fov))
        self.viewer.set_camera(wp.vec3(*(self.center - dist * self.front)), self.pitch, self.yaw)


def annotate(image: Image.Image, viewer, labels) -> Image.Image:
    """Draw each ``(text, world point, color)`` label, anchored at its point's projection, onto ``image``."""
    cam = viewer.camera
    pos = np.array(cam.pos)
    front, right, up = (np.array(v) for v in (cam.get_front(), cam.get_right(), cam.get_up()))
    alpha = np.tan(np.deg2rad(0.5 * cam.fov))
    font = ImageFont.load_default(size=max(12, image.height // 34))
    draw = ImageDraw.Draw(image)
    for text, point, color in labels:
        d = np.asarray(point) - pos
        depth = d @ front
        u = (d @ right) / (depth * alpha * cam.width / cam.height)
        v = (d @ up) / (depth * alpha)
        x, y = 0.5 * (u + 1.0) * image.width, 0.5 * (1.0 - v) * image.height
        fill = tuple(int(255 * (0.65 * c + 0.35)) for c in color)  # lightened to read on the sky
        draw.multiline_text((x, y), text, font=font, fill=fill, anchor="ld", spacing=2, stroke_width=1,
                            stroke_fill=(20, 22, 28))
    return image


def render(name: str, width: int, height: int, stride: int, frames: int | None):
    cls, default_frames, pitch, yaw, zoom, track_core = SCENES[name]
    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
    viewer.show_ui = False
    ex = cls(viewer, None)
    x = ex.state_0.particle_q.numpy()
    if pitch is not None and yaw is None:
        d = x[-1] - x[0]
        yaw = float(np.rad2deg(np.arctan2(d[1], d[0]))) + 90.0
    camera = Camera(viewer, pitch, yaw, zoom) if pitch is not None else None
    images = []
    for f in range(frames or default_frames):
        if f % stride == 0:
            x = ex.state_0.particle_q.numpy()
            if camera is not None:
                camera.track(core(x, ex.radius) if track_core else x, min_radius=8.0 * getattr(ex, "radius", 0.0))
            ex.render()
            image = Image.fromarray(viewer.get_frame().numpy())
            images.append(annotate(image, viewer, ex.labels) if hasattr(ex, "labels") else image)
        ex.step()
    path = f"docs/admm_{name}.webp"
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 * stride / ex.fps),
                   loop=0, quality=80, method=4)
    print(f"wrote {path} ({len(images)} frames)")
    viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=list(SCENES), choices=list(SCENES))
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--stride", type=int, default=2, help="render every n-th simulated frame")
    parser.add_argument("--frames", type=int, default=None, help="override the simulated frame count")
    args = parser.parse_args()
    if len(args.scenes) == 1:
        render(args.scenes[0], args.width, args.height, args.stride, args.frames)
    else:  # one headless GL context per process: later viewers in the same process render black
        for name in args.scenes:
            rest = [a for a in sys.argv[1:] if a not in ("--scenes", *SCENES)]
            subprocess.run([sys.executable, __file__, "--scenes", name, *rest], check=True)
