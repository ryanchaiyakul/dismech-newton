"""The MIDAS axial slip with the PaXini fingertips (utils.tactile) read at the sensor's 83.3 Hz: a frame = one
sensor period = 6 steps of 2 ms = one CUDA graph (steps, per-step commits, the sample). Per frame: each sensor's
contact force (sensor frame) and its taxels' sum (turned back from their frames; equal by construction: a check of
the binning), the counts; the example's test at the end. Saves $OUT (npz). Data: $PAXINI."""
import os
import sys
import time

sys.path[:0] = ["examples", "tests"]
import newton  # noqa: E402
import numpy as np  # noqa: E402
from test_examples import _load  # noqa: E402
from utils import tactile  # noqa: E402

made = {}
orig = newton.ModelBuilder.add_mjcf


def add_mjcf(self, *a, **k):
    r = orig(self, *a, **k)
    made.update(tactile.mount(self))
    return r


newton.ModelBuilder.add_mjcf = add_mjcf
Base = _load("midas_axial").Example


class Example(Base):
    fps, substeps = 1000.0 / 12.0, 6  # one frame per sensor sample (12 ms), 2 ms steps

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.tactile = tactile.Tactile(self.model, self.soft, made)
        self.tactile.hook()
        self.log = []

    def post_substep(self):
        super().post_substep()
        self.tactile.commit(self.sim_dt)

    def simulate(self):
        super().simulate()
        self.tactile.sample()

    def step(self):
        super().step()
        t = self.tactile
        f, raw, tot = t.force.numpy(), t.raw.numpy(), t.total.numpy()
        F = np.array(t.frame.numpy())
        back = np.einsum("tji,tj->ti", F, f * [1.0, 1.0, -1.0])  # taxel frame -> sensor frame
        sums = np.array([back[t.start[k]:t.start[k + 1]].sum(0) for k in range(len(t.links))])
        self.log.append((self.sim_time, f.copy(), raw.copy(), tot.copy(), sums))


frames = int(os.environ.get("FRAMES", 225))  # the test's 2.7 s
ex = Example(None, None)
t0 = time.perf_counter()
for i in range(frames):
    ex.step()
    if i == 1:
        t1 = time.perf_counter()
wall = time.perf_counter() - t1
h = np.array(ex.history)
tot = np.array([r[3] for r in ex.log])
sums = np.array([r[4] for r in ex.log])
err = np.abs(sums - tot).max() / max(np.abs(tot).max(), 1e-12)
print(f"frames {frames}, wall per frame (graph) {1e3 * wall / (frames - 2):.1f} ms, links {ex.tactile.links}")
print(f"taxel sums vs sensor contact force: worst {err:.2e} (of the largest force)")
for k, link in enumerate(ex.tactile.links):
    n = tot[:, k, 2]
    raw = np.array([r[2][ex.tactile.start[k]:ex.tactile.start[k + 1]] for r in ex.log])
    print(f"{link}: |F| max {np.linalg.norm(tot[:, k], axis=1).max():.3f} N, normal (sensor -z) max {-n.min():.3f} N, "
          f"taxels lit max {(raw[..., 2] > 0).sum(1).max()}, max Fz count {raw[..., 2].max()}")
print("travel mm", round(1e3 * h[-1, 1], 2), "rod shift mm", round(1e3 * h[-1, 5], 3))
np.savez(os.environ.get("OUT", "exp/pads/logs/midas_readout.npz"), time=[r[0] for r in ex.log],
         force=np.array([r[1] for r in ex.log]), raw=np.array([r[2] for r in ex.log]), total=tot, sums=sums,
         start=ex.tactile.start, links=ex.tactile.links, history=h)
ex.test_final()
print("MIDAS READOUT TEST PASSED")
