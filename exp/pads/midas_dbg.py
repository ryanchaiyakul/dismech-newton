"""MIDAS axial slip with / without pads (DM_PADS): per frame the hand's push (MuJoCo), the rod's axial force (EA eps,
ours), mu sum N (sum mu rho), split over pad points and other pairs, and the pairs' count; means while sliding."""
import os
import sys

import numpy as np
import warp as wp

sys.path[:0] = [os.path.abspath("tests"), os.path.abspath("examples")]
from test_examples import _load  # noqa: E402

M = _load("midas_axial")
ex = M.Example(None, None)
s = ex.soft.contact.s
extra = []
post = ex.post_substep


def post_substep():
    post()
    act = s.active.numpy().astype(bool)
    pad = s.pad.numpy()[:, 0] > 0.0
    mr = s.mu.numpy() * s.rho.numpy()
    extra.append((ex.sim_time, mr[act & pad].sum(), mr[act & ~pad].sum(), (act & pad).sum(), (act & ~pad).sum(),
                  (act & pad & (s.rho.numpy() > 0)).sum()))


ex.post_substep = post_substep
for _ in range(162):
    ex.step()
h = np.array(ex.history)
e = np.array(extra)
go = ex.sliding(h)
print(f"DM_PADS={os.environ.get('DM_PADS', '')}  frames {len(h)} sliding {go.sum()}")
print(f"push {h[go, 2].mean():.3f}  rod {h[go, 3].mean():.3f}  mu sum N {h[go, 4].mean():.3f}  "
      f"push/muN {np.mean(h[go, 2] / h[go, 4]):.3f}  rod/push {np.mean(h[go, 3] / h[go, 2]):.3f}")
t0 = h[go, 0].min()
w = e[:, 0] >= t0
print(f"per substep while sliding: mu rho pads {e[w, 1].mean():.3f}  others {e[w, 2].mean():.3f}  "
      f"pairs pad {e[w, 3].mean():.0f} (loaded {e[w, 5].mean():.0f})  others {e[w, 4].mean():.0f}")
