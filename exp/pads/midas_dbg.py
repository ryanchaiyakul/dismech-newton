"""MIDAS axial slip with / without pads (DM_PADS): per frame the hand's push (MuJoCo), the rod's axial force (EA eps,
ours), mu sum N (sum mu rho), split over pad points and other pairs, and the pairs' count; means while sliding.
(post_substep is graph-captured: device sums, read per frame.)"""
import os
import sys

import numpy as np
import warp as wp

sys.path[:0] = [os.path.abspath("tests"), os.path.abspath("examples")]
from test_examples import _load  # noqa: E402

from dismech_newton.contact import Slots  # noqa: E402


@wp.kernel
def _split(s: Slots, out: wp.array[float]):
    """out: [mu rho pads, mu rho others, pad pairs, loaded pad pairs, other pairs], summed over substeps."""
    i = wp.tid()
    if s.active[i] == 0:
        return
    mr = s.mu[i] * s.rho[i]
    if s.pad[i][0] > 0.0:
        wp.atomic_add(out, 0, mr)
        wp.atomic_add(out, 2, 1.0)
        if s.rho[i] > 0.0:
            wp.atomic_add(out, 3, 1.0)
    else:
        wp.atomic_add(out, 1, mr)
        wp.atomic_add(out, 4, 1.0)


M = _load("midas_axial")
ex = M.Example(None, None)
acc = wp.zeros(5, dtype=float, device=ex.model.device)
post = ex.post_substep


def post_substep():
    post()
    s = ex.soft.contact.s
    wp.launch(_split, dim=s.active.shape[0], inputs=[s, acc])


ex.post_substep = post_substep
extra = []
for _ in range(162):
    acc.zero_()
    ex.step()
    extra.append(acc.numpy() / ex.substeps)
h = np.array(ex.history)
e = np.array(extra)
go = ex.sliding(h)
print(f"DM_PADS={os.environ.get('DM_PADS', '')}  frames {len(h)} sliding {go.sum()}")
print(f"push {h[go, 2].mean():.3f}  rod {h[go, 3].mean():.3f}  mu sum N {h[go, 4].mean():.3f}  "
      f"push/muN {np.mean(h[go, 2] / h[go, 4]):.3f}  rod/push {np.mean(h[go, 3] / h[go, 2]):.3f}")
print(f"per substep while sliding: mu rho pads {e[go, 0].mean():.3f}  others {e[go, 1].mean():.3f}  "
      f"pairs pad {e[go, 2].mean():.0f} (loaded {e[go, 3].mean():.0f})  others {e[go, 4].mean():.0f}")
