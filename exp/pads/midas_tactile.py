"""The MIDAS axial slip (examples/midas_axial.py, its test) with the PaXini fingertips mounted (utils.tactile.mount):
sensor surfaces as the fingertip contact shapes. Prints the shapes made and the test's numbers; FRAMES (default 162,
the test's). Data: $PAXINI."""
import os, sys
sys.path[:0] = ["examples", "tests"]
import newton, numpy as np
from utils import tactile

orig = newton.ModelBuilder.add_mjcf
def add_mjcf(self, *a, **k):
    r = orig(self, *a, **k)
    old = {lbl.split("/")[-1]: s for s, lbl in enumerate(self.shape_label)}
    for link, cap in tactile.CAPSULES.items():
        s = old[cap]; print(link, "old capsule xform", np.round(self.shape_transform[s], 4), "scale", self.shape_scale[s])
    made = tactile.mount(self)
    print("mounted", made, "shapes", len(self.shape_label))
    return r
newton.ModelBuilder.add_mjcf = add_mjcf
from test_examples import _load
ex = _load("midas_axial").Example(None, None)
for _ in range(int(os.environ.get("FRAMES", 162))):
    ex.step()
h = np.array(ex.history)
print("travel mm", round(1e3 * h[-1, 1], 2), "rod shift mm", round(1e3 * h[-1, 5], 3))
ex.test_final()
print("MIDAS TACTILE TEST PASSED")
