import sys
sys.path[:0] = ["tests", "examples", "exp/pads"]
import numpy as np
import tilt
from dismech_newton import SolverDiSMech
orig = SolverDiSMech._search
k = [0]
def search(self, state_in, q, f):
    orig(self, state_in, q, f)
    if k[0] < 12:
        e = self._energy.numpy()[: self.num_candidates]
        print("  E", np.round(e - e[0], 9), "slope", self._slope.numpy()[:1], "t", self._impact.numpy()[:1],
              "pick", self._pick.numpy()[:1], "|dq|", np.abs(self._dq.numpy()).max(), "lim", self.contact.source.limit.numpy().min(), flush=True)
    k[0] += 1
SolverDiSMech._search = search
print(tilt.sim(int(sys.argv[1]))[:2])
