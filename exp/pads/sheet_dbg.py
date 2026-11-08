import sys
sys.path[:0] = ["tests", "examples", "exp/pads"]
import numpy as np
import newton
from press import pad, TOP, H, KH, grid, cloth
from dismech_newton import SolverDiSMech, flatten_state
g = float(sys.argv[1])
b = newton.ModelBuilder(gravity=(0.0, 0.0, -g))
pad(b)
v, f = grid(11, 11, 0.2, 0.2)
v += [-0.1, -0.1, TOP + H + 0.002 + 1.0e-4]
cloth(b, v, f, density=0.5, tri_ke=1.0e3, tri_ka=1.0e3, edge_ke=1.0e-3, particle_radius=0.002, damping=1.0e-2)
model = b.finalize(); model.particle_mu = 0.5
solver = SolverDiSMech(model, iterations=20)
s0, s1 = model.state(), model.state(); flatten_state(s0); flatten_state(s1)
for k in range(12):
    solver.step(s0, s1, None, None, 1 / 240)
    s0, s1 = s1, s0
    z = s0.particle_q.numpy()[:, 2] - (TOP + H + 0.002)
    print(k, f"z {z.mean()*1e6:.3f} um", solver.counts.numpy(), flush=True)
if len(sys.argv) > 2:
    orig = SolverDiSMech._search
    def search(self, state_in, q, f):
        orig(self, state_in, q, f)
        e = self._energy.numpy()[: self.num_candidates]
        print("  E", np.round(e - e[0], 9)[:8], "slope", self._slope.numpy()[:1], "t", self._impact.numpy()[:1],
              "pick", self._pick.numpy()[:1], "|dq|", np.abs(self._dq.numpy()).max(), "cnt", self.contact.count)
    SolverDiSMech._search = search
    s0, s1 = model.state(), model.state(); flatten_state(s0); flatten_state(s1)
    solver2 = SolverDiSMech(model, iterations=6)
    solver2.step(s0, s1, None, None, 1 / 240)
