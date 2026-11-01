"""Four cantilevers driven through slow one-sided cycles, elastic against Dahl hysteresis (Newton's
``vbd/example_cable_dahl_hysteresis``): the same cables, stiffness, drive and timing. Two take a transverse tip force
ramped 0 -> 3 N -> 0 twice, two a prescribed tip twist 0 -> 45 deg -> 0 twice; one of each pair has Dahl friction
(``add_rod``'s ``dahl_eps_max`` 0.1 rad, ``dahl_tau`` 0.05 rad). Newton's force is 0.5 N, but its VBD bends these
cables 6.6x further than their stiffness says (0.146 m elastic; the discrete cantilever's is 0.022 m, and its twist
agrees with ours): 3 N gives Newton's deflections, curvatures that reach Dahl's ``tau``.

Exact curves (no inertia to speak of, deflection under 10% of the length): a twisted cable twists uniformly, each
triplet by the tip's twist over the 15 triplets, so its reaction ``K tau + sigma`` follows the closed-form Dahl loop
(branches exponential toward ``+-sigma_max = +-K eps_max`` at rate ``1/tau``); a bent cable's hinges are independent,
``K kappa + sigma + C dkappa/dt = a F`` with ``a`` the hinge's lever arm, and the tip deflects ``sum b kappa`` (``b``
the hinge's distance to the tip). Within 2% (0.6%, 0.5% and 0.05%, 2026-11-01). Newton's checks besides: the Dahl
cable deflects less, its loop is fatter and it keeps a set after unloading (10% of its peak; Newton's 13%); the Dahl
twist loop is larger and keeps a reaction at zero twist; bend and twist do not leak into each other.

    uv run examples/newton/cable_dahl_hysteresis.py
    uv run examples/newton/cable_dahl_hysteresis.py --viewer null --test
"""

import sys
from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/: utils

from utils.common import MUTED, SIM, THEORY, CableExample, Drive, inset, inset_scale, segment_dofs

from dismech_newton import DiSMechSolver


def dahl_advance(d, sigma, sigma_max, tau):
    """Newton's Dahl law over a strain change ``d`` along one branch (as :func:`dismech_newton.rod.dahl_advance`)."""
    s = np.sign(d) if abs(d) > 1.0e-6 else (-1.0 if sigma < 0.0 else 1.0)
    e = np.exp(-s * d / tau)
    return s * sigma_max * (1.0 - e) + sigma * e


class Example(CableExample):
    substeps = 10
    num_elements, segment_length, radius = 16, 0.1, 0.01
    stiffness, damping = 250.0, 75.0
    force_max, twist_max = 3.0, np.radians(45.0)
    eps_max, tau = 0.10, 0.05
    phase, phases, settle = 3.0, 4, 2.0
    cases = {"bend elastic": -0.72, "bend Dahl": -0.42, "twist elastic": 0.18, "twist Dahl": 0.48}

    def __init__(self, viewer, args=None):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        self.length = self.num_elements * self.segment_length
        self.edges = {}
        for name, y in self.cases.items():
            rod = newton.Rod.create_straight(start=wp.vec3(0.0, y, 0.0), direction=wp.vec3(1.0, 0.0, 0.0),
                                             length=self.length, segment_count=self.num_elements, radius=self.radius)
            dahl = dict(dahl_eps_max=self.eps_max, dahl_tau=self.tau) if "Dahl" in name else {}
            bodies = DiSMechSolver.add_rod(builder, rod, stretch_stiffness=1.0e6, bend_stiffness=self.stiffness,
                                           twist_stiffness=self.stiffness, bend_damping=self.damping,
                                           twist_damping=self.damping, collide=False, **dahl)
            DiSMechSolver.fix_segment(builder, bodies[0])
            if name.startswith("twist"):
                DiSMechSolver.fix_segment(builder, bodies[-1])  # the tip's pose prescribed, its twist driven
            self.edges[name] = bodies
        model = builder.finalize()
        self.start(viewer, model, DiSMechSolver(model, self_contact=False))
        twisted = [b[-1] for name, b in self.edges.items() if name.startswith("twist")]
        self.twist = Drive(model, [d for b in twisted for d in segment_dofs(model, b, True)])
        self.drives = (self.twist,)
        nodes, triplets = self.num_elements + 1, self.num_elements - 1
        self.nodes = {name: np.arange(k * nodes, (k + 1) * nodes) for k, name in enumerate(self.cases)}
        self.triplets = {name: np.arange(k * triplets, (k + 1) * triplets) for k, name in enumerate(self.cases)}
        self.force = np.zeros((model.particle_count, 3), dtype=np.float32)
        self.rest = model.particle_q.numpy()
        self.history = {"force": [], "twist": [], **{name: [] for name in self.cases}}
        self.leak = dict.fromkeys(self.cases, 0.0)
        self.drift = 0.0

    def drive_at(self, t):
        """The cycle's 0 -> 1 -> 0 triangle wave, 0 after it."""
        if t >= self.phases * self.phase:
            return 0.0
        i, frac = divmod(t / self.phase, 1.0)
        return frac if int(i) % 2 == 0 else 1.0 - frac

    def drive(self, t0, t1):
        a0, a1 = self.drive_at(t0), self.drive_at(t1)
        self.twist.set(np.full(2, self.twist_max * a0), np.full(2, self.twist_max * a1))
        self.force[:] = 0.0
        for name in ("bend elastic", "bend Dahl"):  # at the tip segment's midpoint, as on Newton's tip body
            self.force[self.nodes[name][-2:], 2] = -0.5 * self.force_max * a1
        for state in (self.state_0, self.state_1):
            state.particle_f.assign(self.force)

    def step(self):
        super().step()
        self.record()

    def record(self):
        x = self.state_0.particle_q.numpy()
        s = self.state_0.dismech
        strain, sigma = s.triplet_strain_q.numpy(), s.triplet_dahl_q.numpy()
        self.history["force"].append(self.force_max * self.drive_at(self.sim_time))
        self.history["twist"].append(self.twist_max * self.drive_at(self.sim_time))
        for name, n in self.nodes.items():
            t = self.triplets[name]
            if name.startswith("bend"):
                self.history[name].append(self.rest[n[-1], 2] - x[n[-1], 2])
                self.leak[name] = max(self.leak[name], np.abs(sigma[t, 2]).max(), np.abs(strain[t, 4]).max())
            else:  # Newton's reaction: the triplets' mean stress
                self.history[name].append(np.mean(self.stiffness * strain[t, 4] + sigma[t, 2]))
                self.leak[name] = max(self.leak[name], np.abs(sigma[t, :2]).max(), np.abs(strain[t, 2:4]).max())
                self.drift = max(self.drift, np.linalg.norm((x[n] - self.rest[n])[:, 1:], axis=1).max())

    def bend_theory(self, dahl: bool, substeps: int = 100):
        """The tip deflection of independent hinges along the recorded force (piecewise constant per frame)."""
        x = self.segment_length * np.arange(1, self.num_elements)  # the hinges
        a, b = self.length - 0.5 * self.segment_length - x, self.length - x
        k, c, smax = self.stiffness, self.damping, self.stiffness * self.eps_max if dahl else 0.0
        kappa, sigma, h, out = np.zeros_like(x), np.zeros_like(x), self.frame_dt / substeps, []
        for f in self.history["force"]:
            for _ in range(substeps):
                rate = (a * f - k * kappa - sigma) / c
                if dahl:
                    sigma += (np.sign(rate) * smax - sigma) / self.tau * np.abs(rate) * h
                kappa += rate * h
            out.append(b @ kappa)
        return np.array(out)

    def twist_theory(self):
        """The reaction of a uniformly twisted Dahl cable along the recorded drive."""
        tau, sigma, out = 0.0, 0.0, []
        for twist in self.history["twist"]:
            new = twist / (self.num_elements - 1)
            sigma = dahl_advance(new - tau, sigma, self.stiffness * self.eps_max, self.tau)
            tau = new
            out.append(self.stiffness * tau + sigma)
        return np.array(out)

    @staticmethod
    def area(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return 0.5 * abs(x @ np.roll(y, -1) - np.roll(x, -1) @ y)

    def metrics(self):
        n = min(len(self.history["force"]), int(round(self.phases * self.phase * self.fps)))
        out = {}
        for name in self.cases:
            y = np.array(self.history[name])
            drive = self.history["force" if name.startswith("bend") else "twist"]
            out[name] = {"max": np.abs(y[:n]).max(), "residual": abs(y[-1]), "area": self.area(drive[:n], y[:n])}
        return out

    def image(self, size=(800, 400)):
        th = self.twist_theory()

        def draw(axes):
            bend, twist = axes
            lw = 1.2 * inset_scale(size, 2)
            for name, dahl, color in (("bend elastic", False, MUTED), ("bend Dahl", True, SIM)):
                bend.plot(self.history["force"], 100.0 * np.array(self.history[name]), color=color, label=name[5:])
                bend.plot(self.history["force"], 100.0 * self.bend_theory(dahl), color=THEORY, ls="--", lw=lw)
            bend.set_xlabel(r"$F$ (N)")
            bend.set_ylabel(r"$w$ (cm)")
            bend.legend(loc="upper left")
            deg = np.degrees(self.history["twist"])
            twist.plot(deg, self.history["twist Dahl"], color=SIM, label="Dahl")
            twist.plot(deg, self.history["twist elastic"], color=MUTED, label="elastic")
            twist.plot(deg, th, color=THEORY, ls="--", lw=lw, label="exact")
            twist.set_xlabel(r"$\Theta$ (deg)")
            twist.set_ylabel(r"$M$ (N m)")
            twist.legend(loc="upper left")

        return inset(draw, size, ncols=2)

    def render(self):
        if self.frame % 30 == 0:
            self.viewer.log_image("hysteresis", self.image())
        super().render()

    def test_final(self):
        assert np.isfinite(self.state_0.particle_q.numpy()).all(), "non-finite positions"
        if self.sim_time < self.phases * self.phase + 0.5 * self.settle:
            return
        m = self.metrics()
        be, bd, te, td = (m[name] for name in self.cases)
        assert 0.03 < bd["max"] < be["max"] < 0.3 * self.length, f"peak deflections: {be}, {bd}"
        assert bd["area"] / bd["max"] ** 2 > be["area"] / be["max"] ** 2, f"Dahl's loop not fatter: {be}, {bd}"
        assert be["residual"] < 0.05 * be["max"], f"elastic bend kept a set: {be}"
        assert bd["residual"] > max(0.05, 3.0 * be["residual"] / be["max"]) * bd["max"], f"no Dahl set: {bd}"
        assert td["area"] > 2.0 * te["area"], f"Dahl twist loop too small: {te}, {td}"
        assert te["residual"] < 0.25 and td["residual"] > max(1.0, 5.0 * te["residual"]), f"twist residual: {te}, {td}"
        assert self.drift < 0.002 * self.length, f"the twisted centerlines moved {self.drift:.2e} m"
        assert max(self.leak.values()) < 1.0e-4, f"bend and twist leaked into each other: {self.leak}"
        for name, th in (("bend elastic", self.bend_theory(False)), ("bend Dahl", self.bend_theory(True)),
                         ("twist Dahl", self.twist_theory())):
            err = np.abs(np.array(self.history[name]) - th).max() / np.abs(th).max()
            assert err < 0.02, f"{name} off the exact curve by {err:.2%}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(CableExample.fps * (Example.phases * Example.phase + Example.settle)))
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
