"""1D model of the lagged proxy grip: MuJoCo finger (mass m, PD servo kp explicit, kd implicit = implicitfast)
pushing into a stiff wall that lives in our solve (penalty k_b), seen through a proxy of mass m_p.
Usage: finger1d.py  -> table of grip mean / 2-cycle amplitude for passes x proxy impedance."""
import numpy as np

h, m, kp, kd, S = 1 / 600, 0.029, 1000.0, 100.0, 5.0
target = S / kp  # wall at 0, servo at rest force S when the finger sits on the wall


def run(passes, mp, kb=1e5, steps=600, relax=1.0, z_imp=None):
    x, v, F = -0.002, 0.0, 0.0
    out = []
    for _ in range(steps):
        Fk = F
        for p in range(passes):
            # MuJoCo implicitfast: (m + h kd) v1 = m v + h (kp (target - x) + F)
            v1 = (m * v + h * (kp * (target - x) + Fk)) / (m + h * kd)
            x1 = x + h * v1
            # proxy: synced start x, end velocity v1, rewind of F with the proxy's mass
            pred = x + h * v1 - h * h * Fk / mp
            # our solve: mp/h^2 (y - pred) + kb max(y, 0) = 0
            y = pred if pred <= 0 else pred * (mp / h**2) / (mp / h**2 + kb)
            Fnew = mp * (y - pred) / h**2
            last_applied = Fk
            Fk = relax * Fnew + (1 - relax) * Fk
        x, v, F = x1, v1, Fk
        out.append(last_applied)
    a = np.array(out[-100:])
    return a.mean(), 0.5 * (a[::2].mean() - a[1::2].mean()), a.std()


m_eff = m + h * kd
print(f"m + h kd = {m_eff:.4f}")
for mp in (0.0855, m_eff, 0.029):
    for passes in (1, 2, 3, 4):
        mean, cyc, sd = run(passes, mp)
        print(f"mp {mp:.4f} passes {passes}: grip {-mean:7.3f}  2-cycle {cyc:+.3f}  std {sd:.3f}")
