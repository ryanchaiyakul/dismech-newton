"""Squeeze mode of the gripper (source: m + h kd = 0.196 kg, servo kp, implicitfast) on a rigid cable through a log
barrier in our solve, lagged proxy coupling with P passes; proxy mass m_p (view 0.0855 or exact 0.196); optional
Robin: the source takes F(x1) = F_ref + K (x1 - x_ref) implicitly (K = -b'' at our last end, x_ref our last end).
Also a free box (no servo) dropped on the barrier: rebound ratio."""
import numpy as np

h, m, kp, kd = 1 / 600, 0.029, 1000.0, 100.0
dh = 8e-5  # barrier activation distance [m]


def barrier(mp):
    kap = 1e3 * mp / h**2 * dh**2  # stiffness ~ 1e3 inertia/h^2 at d ~ dh/2

    def f(d):  # force on the finger (pushes back, -x) and its derivative wrt x (x = -d)
        if d >= dh:
            return 0.0, 0.0
        g = -kap * (2 * (d - dh) * np.log(d / dh) + (d - dh) ** 2 / d)  # db/dd (<= 0)
        H = -kap * (2 * np.log(d / dh) + 4 * (d - dh) / d - (d - dh) ** 2 / d**2)  # d2b/dd2 (> 0)
        return g, -H  # F = db/dd on +x finger pushing toward the wall at 0 (gap d = -x); dF/dx = -H
    return f


def solve(pred, mp, f, y0):
    """Our step: mp/h^2 (y - pred) = F(y), Newton from y0 (feasible, y < 0)."""
    y = y0
    for _ in range(100):
        F, dF = f(-y)
        r = mp / h**2 * (y - pred) - F
        J = mp / h**2 - dF
        dy = -r / J
        while y + dy >= 0:
            dy *= 0.5
        y += dy
        if abs(dy) < 1e-14:
            break
    return y, f(-y)


def run(passes, mp, robin, servo=True, steps=1200, x=-0.0005, v=0.0, mass=m, damp=kd):
    f = barrier(mp)
    S = 5.0
    target = S / kp if servo else 0.0
    F, y, K, ms = 0.0, x, 0.0, mass + h * damp
    out, xs, vs = [], [], []
    for _ in range(steps):
        free = lambda Fa: (mass * v + h * ((kp * (target - x) if servo else -9.81 * mass * 0) + Fa)) / ms
        if robin and K != 0.0:  # F = F_ref + K (x + h v(F) - y), v(F) = free(0) + h F / ms
            F = (F + K * (x + h * free(0.0) - y)) / (1 - K * h * h / ms)
        Fk = F
        for _ in range(passes):
            v1 = free(Fk)
            pred = x + h * v1 - h * h * Fk / mp
            y, (Fc, dF) = solve(pred, mp, f, min(y, -1e-9))
            Fapp, Fk = Fk, mp * (y - pred) / h**2
        x, v, F, K = x + h * v1, v1, Fk, dF
        out.append(Fapp), xs.append(x), vs.append(v)
    return np.array(out), np.array(xs), np.array(vs)


ms = m + h * kd
print("grip (servo 5 N), last 200 steps: mean, +- std, flip fraction")
for mp in (0.0855, ms):
    for robin in (False, True):
        for P in (1, 2):
            F, _, _ = run(P, mp, robin)
            a = -F[-200:]
            d = np.diff(a)
            flips = np.mean(np.sign(d[1:]) != np.sign(d[:-1])) if d.std() > 1e-9 else 0
            print(f"  m_p {mp:.4f} robin {robin!s:5} passes {P}: {a.mean():.4f} +- {a.std():.4f} flips {flips:.2f}")
print("free mass 0.1 kg at 0.3 m/s onto the barrier (no servo, no gravity): rebound / impact speed")
for robin in (False, True):
    for P in (1, 2):
        F, xs, vs = run(P, 0.1, robin, servo=False, steps=300, x=-0.0003, v=0.3, mass=0.1, damp=0.0)
        print(f"  robin {robin!s:5} passes {P}: rebound {-vs[-1] / 0.3:.3f}  deepest gap {(-xs).min() * 1e6:.1f} um")
