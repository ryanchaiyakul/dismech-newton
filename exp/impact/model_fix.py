"""1-DOF: lagged pred vs 'ahead' (predict the source's next step: pred = x1 + h (v1 + (v1 - v0) - h F_old / m), its
other forces held at last step's) on (a) a drop on a hard / penalty stop, (b) a PD servo pressing into a stop
(the grip): steady force vs the servo's, oscillation amplitude. Stop: penalty kpen m / h^2 (stiff barrier) or a
soft Winkler pad k_pad."""
import numpy as np

def sim(mode, servo=False, V=0.44, h=2e-3, g=-9.81, m=0.032, steps=1500, kstop=None):
    kp, kd, xt = 50.0, 2.0, -0.002  # servo pulling 2 mm into the stop: kp * 2 mm = 0.1 N + gravity
    x, v, y, F_old, v0 = (0.0, 0.0, 0.0, 0.0, 0.0) if servo else (5 * V * h, -V, 5 * V * h, 0.0, -V)
    if not servo: g_eff = g
    hist = []
    for k in range(steps):
        a_oth = (-kp * (x - xt) - kd * v) / m if servo else 0.0
        v1 = v + h * (g + a_oth + F_old / m); x1 = x + h * v1
        if mode == "lagged":
            pred = x + h * v1 - h * h * F_old / m
        elif mode == "ahead":
            pred = x1 + h * (v1 + (v1 - v0) - h * F_old / m)
        elif mode == "ahead0":  # staggered-like: no other-force extrapolation
            pred = x1 + h * (v1 + h * g)
        F = stop(pred, m, h, kstop)
        x, v0, v, F_old = x1, v1, v1, F
        hist.append((x, v, F))
    return np.array(hist)

def stop(pred, m, h, kstop):
    k = 1e3 * m / (h * h) if kstop is None else kstop
    return -k * pred / (1 + k * h * h / m) if pred < 0 else 0.0  # implicit penalty on the proxy

for ks, name in ((None, "barrier"), (2e3, "pad")):
    for mode in ("lagged", "ahead0", "ahead"):
        d = sim(mode, kstop=ks); vmin = d[:, 1].min(); k0 = d[:, 1].argmin()
        s = sim(mode, servo=True, kstop=ks)[-200:, 2]
        want = 50.0 * 0.002 + 0.032 * 9.81  # servo pull at the stop surface... (+ sinkage correction below)
        print(f"{name:7s} {mode:7s} drop rebound/impact {d[k0:, 1].max() / -vmin:5.2f} | grip F {s.mean():.4f} "
              f"+- {s.std():.4f} N")
