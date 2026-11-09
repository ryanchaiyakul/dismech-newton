"""1-DOF model of SolverCoupledProxy's lagged exchange at a rigid stop (box onto a clamped rod).

Source (MuJoCo, semi-implicit Euler) applies the feedback harvested last step:
    v1 = v + h (g + F_old / m),  x1 = x + h v1.
Destination (us) per step, after the source: the proxy's predicted end pose from the coupler's sync (source start
pose x, end velocity v1, rewound body_f = -F_old - m g):
    pred = x + h v1 + h^2 (-F_old / m)  =  x + h v + h^2 g,
our implicit contact solve pushes the proxy to y = pred + h^2 F / m (F the contact force on it; hard stop: y >= 0, or
a stiff penalty k (= STIFFNESS m / h^2) on y < 0), and the harvest m ((y - x)/h - v1)/h + F_old = F is applied next
step. Variants:
  lagged   : the above (the coupler's default, ours now).
  own      : pred from the proxy's own end pose y (no offset dc = x - y): pred = y + h v + h^2 g.
  delay    : the stop computed on the source's true next state (pred = x + h v1, F_old included) and applied a step
             late: the plain one-step delay of an explicit exchange.
  stagger  : sync the source's END pose x1 (staggered): pred = x1 + h(v1 - ...) ... = implicit, F applied now.
"""
import numpy as np

def run(mode, stop="hard", V=0.44, h=2e-3, g=-9.81, m=1.0, steps=200, kpen=1e3):
    x, v, y, F_old = 5 * V * h, -V, 5 * V * h, 0.0
    vmin = vmax = 0.0; out = []
    for k in range(steps):
        if mode == "stagger":  # implicit: the stop acts on this step's source
            pred = x + h * v + h * h * g
            F = contact(pred, m, h, stop, kpen)
            v = v + h * (g + F / m); x = x + h * v
        else:
            v1 = v + h * (g + F_old / m); x1 = x + h * v1
            if mode == "lagged":
                pred = x + h * v + h * h * g
            elif mode == "own":
                pred = y + h * v + h * h * g
            elif mode == "delay":
                pred = x1 + h * v1 + h * h * g  # our solve on the source's end state; force lands one step later
            F = contact(pred, m, h, stop, kpen)
            y = pred + h * h * F / m
            x, v, F_old = x1, v1, F
        vmin, vmax = min(vmin, v), max(vmax, v)
        out.append((x, v, F_old))
    return vmax / -vmin, np.array(out)

def contact(pred, m, h, stop, kpen):
    if pred >= 0: return 0.0
    if stop == "hard": return -m * pred / (h * h)
    k = kpen * m / (h * h)  # penalty: y = pred + h^2 k (-y) / m  ->  y = pred / (1 + kpen)
    return -k * pred / (1 + kpen)

if __name__ == "__main__":
    for stop in ("hard", "pen"):
        for mode in ("lagged", "own", "delay", "stagger"):
            r, tr = run(mode, stop)
            print(f"{stop:5s} {mode:8s} rebound/impact {r:.3f}")
    r, tr = run("lagged")
    for k in range(3, 10):
        print(k, f"x {tr[k,0]*1e3:+.3f}mm v {tr[k,1]:+.3f} F {tr[k,2]:+.1f}")
