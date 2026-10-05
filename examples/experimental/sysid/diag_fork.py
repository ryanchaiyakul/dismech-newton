import numpy as np
import gen_baseline as g
R = np.radians
g.scan(configs=[
    ("rho .1, pitch .8", dict(DENSITY_SCALE=0.1, S_GRIP_R=R(0.8))),
    ("rho .1, pitch 1.0", dict(DENSITY_SCALE=0.1, S_GRIP_R=R(1.0))),
    ("rho .1, pitch .6, theta .5", dict(DENSITY_SCALE=0.1, S_GRIP_R=R(0.6), SOLVER_THETA=0.5)),
    ("rho .1, pitch .6, theta .5, q0 1mm", dict(DENSITY_SCALE=0.1, S_GRIP_R=R(0.6), SOLVER_THETA=0.5, S_Q0=1e-3)),
])
