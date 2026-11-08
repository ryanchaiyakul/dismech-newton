# proxy-impact (2026-11-08): normal impulse through SolverCoupledProxy

Off ipc-contact's uncommitted state (its diff applied on fa9a553). Experiment switches (not for merging):
`DM_PROXY` in `coupling._proxy_update_kernel` (0 now; 1 = offset `dc` out of pred and velocity out; 2 = harvest
minus `M dc / h^2`), `DM_MODE` / `DM_ITER` in `dismech_newton/__init__.py` override every `SolverCoupledProxy.Proxy`
mode / `Config.iterations`. Scripts: `exp/step4/opts.sh`, `gates.sh`, `bounce_trace.py`; logs `exp/step4/logs/`.

## Mechanism (by hand, then confirmed)

Lagged = one-step-delayed deadbeat feedback. After the impact step MuJoCo has neither the stop's position (overshoot
h v) nor its velocity (the stop is pending). Next step the proxy re-stops MuJoCo's free motion: + m v / h from the
velocity, + m v / h from `dc`: total 3 m v, restitution ~2. Removing `dc` (opt 1) leaves the velocity half:
restitution ~1. A lagged proxy can't use its own state either: MuJoCo's position drifts by sum h^2 F / m.
Staggered syncs MuJoCo's end state, the force is applied in the interval it was computed for: MuJoCo lands on the
proxy's pose to h^2 g (gravity is rewound out of pred): monolithic implicit Euler. Cost: the soft side is a step
ahead of MuJoCo, and the proxy's pred lacks MuJoCo's own forces for the coming step (servo, gravity).

## Numbers

| | bounce ratio | pinned rest | test_coupling (9, incl. slow) | trampoline x4 |
|---|---|---|---|---|
| lagged (now) | 1.91 | flies off (z 1.22) | 9 pass (midas passed this run) | 3/4 (body 2 falls) |
| opt 1 | 1.05 | rests, f 0.28 / weight 0.31 | 6: pin gap 9e-4, pick-place grip 4.83/5, swap 4.81/5 | 2/4 (body 2 falls) |
| opt 2 | 2.05 | flies off (z 0.47) | not run | not run |
| staggered | 0.15 | pass (stop 6.93 N = m v / h) | 4: pin gap 0.043 (vs MuJoCo pose, a step behind), grips 4.83/5, swap, fold grip 1.66-2.38/2, midas 0.773 | 0/4 (3 non-finite, 1 penetration 2.5 mm) |
| staggered, iter 2 | | | momentum 0.27: broken | |

Bounce 0.15 (staggered, end-clamped rod) is the rod's elastic spring-back (no coupling force after the stop); fully
clamped: rebound 0.002 m/s, stop exactly m v / h.
The 3.4% grip deficit appears whenever `dc` leaves pred (opt 1 and staggered): the lagged offset is what carries the
servo's position into the proxy.

## Round 2 (user leans staggered; Newton's own levers)

Newton offers proxy iterations (+ fixed / Aitken relaxation), mass_scale, and SolverCoupledADMM (coupler-owned
contacts: bypasses OGC, not for us).

- Lagged + iterations 2: bounce 0.15, pinned exact (z 0.124079, f = weight); 4 / 8 / Aitken identical (2 passes
  converge here). sci6000: trampoline 4/4, suite 74 + 3 xfail but momentum tests fail (0.258, dp 0.021): partly the
  tests' pending `dt F` term (momentum test w/o it: 0.052 -> 0.0037, still > 1e-4: 2 passes don't converge a hard
  impact). Grips noisy: pick-place 4.74-5.30, swap 4.72-5.26 (servo 5). Costs 2x.
- Staggered + extrapolated source acceleration (`DM_EXTRAP=1`: a_own = (v_end - v_prev)/h - F_prev/m into pred,
  out of the harvest via the offset): bounce 0.15, pinned exact and MuJoCo == proxy at rest (0 um; was h^2 g = 39 um).
  test_coupling 6/9: fold grip + swap pass now; pick-place grip centred but noisy (4.82-5.03); pin gap 0.043 (the
  test compares the rod with MuJoCo's pose, a step behind); MIDAS NaN at its iterations 2, 0.25 at iterations 1
  (plain staggered 0.773). Likely: an articulated hand's link velocity change includes the other links' contact
  responses, not only F/m_eff: extrapolation feeds them back.
- `DM_EXTRAP=2` (gravity only): grips 4.83 as plain staggered.

## Round 3: lagged passes vs the grip (2026-11-08; grip_trace.py, sci6000 iters logs)

Iterations on the failing tests (PEND=0): momentum exact from 3 passes; MIDAS ok at 2-4; grips worse with passes
(x2 4.71-5.26, x3 4.74-5.38, x4 4.62-5.38, x4 Aitken 4.61-5.44). Trace (pick-place, held, 1/600 s): x1 5.00 +- 0.01;
x2 alternates step to step (-4.82 / -5.34 converged per step; per-pass slope 0.46). Not fixed by relaxation 0.5/0.7,
starting at MuJoCo's pose (DM_SYNC), offset out (DM_NODC: worse), mass_scale 2.9 (worse). Solves all converged.
Finger: model 0.029 kg, proxy 0.0855 kg, PD ke 1000 kd 100 implicitfast: step response ~0.2 kg + articulation.
Diagnosis: wrong interface impedance (partitioned-coupling literature). Next: scratch/handoffs/proxy-coupling-principled.md.

## Conclusion (round 1)

No option passes all gates. Staggered is the right impact physics but loses steady servo forces and breaks the
trampoline. Candidate next: staggered with MuJoCo's own acceleration extrapolated into pred
(a_own = (v_end - v_prev)/h - F_prev/m: exact for gravity and steady servos), or lagged with the pending impulse
applied (hybrid). Awaiting the user's call.

## Round 4: principled coupling, idea 1 (Schur proxy inertia) — 2026-11-08

Handoff: scratch/handoffs/proxy-coupling-principled.md. Switch `DM_SCHUR=full|body|diag` (`src/dismech_newton/
schur_exp.py`, installed from `__init__`; `DM_PASSES` overrides every coupler's passes): each coupled step MuJoCo is
stepped k+1 extra times from the same input (proxy forces F0, F0 + delta e_j) -> R = dv/dF per world, Minv_s = R / h
(J A^-1 J^T with articulation, implicit damping, equality constraints). ProxyArm: M = (Minv_s + eps Minv_view)^-1,
pred rewinds F with Minv_s, harvest M (z - pred) / h^2. contact.py arm loops over the world's full k block.

Per-pass slope by hand (lagged, rigid wall): F_new - F* = (1 - m_p / m_s) (F - F*), m_s the source's response mass
(finger: m + h kd = 0.196). View m_p = 0.0855 -> 0.56 (measured 0.46). 1D single-finger numpy model
(`exp/principled/finger1d.py`) never 2-cycles: the cycle needs the second finger.

Measured response, pick-place held at 3.6 s (`dump_schur.py`; bodies hand, left, right finger):
- finger y: Minv 3.01 (0.33 kg), view 0.0855 kg; **left-right y coupling -2.09** (the fingers' mimic equality):
  squeeze mode 3.01 + 2.09 = 5.1 -> **0.196 kg = m + h kd**, antisymmetric 0.92. Hand ~5 kg, rank ~10 of 18.
- probe asymmetry 16% (constraint nonlinearity / torque probes).

Grip trace (3.6-3.7 s, per finger |grip|, servo 5 N):

| variant | passes 1 | passes 2 |
|---|---|---|
| off (view mass) | 5.00 [4.99, 5.01] | 5.04 [4.71, 5.27] |
| body (6x6 blocks of Minv_s, eps .01) | 5.04 [4.38, 5.44], 2-cycle | 5.01 [4.51, 5.54] |
| diag | 5.05 [4.39, 5.45] | 5.01 [4.52, 5.48] |
| full eps .01 | stalls every solve, no grip | same |

Body/diag drop the cross term: squeeze mass 0.33 > 0.196 -> slope 1 - 0.33/0.196 = -0.7: alternation, as seen.
The view mass underestimates (slope +0.56: passes converge monotonically, x1 steady). Full needed; eps .01 gives
arm masses ~550 kg in rigid directions (float32 energies).

Probe noise is not the cause (full, eps .1, passes 1): delta 0.1 / 10 N, refresh every 1 / 60 / 600 steps all
[4.56-4.70, 5.28-5.32]; delta 0.01 loses the cable. Rewind by the unregularized Minv_s (`DM_SCHUR_REWIND=s`):
eps .03 / .1 / .3 x1 [4.55, 5.31] / [4.63, 5.26] / [4.70, 5.15], x2 [4.70, 5.33] / [4.72, 5.28] / [4.75, 5.14]
(eps .3 mean 4.95); proxy - MuJoCo drift grows with eps (35 / 50 / 100 um). Per-pass slope with full Schur ~ -0.25
(was +0.46), but the converged force still jumps +-0.4 N step to step: the jitter is in the converged answer, not
the passes. Open: is a +-0.3 N grip chatter physical with a stiff (true-impedance) finger on a moving cable
(monolithic reference, idea 7), and the drift. View-mass x1 (5.00 +- 0.01) may be smooth because a light, lagged
proxy filters.

## Round 5: Robin exchange (idea 3) — 2026-11-08

Chatter check (`chatter.py`, phases lift / carry / settle): the +-0.2 N grip chatter of lagged x2 and of every
Schur variant is the same in all phases, all in the squeeze mode (antisymmetric +-0.001), ~80% sign flips per step,
uncorrelated with the cable's acceleration: numerical (the squeeze mode is 0.196 kg on 1000 N/m: 11 Hz, not 300 Hz).

numpy squeeze model with the log barrier (`finger_robin.py`): reproduces the impact (lagged x1 rebound 2.4, x2
0.001); grip instability appears at x1 in the model but x2 in the scene (the scene's cycle needs something more:
adaptive load scale / cable). Robin fixes both in the model at x1.

`DM_ROBIN=1`: after our solve, A reassembled at the end, condensed on the arm (scipy): K = -(S - M/h^2) (contact
stiffness on the proxy DOFs, cable re-equilibrated). Before the passes the coupler applies F solving
F = F_ref + K (x0 + h v(F) - x_ref), v(F) = v_base + R (F - F0) (R from the probes; MuJoCo unchanged).

| variant | grip lift / carry / settle | bounce ratio (deepest) |
|---|---|---|
| lagged x1 (today) | 4.997 +- .002 / .003 / .003 | 1.91 (2.41 mm) |
| lagged x2 | 5.05 +- .17 / .18 / .19 | 0.15 |
| **Robin, view mass, x1** | **4.917 +- .001 / .002 / .002** | **0.21 (0.36 mm)** |
| Robin, view mass, x2 | 5.04 +- .02 / .09 / .16 | 0.15 (0.19 mm) |
| Robin + full Schur x1 / x2 | 2.3 / 6.9 (wrong) | 0.23 |

Robin x1 grip 4.917: servo 1000 N/m x (5 mm - gap): the finger rests 83 um into the barrier's 100 um margin
(SMOOTH 0.02 x radius), MuJoCo pose == proxy pose at steady state; today's 5.00 carries the lagged offset.

### Round 5 gates (branch exp/proxy-robin; proxy-impact worktree was taken over by another session at 13:03)

Probe-free Robin (`DM_ROBIN=2`, in our harvest): v_free = 2 v1 - v0 - h Minv F_app (the source's own acceleration
carried over; the first try v1 - h Minv F_app was wrong at impacts: rebound 1.07). K from the step's last factorized
matrix (`DM_ROBIN_K=chord`) == end reassembly (bounce 0.21, grip 5.006 +- .003): k back-solves, no extra factor.
Closed form: F = (M/h^2) [h X M (2 v1 - v0) - (pred - z) - h (2 v1 - v0)], X = (A^-1)_zz (no inverse).

| gate | Robin + probes | Robin probe-free | Robin probe-free + debt |
|---|---|---|---|
| bounce ratio | 0.21 | 0.21 | 0.39 |
| pick-place grip | 4.917 +- .002 | 5.006 +- .003 | 4.998 +- .004 |
| momentum (1e-4) | 1.03 | 1.03 | 0.20 |
| pin dp (1e-4) | 0.785 | 0.787 | 0.021 |
| napkin fold grip (2 N) | pass | 1.81-2.20 fail | - |
| MIDAS | NaN | push/mu 0.54 (-0.29..2.6) | - |
| trampoline | 0/4 | 3/4 (body 2: known) | - |
| impact, static, pick-place, cross-slide | pass | pass | - |
| suite | 75 + 2 failed (momentum, pin) | same | - |
| Franka swap | capture error (experiment's host numpy) | same | - |

Momentum: Robin hands the source a prediction of the next contact force; prediction errors accumulate (no
telescoping). Debt (D += F_app - F_ref, source gets R - D) bounds the imbalance to the last step's error (20% of p0
at the rod impact) and costs impact (0.39).

### Round 6: Robin iterated within the step (DM_ROBIN=2, repeat passes solve x1(F) = x1 + h^2 Minv (F - F_app))

| gate | x2 | x3 | x4 |
|---|---|---|---|
| momentum (1e-4) / pin | 0.0029 / pass | pass / pass | |
| napkin fold / MIDAS | pass / pass | | |
| bounce ratio | 0.15 (deepest 0.19 mm) | | |
| pick-place grip | 5.05 +- .19 | 5.05 +- .27 | 5.05 +- .31 |

Pick-place grip chatter grows with passes (as plain lagged x2): the converged interface itself oscillates in the
squeeze mode in this scene; one pass filters it. MuJoCo restart checked (`restart_check.py`): each pass does start
from the restored state (SolverMuJoCo re-reads joint_q each step); the finger velocity flips sign pass to pass.
Open: why the converged Franka-finger/cable interface 2-cycles (mimic equality softness? barrier stiffness scaled by
the view mass while the finger's response mass is 0.196?).
