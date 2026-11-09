# Part A: rigid impact and momentum under lagged one-pass coupling (2026-11-08)

Scripts: `exp/impact/` (model1d.py, model_fix.py, trace.py, passes.py, remote_gates.sh; logs on sci6000).

## Q1: the ~2-3x rebound is a double count, not the one-step delay

The coupler (lagged) syncs the source's START pose x and END velocity v1 = v + h (g + F_old/m), rewinds body_f to
-F_old - m g, so ProxyArm's pred = x + h v + h^2 g: the source's step WITHOUT the feedback F_old it is applying in that
same step. The contact then asks for the full stop again: F_old and F both stop the same overlap.

1-DOF model (hard / stiff penalty stop): lagged 2.3-2.7x rebound; a pure one-step delay (force computed on the
source's consistent next state, applied late) 0.00; pred from the proxy's own pose ("no dc") 1.00.
Real drop (test_proxy_impact scene, h 2 ms, mV/h 6.9 N): F = 0.06, 7.24, **14.64**, 7.96 N, rebound 1.36 on 0.45
m/s (~3x): the second step's force is 2 mV/h, as the model.

## Q2: 2 passes break momentum by accounting + pass-to-pass warm start

- The test's pending term `dt * feedback` is wrong for 2 passes: the box's final state took pass 1's force (applied
  by pass 2's MuJoCo step); the last pass's harvest is applied only in the next step's pass 1, which is discarded.
  Seed 0: 6.8e-2 with the pending term, 4.3e-3 without.
- The 4.3e-3 left = sum h (F_pass1 - F_pass2) exactly (4.32e-3 vs 4.33e-3): the rod takes pass 2's contact force,
  the box pass 1's, and they differ because our solver warm-starts pass 2 from pass 1 (iteration_restart only resets
  the proxy's start pose; multipliers / ADMM state carry). 2 passes: no rebound (0.002 m/s).

## Candidate fix: the proxy predicts the source's next step (DM_AHEAD, forward only, adjoint not updated)

pred = x1 + h v2', x1 = x + h v1 (the source's end pose, F_old in it), v2' = v1 + h g (1) or with the source's other
forces held at last step's, v2' = 2 v1 - v0 - h F_old/m (2). Harvest kept = our contact force. The proxy then runs a
step ahead of MuJoCo (its end pose = the source's next end pose).

| gate | lagged (now) | AHEAD=1 | AHEAD=2 |
|---|---|---|---|
| impact rebound | ~2-3x (xfail) | 0.002 m/s (XPASS) | 0.002 m/s (XPASS) |
| momentum (seed 0) | 6e-6 | 1.1e-5 | 9.9e-6 |
| pin gap / r (< 4e-4) | pass | 0.043 | 0.043 |
| static load, pick-place, fold, cross-slide | pass | pass | pass |
| MIDAS push / mu N (1 +- .02) | pass | -0.136 (broken) | 0.968 (0.92-1.00) |
| Franka swap grip (5 N, 2%) | pass | 4.83-4.84 | pass |
| trampoline (2 runs) | flaky 1/2 | 0.18 / 0.51 mm triangle in body | 2.8 mm / 0.29 mm |

Pin gap is measured to MuJoCo's pose, a step behind the proxy now (by construction); trampoline penetration is also
measured against MuJoCo's poses. AHEAD=1's MIDAS / swap failures = the servo's force left out of the prediction
(like staggered's grip -3.4%).

## Decision (user, 2026-11-08): parked

Accepted as a known limit for now (lagged x1, test_proxy_impact xfail); the tactile sensor first. DM_AHEAD lives in
commit d562640 (src reverted after). Before RL / sim2real with fast impacts: log contact-onset normal speed and onset force
overshoot in rollouts; if onsets reach tens of cm/s, develop AHEAD=2 (pin / trampoline measured vs the proxy, MIDAS
0.968, the adjoint's pred).
