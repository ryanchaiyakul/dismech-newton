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

## Conclusion (round 1)

No option passes all gates. Staggered is the right impact physics but loses steady servo forces and breaks the
trampoline. Candidate next: staggered with MuJoCo's own acceleration extrapolated into pred
(a_own = (v_end - v_prev)/h - F_prev/m: exact for gravity and steady servos), or lagged with the pending impulse
applied (hybrid). Awaiting the user's call.
