# Hydroelastic pads (exp/hydro-pads, off exp/proxy-robin b45a3ff), 2026-11-08

Handoff: scratch/handoffs/hydroelastic-coupling.md (exp/HANDOFF_hydroelastic.md).

## A1: barrier stiffness under iterated Robin x3 (DM_ROBIN=2, 3 passes; exp/principled/remote_stiff.sh)

| barrier | pick-place grip lift / carry / settle | flips | bounce | momentum / pin | fold |
|---|---|---|---|---|---|
| STIFFNESS 1e3 | 5.05 +- .23 / .28 / .27 | 0.84-0.94 | 0.15 | pass | corner 4.4 cm (flaky) |
| 1e2 | 5.00 +- .07 / .03 / .01 | 0.65-1.0 | 0.16 | 1 fails (0.0025) | grip 1.6-5.0 N (servo 2) |
| 1e1 | 4.99 +- .006 / .007 / .038 | 0.86-0.98 | 0.23 | 1 fails (0.025) | grip 1.8-3.8 N |
| 1e3, arm flex x 0.436 (squeeze mass) | 5.05 +- .26 | 0.83-0.90 | 0.15 | pass | pass |

The converged chatter is the interface stiffness (x40 less at 1e1), not the view mass. A softer *barrier* breaks
momentum and the fold grip, so it is not the cure by itself.

## B: pad potential (src: contact.py, collide.py, adjoint.py one line; ~80 lines)

- Pad = a shape with Newton's `is_hydroelastic` flag; `kh` [N/m^3] = E / h its pressure per depth; its margin the pad
  thickness h over a rigid core (Newton asks such a shape for an SDF: `sdf_max_resolution=8`, unused).
- Pairs: every node within reach of a pad holds a pair (the quadrature: rod node half its edges, shell node a third
  of its faces, `Mesh.area`); rod edge / shell interior pairs skipped against pads.
- Energy per pair: core barrier (thickness to the core) + `c delta^p`, delta = depth of the node's surface in the
  pad: rod `kh L (8/15) sqrt(2R) delta^(5/2)` (Winkler cylinder, integrated across the section), shell
  `kh A delta^2 / 2`. Friction's lagged normal force includes it.
- Line search: an untouched pad stops the step at its surface (as the barrier's onset); without it a sheet at >= 10 g
  stalled (Newton's A had no pad curvature, the shortest candidate still overshot).
- Switch `DM_PADS=kh,h,pattern` (experiment): pads the shapes of bodies whose label contains pattern.

### Gates 1-2 (exp/pads/press.py): settled depth vs Winkler, gravity 1-300 g

| | 9.8 | 98 | 981 | 3000 |
|---|---|---|---|---|
| rod R 5 mm, kh 3e7 (um, sim / exact) | 33.356 / 33.353 | 154.80 / 154.81 | 718.56 / 718.57 | 1513.91 / 1513.91 |
| sheet (um) | 0.164 / 0.164 | 1.635 / 1.635 | 16.350 / 16.350 | 50.001 / 50.000 |

Not done: dissipation (Hunt-Crossley), moment/quadrature convergence (gate 3), incline (5), tactile readout (6).
