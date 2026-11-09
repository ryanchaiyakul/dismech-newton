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

### Gate 3 (exp/pads/tilt.py): stiff rod, off-centre point load, tilt vs exact (Winkler, exact integral)

| nodes | delta0 err | theta err |
|---|---|---|
| 6 | 1.5e-3 | 8.1e-2 |
| 11 | 7.1e-4 | 2.4e-2 |
| 21 | 3.3e-4 | 7.9e-3 |
| 41 | 1.9e-4 | 3.4e-3 |

(n = 81: the float32 solve returns dq = 0 from step 0: stretch 1e7 + EI/le 1e4 at 5 mm edges, a setup limit.) Added
`PAD_FLOOR`: A holds a pad pair's curvature at least at 0.05 h deep (a rod's delta^1/2 is 0 at the surface).

### Gate 7 preview: Franka finger pads (DM_PADS on 12 finger shapes), pick-place grip (servo 5 N)

| variant | pad | grip lift / carry / settle | flips | fold | pick test |
|---|---|---|---|---|---|
| Robin x3, no pad (A1 row 1) | - | 5.05 +- .23 / .28 / .27 | 0.84-0.94 | | |
| Robin x3 | out, kh 5e8, h 2 mm | 6.38 +- .04 / 5.71 +- .10 / 0.42 | 0.66 / 0.41 / 0 | grip 3.5-4.5 (servo 2) | |
| Robin x1 | out, 5e8 | 6.36 +- .04 / 5.67 +- .11 / 0.49 | 0.68 / 0.36 / 0 | napkin into table | |
| Robin x3 | out, 5e7 | 5.86 +- .03 / 5.26 +- .10 / 0.32 | 0.66 / 0.28 / 0 | grip 3.7-3.9 | |
| lagged x1 | out, 5e8 | 6.39 +- .04 / 5.75 +- .10 / 0.44 | 0.66 / 0.41 / 0 | napkin into table | grip 1.5-6.3 |
| Robin x3 | in, kh 1e9, h 1 mm | 3.86 +- .14 / 1.04 +- .22 / 0.01 | 0.03 / 0 | grip 0.5-36 | grip 0.02-3.4 |
| Robin x1 | in | 3.89 +- .14 / 1.20 +- .23 / 0.01 | 0.02 / 0 | grip 0.9-18 | grip 0.05-3.5 |
| lagged x1 | in | 3.80 +- .15 / 0.95 +- .20 / 0.01 | 0 / 0 | grip 0-27 | grip 0-3.3 |

(out: the pad 2 mm outside the finger's surface, its geometry shifts the servo's stop; in: the pad inside the contact
thickness, same geometry.) Momentum/pin have no fingers (unchanged). Suite without pads: same as b45a3ff (75 + the
2 coupling momentum/pin failures of this branch's default env).

**The 2-cycle is gone with pads in every coupling variant, even lagged x1** (flips 0-0.4, the barrier's 0.84-0.94):
the interface stiffness was the cause. **But the grip is wrong**: the cable's nodes are 2 cm apart (0.38 m / 19), about
a finger pad's width, and pads take only node quadrature points (edge interiors skipped), so 0-1 cable node is in a
pad: the cable slips out (settle ~0). Next: quadrature along rod edges / over shell triangles against pads
(Drake's resolution hint: points per edge, weights (1-u, u)), then rerun.

## Round 2 (after reading Elandt et al. 2019 / Drake docs): Drake's form

- Drake: pressure 0 on the shape's own surface, E at the medial axis or a rigid core; bodies interpenetrate the
  undeformed shape. Its `margin` inflates only for detection (pressure negative there: speculative constraints, no
  force). So the "out" variant (pad outside the finger) was an artifact; "in" is Drake's. Its contact surface is the
  intersection of the compliant tets with the partner's mesh, triangulated about polygon centroids, quadrature per
  triangle (continuous: pieces enter at zero area); resolution from the meshes, not the partner's physics nodes.
- Now: a pad holds the outer half of each pair's thickness (`PAD_DEPTH`), its core's barrier the inner half; only
  Newton's `kh` (`DM_PADS=kh,pattern[,d]`). Points: per rod edge `ceil(l / r)` fixed points (<= SIDES = 16), per
  triangle `n^2` sub-triangle centroids (`n = ceil(size / r)` <= 4); nodes keep the core's barrier only. Pair keys
  `key0 + SIDES shape + side`. Fixed points: energy continuous (a point enters at zero pressure).
- Hunt-Crossley lagged as Drake's: damping `d f0 / dt` on the depth change, a quadratic potential (`DM_PADS` 3rd
  field or `DM_PAD_HC`).

| gate | result |
|---|---|
| 1-2 press (rod, sheet, 1-300 g) | ratio 1.0000 (sheet 1.0025 at 1 g) |
| 3 tilt, n = 11 / 21 / 41 | delta0 5.0e-3 / 3.1e-3 / 2.2e-3, theta 1.0e-2 all (rod bending 8-9 um vs the rigid exact: the floor) |
| 4 drop, rebound vs HC's d v_o + ln(1 - d v_o) = -(d v_i - ln(1 + d v_i)), v_i 0.5, dt 5e-5 | d 0: 0.965 (BE); d 0.5 / 1 / 2 / 4: e 0.829 / 0.726 / 0.577 / 0.400 vs 0.857 / 0.748 / 0.594 / 0.411 (sim / exact 0.967-0.974: BE's own) |

(Detached-HEAD incident: something checked out 92886a1 in this worktree mid-session; content was intact, moved back
onto the branch as c9f4848. The first "new quadrature" press/tilt numbers were from the old code on sci6000.)

### Gate 7 with the new pads (DM_PADS on the 12 Franka finger shapes; pick-place servo 5 N, fold servo 2 N)

| variant | kh | grip lift / carry / settle | pick (rtol 2%) | fold (rtol 10%) | fold time |
|---|---|---|---|---|---|
| Robin x3 | 1e9 | 4.871 +- .000 / 4.870 +- .000 / 4.874 +- .003 | 4.870-4.880 | 1.86-2.50 | 1066 s |
| Robin x1 | 1e9 | 4.871 +- .000 / 4.871 +- .000 / 4.873 +- .003 | 4.869-4.873 | 1.84-2.52 | 479 s |
| lagged x1 | 1e9 | 4.871 +- .000 / 4.870 +- .000 / 4.874 +- .003 | 4.868-4.872 | 1.85-2.48 | 86 s |
| lagged x1 | 1e8 | 4.374 +- .000 / 4.374 +- .001 / 4.374 +- .002 | 4.374-4.375 | **pass** | 80 s |
| (no pads, Robin x3, A1) | - | 5.05 +- .23 / .28 / .27 | | | |

The grip is steady in every variant, even lagged x1 (the "flips" fraction is meaningless at +- 0.000). Its offset from
the servo is the pad's sink: a position servo (1000 N/m) on a compliant fingertip loses k_servo delta, delta from
Winkler (cable R 5 mm, ~2 cm in the pad): kh 1e9 delta ~0.14 mm -> 0.13 N; 1e8 ~0.65 mm -> 0.63 N, as measured.
The tests assume a rigid stop. Fold at kh 1e9: the napkin's grip spreads 1.85-2.5 (sheet points); passes at 1e8.
MIDAS (no padded shapes) flaky on both branches: 5/6 pass here, 5/6 on b45a3ff. Suite = b45a3ff.
