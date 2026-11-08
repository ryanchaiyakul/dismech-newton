# Handoff: finish the proxy coupling (Robin), then hydroelastic pad contact (Drake-style) with tactile output

Written 2026-11-08 after `scratch/handoffs/proxy-coupling-principled.md` (rounds 4-6 done). Two parts: A cements the
MuJoCo <-> SolverDiSMech coupling (the user approved a src change for it), B adds Drake's hydroelastic contact for
compliant fingertip pads, because the MIDAS hand has tactile sensors the user wants to mimic. A's stiffness sweep
(A1) is the cheap preview of whether B's softer pad contact also settles the coupling. Bring each table to the user
before adopting; never commit to `benchmark` or merge without asking. Experiments on **sci6000** (see the previous
handoff's "Running" section: 4x RTX PRO 6000, `ssh sci6000`, one config per GPU, scripts from files, background ssh).

## Where things are

- **Branch `exp/proxy-robin`** (pushed): local worktree `.claude/worktrees/proxy-robin`, sci6000 worktree
  `~/dev/dismech-newton/.claude/worktrees/proxy-robin` (on branch `exp/proxy-robin`; `git pull origin exp/proxy-robin`).
  Base: `exp/proxy-impact` 136a2d5 (ipc-contact's uncommitted contact rebuild on fa9a553 + experiment switches).
  `NOTES.md` rounds 4-6 have every number; logs in `exp/principled/logs/`.
- **Don't use `.claude/worktrees/proxy-impact`**: another session took it over at 13:03 on 2026-11-08 (checked out
  410a9c6). A stray commit of mine with their working changes is on local ref `exp-robin-stray` (delete when they
  are done).
- Experiment code: `src/dismech_newton/schur_exp.py` (installed from `__init__`), switches below. Tools in
  `exp/principled/`: `chatter.py` (pick-place grip per phase: squeeze / antisymmetric / sign flips / cable accel),
  `dump_schur.py` (source response), `finger_robin.py` (numpy squeeze-mode model with the log barrier),
  `remote_gates.sh SCHUR PASSES TAG [ENV=...]` (all gates over 4 GPUs), `remote_robin_iter.sh`, `restart_check.py`.
- Switches (all experiment, strip in A3): `DM_SCHUR=full|body|diag|probe` (+`_EPS`, `_DELTA`, `_EVERY`, `_REWIND`),
  `DM_ROBIN=1` (Robin by MuJoCo probes) / `2` (in our harvest, no probes), `DM_ROBIN_K=chord|end`,
  `DM_ROBIN_DEBT=1`, `DM_PASSES` (every coupler), plus the old `DM_NODC`, `DM_SYNC` in coupling.py, `PEND` and
  `DM_ITER` in tests/test_coupling.py, the full-k arm loops in contact.py (`_arm_inertia`, `_arm_assemble_kernel`).

## What is known (A)

Lagged proxy coupling = partitioned coupling with a wrong interface model. Measured source response (Franka gripper):
finger y 0.33 kg alone, left-right coupling -2.09 (the fingers' mimic equality: one motor), squeeze mode
1/(3.01 + 2.09) = **0.196 kg = m + h kd**; ProxyArm's view mass 0.0855 kg per finger, no cross terms. Giving the
proxy the measured k x k inertia alone does not fix anything (the jitter is in the converged force, not the passes).

**Robin exchange** (Badia-Nobile-Vergara 2008; Sicklinger 2014 interface Jacobians; Akbay et al. SIGGRAPH 2018
reduced interface system): MuJoCo takes the force the contact will have at its end pose,
F = F_ref + K (x1(F) - x_ref), K = our contact stiffness condensed on the proxy DOFs (cable re-equilibrated),
x_ref our proxy end pose. In arm coordinates, with X = (A^-1)_zz (our step's inverse on the arm DOFs, k back-solves
with the existing factorization; the last factorized matrix is as good as an end reassembly):

    next step:   F = (M / h^2) [h X M (2 v1 - v0) - (pred - z) - h (2 v1 - v0)]     (no inverse)
    repeat pass: F solves (I - h^2 K Minv) F = F_ref + K (x1 - h^2 Minv F_app - x_ref)

(v1 the synced velocity, v0 the last step's; source acceleration carried over as 2 v1 - v0; view Minv.)

| gate | today x1 | Robin x1 | Robin iterated x2 | x3 |
|---|---|---|---|---|
| box on clamped rod, rebound ratio | 1.91 | 0.21 | 0.15 (rod's spring-back) | |
| pick-place grip (5 N servo) | 4.997 +- .003 | 5.006 +- .003 | 5.05 +- .19 | +- .27 (x4 +- .31) |
| momentum / pin (1e-4) | pass / pass | 1.03 / 0.79 | 0.0029 / pass | pass / pass |
| napkin fold / MIDAS | pass / ~0.93 | fail (grip +-10%) / broken | pass / pass | |
| trampoline | 3/4 | 3/4 | | |

Robin x1 is momentum-inexact by design (it hands MuJoCo a predicted force; a momentum debt bounds the error to one
step's prediction, 20% of p0 at the rod impact, and costs rebound 0.39). Iterated Robin converges to F_applied =
F_contact: momentum exact at x3, everything passes **except the pick-place grip, whose chatter grows with passes**:
the converged interface oscillates in the gripper's squeeze mode at the step rate (300 Hz; physical squeeze mode
~11 Hz), the fold with the same gripper is steady. The one-pass lag filters it. Not the MuJoCo restart (checked).

## Part A: finish the coupling

1. **Stiffness sweep (first, cheap).** Iterated Robin x3 (`DM_ROBIN=2 DM_PASSES=3`): barrier `STIFFNESS` 1e3 / 1e2 /
   1e1 (contact.py), and the barrier scaled by the source's squeeze response instead of the view mass (the
   `arm_flexibility` term uses `Minv`). Gates: `chatter.py` (grip +- within 2%, no sign-flip pattern), fold, bounce,
   momentum. If a softer contact stops the converged chatter: the interface stiffness is the cause and B's pads (a
   physical, softer stiffness) are the principled cure. If not: stiffen MuJoCo's finger mimic equality (solref) to
   test whether it is the gripper model.
2. **Decide with the user**: iterated Robin (passes, cost x passes) vs Robin x1 (cheap, momentum to one step's
   debt; MIDAS/fold fail today). The user leans single-pass Robin; the numbers above say x1 fails MIDAS/fold.
3. **Src (approved)**: Robin in our harvest hook (`coupling_harvest_proxy_wrenches` override on SolverDiSMech, no
   coupler change, no MuJoCo internals): ProxyArm keeps v1 / v0 per DOF; after the step, k back-solves of unit RHS
   (all worlds at once) -> X [W, k, k]; one kernel per world for F. Graph-capturable. Then **strip every switch**
   (list above, `schur_exp.py`, full-k loops unless Robin needs them), keep src small (user judges by lines). Tests:
   test_proxy_impact (rebound), a steady-grip test, momentum with the scheme's accounting, examples' "lagged, one
   pass" docstrings, ProxyArm's docstring stating the interface model.

## Part B: hydroelastic pad contact (Drake's model) with tactile output

### Model (Drake; verify against the papers)

- Elandt, Drumwright, Sherman, Ruina, IROS 2019, "A pressure field model for fast, robust approximation of net
  contact force and moment between nominally rigid objects": each compliant body carries a pressure field
  p = E e(x), e the extent field (0 on the surface, 1 at depth h, the "hydroelastic margin/thickness"), E the
  hydroelastic modulus [Pa]. Compliant vs rigid: the contact surface is the rigid surface inside the compliant
  volume; force = integral of p n dA, moment likewise. With a linear field and rigid partner this is an elastic
  foundation (Winkler) of depth h: p = (E / h) * depth.
- Hunt-Crossley dissipation: p_d = p (1 + d v_n) (d [s/m], v_n approach speed).
- Masterjohn et al., RA-L 2022, "Velocity level approximation of pressure field contact patches"; Castro et al.,
  "An unconstrained convex formulation of compliant contact" (SAP, T-RO 2022) and "A theory of irrotational contact
  fields" (2023): the conservative (potential) form, friction per quadrature point mu p dA, regularized.
- Drake docs: "Hydroelastic contact" user guide (compliant/rigid roles, `hydroelastic_modulus`,
  `hunt_crossley_dissipation`, `resolution_hint`).

### Our form (a potential in q, implicit, IPC-style pair list)

- Roles: the **pad** (a proxy / rigid body's shape flagged compliant: modulus E, thickness h) is compliant; rods
  (capsules) and shells (triangles with thickness) are its rigid partners. Soft-soft and everything else stays on
  the barrier. Pad pairs come from the global sorted pair list (extend `collide.Offset` per CLAUDE.md; no Newton
  pipeline sources).
- Energy per partner quadrature point q (rod: points around/along the segment surface; shell: triangle quadrature on
  the facing side): depth s_q = max(0, -sd_pad(x_q)) capped at h; E_q = A_q (E / h) s_q^2 / 2 (+ beyond h a stiff
  barrier so nothing passes the pad). Gradient / Hessian through `arm_move` / `arm_assemble` for the pad body, nodes
  as today. Dissipation: Hunt-Crossley as a velocity term in the step (check it keeps A SPD).
- Friction: the existing smoothed Coulomb with normal force p_q A_q per point (feeds the friction thread,
  contact-step3.md).
- **Tactile output: the MIDAS hand's PaXini PX-6AX GEN3 kit** (user, 2026-11-08): 3 fingertip sensors
  PX6AX-GEN3-DP-S2015-Elite (~20 x 15 mm footprint, "15 mm" fingertip class), 1 thumb sensor PX6AX-GEN3-DP-M2826-Omega
  (26 mm class, ~28 x 26 x 14 mm), a high-speed comm board. GEN3 is **Hall-effect, magnet-in-elastomer**: each taxel
  reads a 3-axis deformation (normal + shear), the sensor reports a 6-axis net wrench plus the taxel array. So the
  simulated readout must be per-taxel **3-axis traction** (pressure p n plus the friction traction, integrated over
  the taxel's area) and the pad's net wrench, not pressure alone. Specs found online disagree (vendor-adjacent
  pages, not PaXini's datasheet): S2015 "52 force signals" vs "up to 121 taxels" (family max), pitch ~1-1.5 mm,
  0.01 N min force, 0-50 N fingertip range, 200 Hz vs 1000 Hz. **Authoritative taxel coordinates are PaXini's
  per-variant XLSX maps: PXSR-STDDP03G.xlsx (S2015), PXSR-STDDP03B.xlsx (M2826)** (PXSR software / comm board):
  get them from the user or the kit, and the elastomer's thickness and modulus (else calibrate E / h from a press
  test: force vs indentation, and shear vs tangential displacement). Sources: roboticscenter.ai/hardware/paxini-gen3
  (+ /specs), roboticscenter.ai/store/product/paxini-px6ax-gen3-dp-s2015-elite.
- Shear: Drake's pressure field has no tangential compliance; the sensor does (elastomer shear). Model the pad's
  tangential stiffness as the friction regularization of a stuck point (a tangential spring k_t = G / h per unit
  area until mu p), so a stuck grip reads shear like the sensor; check against a tangential press test.
- Pads only on **flagged shapes** (user's decision): every other rigid-vs-soft pair stays on the barrier.
- Coupling: the pad's stiffness is the interface K Robin needs (physical, much softer than the barrier's 1e3 x
  inertia / h^2): rerun A's gates with pads on the Franka fingers / MIDAS links.

### Build order and gates (each with a theory figure: dense sim-vs-exact curves, one-word titles)

1. Static cylinder (rod, radius R, length L, held rigid) pressed depth delta into a flat pad: Winkler exact
   F = (E L / h) (4/3) a delta, a = sqrt(2 R delta); pressure profile p(x) = (E/h)(delta - x^2 / 2R). Sweep delta;
   also mesh/quadrature convergence (resolution like Drake's `resolution_hint`).
2. Weight balance: a rod resting on a pad under gravity sinks to the depth the formula gives; shell sheet on a pad:
   uniform p = rho g t, depth p h / E.
3. Moment: off-centre load tilts as the foundation predicts (net moment = integral x p dA).
4. Dissipation: drop on a pad, restitution vs d (Hunt-Crossley's closed form for a linear foundation).
5. Friction: incline stick/slip threshold mu with the pressure-weighted normal force.
6. Tactile: taxel pressure map of a known indentation (cylinder: the parabola above sampled at the taxels).
7. Coupled: Franka pick-place with pad fingers (grip steady, the A gates), MIDAS with pads (axial slip, tactile
   map during the slip demo).

Default suite must stay green (76 passed / 3 xfailed at the A baseline); `uv run ruff check`; FD tolerances from
`--seeds 16` spreads for any new gradient test.

## Open questions for the user

- PaXini taxel maps (PXSR-STDDP03G.xlsx, PXSR-STDDP03B.xlsx), elastomer thickness / modulus, which MIDAS links
  carry which sensor, and the readout units the user wants matched (raw Hall counts vs calibrated N).
- Single-pass vs iterated Robin after A1.
- (Decided: pads only on flagged shapes.)
