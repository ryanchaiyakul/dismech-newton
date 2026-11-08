# Handoff: principled proxy coupling (MuJoCo <-> SolverDiSMech through SolverCoupledProxy)

Written 2026-11-08 after `scratch/handoffs/proxy-impact.md` (all its options measured). Goal: one cemented,
defensible coupling algorithm that gets **impacts** (no energy gain), **steady servo grips** (the servo's force, no
jitter), **momentum** and **articulated hands** (MIDAS) right at once, with clean solver separation (each solver its
own physics; only states, forces and an interface model cross). Explore every idea below on sci6000, bring the table
to the user before adopting; never commit to `benchmark` or merge without asking.

## Where things are

- **Branch `exp/proxy-impact`** (pushed; commit/push there freely). Local worktree `.claude/worktrees/proxy-impact`;
  sci6000 worktree `~/dev/dismech-newton/.claude/worktrees/proxy-impact` (`git pull origin exp/proxy-impact`).
  Base: ipc-contact's uncommitted contact rebuild applied on fa9a553 (src = ipc-contact's diff + experiment switches).
- `NOTES.md` there has every number (rounds 1-3). Logs in `exp/step4/logs/`.
- **Experiment switches now in the worktree (strip before any merge):** `coupling.py`: `DM_NODC` (offset out of pred),
  `DM_SYNC` (proxy starts at MuJoCo's synced pose each step); `tests/test_coupling.py`: `PEND` (count the pending
  feedback in the momentum tests), `_proxy_scene` iterations from `DM_ITER` (default 2), `fixed=` option, and the new
  `test_proxy_impact` (box dropped 1 cm on a fully clamped rod: no rebound, rests r up, feedback = weight).
  The earlier `DM_PROXY` / `DM_EXTRAP` / `DM_MODE` overrides were reverted (in git history of the branch).
- Tools (`exp/step4/`): `bounce.py` / `bounce_trace.py` (box on an end-clamped rod; per-step z, vz, feedback, proxy -
  MuJoCo), `test_pinned.py` (draft of test_proxy_impact), `grip_trace.py` (pick-place grip per soft solve = per pass,
  with each pass's input force and offset; env PASSES, RELAX, RELAX_MODE, MASS_SCALE), `finger_mass.py` (finger model
  vs proxy mass, joint gains), `remote_gates.sh` / `remote_iters.sh` / `remote_stag.sh` (multi-GPU gate runs).

## Running: sci6000, not the local GPU

The user wants experiments on **sci6000** (4x RTX PRO 6000, 96 GB each; `ssh sci6000`, uv at `~/.local/bin`), local
GPU preferably unused. Pattern: commit + push to `exp/proxy-impact`, `git pull` in the sci6000 worktree, launch one
config per GPU with `CUDA_VISIBLE_DEVICES=i` from a script file (inline multi-line ssh commands lost variables
once), run the ssh in the background and wait for the notification. Check `nvidia-smi` first (GPUs 0-1 often carry
8-10 GB of someone's idle memory: fine to share). Kill your own leftovers by PID (`pgrep -af "pytest|cloth_rigid"`):
`pkill -f proxy-impact` kills the ssh session itself. Run many variants in parallel: 4 GPUs, and each GPU fits several
small scenes at once (the coupling scenes use < 1 GB).

## What is known (numbers in NOTES.md)

Coupled step (lagged): MuJoCo steps with the feedback F; its start pose + end velocity sync into our proxies
(`ProxyArm`: 6 DOFs per MuJoCo body in our solve, inertia = Newton's per-body effective mass from the view, target =
synced pose + h v_free); our step; the coupler harvests the proxies' momentum change as next F. Staggered syncs the
end pose and applies F in the interval it was computed for.

| scheme | impact (rebound/impact) | rest on clamped rod | momentum | pick-place grip (5 N) | MIDAS | trampoline |
|---|---|---|---|---|---|---|
| lagged x1 (today) | 1.91 (box flies off) | fails | ok (counting pending F) | 5.00 +- 0.01 | ~0.93 | 3/4 |
| lagged x2 | 0.15 (rod's own spring-back) | exact | 0.0029 x2, exact x3+ | 5.04, +-0.2 2-cycle | ok | 4/4 |
| lagged x3 / x4 / x4 Aitken | same | exact | exact | worse: +-0.35 / +-0.4 | ok (Aitken: axial 1.04) | - |
| staggered | 0.15 | exact | exact | 4.83 (3.4% low) | 0.773 | 0/4 (NaN) |
| staggered + extrapolated source accel | 0.15 | exact, MuJoCo == proxy | exact | 4.82-5.03 | NaN (x2) / 0.25 | - |
| lagged x1, offset out (opt 1) / harvest - offset (opt 2) | 1.05 / 2.05 | weight 0.28/0.31 / fails | | 4.83 | | 2/4 |

Grip 2-cycle at 2 passes (`grip_trace.py`): passes converge within a step (slope ~0.46 per pass: output force vs input
force), but the converged force alternates step to step (-4.82 / -5.34 N). Not: unconverged solves (counts 0),
relaxation 0.5/0.7, starting the proxy at MuJoCo's pose (`DM_SYNC`: same), dropping the offset (`DM_NODC`: worse),
`mass_scale` 2.9 (worse, +-10%, also scales the hand). The finger: model mass 0.029 kg, proxy (view) mass 0.0855 kg,
PD servo ke 1000, kd 100 under MuJoCo `implicitfast`, h = 1/600 s: its step response is ~m + h kd = 0.2 kg along the
joint (implicit damping), and articulated (cross terms with the hand / other finger).

## Diagnosis: a partitioned scheme with the wrong interface impedance

Literature (from memory, verify): FSI partitioned coupling and multibody co-simulation.
- Causin, Gerbeau, Nobile, CMAME 2005, added-mass effect: explicit (one-pass) coupling gains energy when the interface
  inertia is misrepresented. = our lagged x1 impact bounce.
- Kuttler & Wall, Comput. Mech. 2008: Aitken-relaxed fixed-point subiterations. Degroote, Bathe, Vierendeels 2009:
  IQN-ILS interface quasi-Newton (builds the interface Jacobian from the passes).
- Badia, Nobile, Vergara, JCP 2008: Robin transmission conditions; with the true impedance one exchange is consistent.
- **Peiret, Andrews, Kovecses, Kry, Teichmann, ACM TOG 2019, Schur complement-based substructuring of stiff multibody
  systems with contact**: couple subsystems through their Schur complements (each side's step response condensed on
  the interface). Closest to us.
- Kubler & Schiehlen 2000 (simulator coupling, interface Jacobians); Gomes et al., ACM CSUR 2018 (co-simulation
  survey); Sadjina et al. 2017 (energy-conserving co-simulation, NEPCE power-bond correction).
- Monolithic alternatives (break separation; for reference): Shinar, Schroeder, Fedkiw SCA 2008; Affine Body Dynamics
  (Lan et al. SIGGRAPH 2022); unified Newton barrier multibody (Chen et al. 2022).

Hypothesis: `ProxyArm`'s inertia (block-diagonal per-body effective mass) is the interface impedance and it is wrong:
it lacks the source's implicit damping/stiffness (h D, h^2 K: servo kd, ke) and the cross terms between bodies. With
the exact impedance (the source's step operator condensed on the proxy DOFs) the per-pass slope -> 0 (one pass is
already the converged force), and the grip cycle, the MIDAS extrapolation failure and the impact lag share one fix.

## Ideas to explore (each on the same gates; parallel on sci6000)

1. **Schur-complement proxy inertia (main idea).** Proxy inertia = (J A^-1 J^T)^-1 per world, A = MuJoCo's step
   matrix (`implicitfast`: M - h dF/dv, plus h^2 K of position servos if treated implicitly; check what mujoco_warp
   integrates), J = the proxy bodies' 6-DOF Jacobians. Full k x k block (ProxyArm's M is already `[world, k, k]`).
   Source: mujoco_warp's `qM`, `qDeriv` / `qLD` (inspect `SolverMuJoCo`'s data), first as an experiment computed in a
   script from the coupler (`solver._entries["mjc"]`), later as an interface call (Newton's
   `coupling_eval_effective_mass_block` is the per-body version: read what it computes). Measure: per-pass slope
   (grip_trace) -> ~0, grip steady at x1 and x2, impact at x1 (does the right impedance remove the one-step bounce
   alone, or does the lag still need a pass?), MIDAS, momentum.
   Sub-variants: diagonal of it only; mass + h kd only (joint-space, cheap) to isolate damping vs cross terms.
2. **Interface quasi-Newton / Aitken passes (IQN-ILS).** If 1 is not available exactly, learn the interface Jacobian
   from passes (Degroote 2009); compare to Newton's Aitken. Check whether passes then converge to a non-alternating
   force (the 2-cycle is the converged answer now, so this alone likely isn't enough: test that claim).
3. **Robin-type exchange.** Feed MuJoCo F + Z (v_mj - v_proxy) (impedance Z) instead of F alone (Badia 2008); Z from
   idea 1 or tuned. Lets lagged x1 correct the velocity mismatch without a second pass.
4. **Why the converged x2 answer alternates.** Before trusting any fix, explain the -4.82/-5.34 cycle analytically:
   model one finger as mass m, PD servo (ke, kd implicit in MuJoCo), barrier of stiffness k_b in our solve, proxy mass
   m_p; iterate lagged with 1 and 2 passes on paper / in numpy (no GPU) and reproduce the 2-cycle and its dependence
   on m_p. That model also says which impedance is exact.
5. **Staggered with the exact impedance.** Staggered failed on servo forces (unknown next-step source forces) and
   articulated hands (extrapolation). With idea 1's step operator, the source's next-step response to F is known:
   revisit staggered + extrapolation; it is the one-pass, momentum-exact scheme.
6. **Energy-based correction** (Sadjina NEPCE): correct the interface energy residual each step. Lower priority.
7. **Monolithic reference** (not for adoption): a rigid box as 6 DOFs fully in our solve (no MuJoCo) on the same
   impact / rest scenes, as the ground truth the coupled runs should match (staggered x1 and lagged x2 matched it to
   h^2 g on the box).

## Gates (same for every variant; record a table row each)

- `tests/test_coupling.py::test_proxy_impact` (in the worktree) and `exp/step4/bounce.py`: rebound <= 5% of impact on
  the clamped rod; 0.15 on the end-clamped rod is the rod's elastic spring-back (no force after the stop).
- `exp/step4/grip_trace.py`: pick-place grip per substep within 2% of the 5 N servo, no 2-cycle; per-pass slope.
- `tests/test_coupling.py -m "slow or not slow"` (9 + impact): momentum (with the right pending-force accounting:
  with passes the leftover is dt (F_last - F_applied), not dt F), pin, static load, pick-place, MIDAS (axial slip
  ~0.93 is the friction thread's: report it), napkin fold (flap check flaky ~1/5 at baseline), cross-slide, Franka
  swap.
- `examples/newton/cloth_rigid_trampoline.py --viewer null --test` x4 (body 2 roll-off is the known failure).
- Default suite (76 passed / 3 xfailed at baseline) and the cost per step (passes x MuJoCo + our solve).

## Then

Cement: one scheme, one setting, tests stating the physics (test_proxy_impact, a steady-grip test, momentum with the
right accounting), the examples' coupler configs and docstrings updated ("lagged, one pass" claims), the switches
stripped, ProxyArm's docstring stating the interface model. Src small (user judges by lines). Then friction
(contact-step3.md "force-accurate friction"), which needs this exact normal force.
