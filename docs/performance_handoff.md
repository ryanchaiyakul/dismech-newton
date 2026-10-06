# Performance Handoff: Forward and Reverse

As of 2026-10-06, at commit `4fa3168`. Measured on an RTX 2070 Super (sm_75, 8 GB; FP64 runs at 1/32 of FP32), with
Warp 1.17, nvmath 1.0, cuDSS 0.8 and Newton's collision pipeline.

The ADMM local kernel is no longer the bottleneck. The previous handoff measured it at 1.7 ms per launch at 59k
triplets, spilling 2 KB per thread. It now takes about 0.11 ms, and the Newton `assemble` kernel dropped from 1.9 ms to
0.6 ms. This document reprofiles every path and ranks what to do next.

## Summary

| Workload | Path | Where the time goes | Next lever |
| --- | --- | --- | --- |
| `fit_buckling.py` gradient (59k triplets, 2,560 rods) | Newton forward, implicit adjoint | **cuDSS: 88% of GPU time** (factorization alone 70%) | Batched per-rod banded solver |
| `cable_pile.py` (3,900 triplets, contacts) | ADMM, graph-captured | collide 29%, linear solve 27%, elastic local 21%, contact local 15% | ADMM iteration count, then fewer launches |
| `plectoneme.py` (499 triplets, twisted) | ADMM, graph-captured | Six 2–9 µs kernels per iteration × 3,200 iterations per frame | ADMM iteration count, then one fused kernel |

- **Reverse (system identification)**: one change dominates. Replacing cuDSS with a batched banded factorization for
  rod chains should cut a gradient evaluation about 5–6× (an Amdahl estimate, not a measurement).
- **Forward (contact simulation)**: every scene runs ADMM to its iteration cap on every substep (50 in `cable_pile`,
  200 in `plectoneme`). ADMM converges slowly; it is not stalling at an fp32 floor. Cutting the iteration count gives
  more than any per-kernel work, and it also changes results, so it needs the end-to-end gates.

## Reverse path: `fit_buckling.py`

The fit runs batches of 2,560 rods (256 trials × 10 windows), with 99 DOFs per rod. That is 58,880 triplets and
253,440 DOFs. One loss-and-gradient evaluation is 8 batches × 24 steps = 192 steps. Each step is a Newton solve on a
`wp.Tape` (tolerance 1e-6), then `StepAdjoint.vjp`.

| Measure | Value |
| --- | --- |
| Full evaluation (192 steps, at the L-BFGS initial guess) | 5.34 s |
| Per step: forward / backward | 18.7 ms / 9.2 ms |
| GPU busy during the traced steps | 98% of wall time (host overhead is not the problem) |
| cuDSS factorizations per step | 3.2 (about 2.2 Newton iterations, plus 1 adjoint) |

GPU time per step (48 steps traced):

| Kernel or phase | ms per step | Share | µs per call |
| --- | --- | --- | --- |
| cuDSS `factorize_ker` (fp64) | 19.38 | 70.5% | 6,039 |
| cuDSS `bwd_ker` + `fwd_ker` (triangular solves) | 3.36 | 12.2% | 590 + 457 |
| cuDSS `copy_matrix_ker`, `diag_ker`, `perm_ker` | 0.99 | 3.6% | |
| `energy_kernels.assemble` | 1.88 | 6.8% | 586 |
| Everything else (inertia, residual tape, theta, frames, memsets) | 1.9 | 7% | |
| **By phase:** forward cuDSS / adjoint cuDSS | 16.5 / 7.5 | 60% / 27% | |

### R1. A batched banded solver for rod chains (replaces cuDSS)

Each rod is one connected component of the Hessian. Order a rod's DOFs as `[x0, θ0, x1, θ1, …]`, four per node. A
triplet then touches 11 consecutive indices, so the half-bandwidth is 10. A banded Cholesky for one 99-DOF rod is
about `n w² ≈ 10⁴` flops, about 26 MFLOP for all 2,560 rods. cuDSS spends 6 ms per factorization on that, in fp64,
on a card with 1/32-rate fp64.

- **What to build.** A `BandedSolver` in `linear.py`, alongside `TridiagonalSolver` and `BlockInverseSolver`. It
  needs a `fits()` test that every component is a chain of bandwidth ≤ w after the per-rod reordering, and a
  `solve(b, x)` that refactors when the values change (the `refactorize` contract that `CudssSolver` follows). Use
  one thread, or one warp, per rod. Pick it in `sparse_solver()` for both `DiSMechSolver` and `StepAdjoint`.
- **Precision.** Factor in fp32 and refine once against the fp64 residual (`_Residual` already does this for the
  increment form). Or factor in fp64: at about 26 MFLOP, the 1/32-rate fp64 still costs well under 1 ms. Check
  gradient accuracy with `test_adjoint.py` either way.
- **Expected effect.** cuDSS's 24 ms per step drops to about 1 ms, so a step goes from 28 ms to roughly 5 ms, about
  5–6× on the whole evaluation. That is an estimate. `assemble` then becomes the largest single kernel, at about 40%
  of what remains.
- **Side benefit.** A Warp-launched solver can run inside `wp.capture_while`. cuDSS cannot, which is why Newton with
  a tolerance is not graph-captured today (`graph_capturable` / `loop_capturable` in `solver.py`).
- **What it does not cover.** The contact adjoint's augmented system `M = [[J, ρCᵀ(I − D)], [C, −D]]` couples rods
  through contact pairs. Keep cuDSS there, and use the banded solver only when `contacts is None`, or when the system
  splits into rod chains.

A cheap experiment to run first: switch `CudssSolver` to `_CUDA_R_32F` for the factorization, refine against the fp64
residual, and time it. That shows how much of the 6 ms is the fp64 penalty and how much is cuDSS's overhead for many
tiny components.

### R2. Smaller reverse-path items (after R1)

- **Reuse the forward factorization in the adjoint.** The adjoint factors `J` at `q_theta`. The last Newton
  iteration factored it at the previous iterate, which is close, and `StepAdjoint.refine` exists to correct exactly
  this kind of approximation. Reusing it saves 1 of the 3.2 factorizations per step. It is mostly moot once R1 makes
  factorizations cheap.
- **`assemble` at 586 µs** (242 registers, 32 B of spill) is the next kernel in line. The per-pair CSR slots already
  removed the binary search. With R1, the band storage could replace the CSR scatter entirely: the assembler writes
  the band directly.

## Forward path: ADMM contact simulation

Both examples ran with Newton's collision pipeline and the default `ADMMDiSMechSolver` (`tol=1e-4`,
`check_every=10`, tridiagonal global solve). Wall times come from graph replays, with a synchronize around `step()`.
Kernel breakdowns come from an eager run of the same frames, because CUPTI does not report repeated iterations
inside a graph's conditional while-loop body (see Tooling). In both scenes, the eager GPU kernel total is 89–94% of
the graph wall time, so the graph runs are GPU-bound and the eager breakdown is representative.

### `cable_pile.py`, frames 5–25 (8 substeps per frame)

| Measure | Value |
| --- | --- |
| Wall per frame (graph) | 16.5 ms |
| GPU kernel time per frame (eager) | 14.7 ms |
| ADMM iterations per substep | 10 before the cables touch, then **50 (the cap)** every substep |

| Phase | ms per frame | Share | Per call |
| --- | --- | --- | --- |
| `collide` (Newton pipeline) | 4.26 | 29% | `_nxn_broadphase_precomputed_pairs` alone is 2.95 ms (20%), 369 µs per substep |
| `linear.solve` (tridiagonal) | 3.94 | 27% | 4 launches per iteration: gather 4.4 + gtsv 3.1 + 5.6 + scatter 2.3 = 15.4 µs |
| `elastic.local` | 3.15 | 21% | 12.3 µs |
| `contact.local` | 2.25 | 15% | 8.8 µs |
| Rest of the step | 1.1 | 8% | |

### `plectoneme.py` at 11.5 turns, frame 3000 (16 substeps per frame, cap 200)

| Measure | Value |
| --- | --- |
| Wall per frame (graph) | 100.7 ms |
| GPU kernel time per frame (eager) | 94.9 ms |
| ADMM iterations per substep | **200 (the cap)** every substep: 3,216 iterations per frame |
| Per iteration | 6 launches, 27.9 µs of kernel time, 31.3 µs of graph wall |

| Phase | Share | µs per iteration |
| --- | --- | --- |
| `linear.solve` | 45% | 13.1 (gtsv loop 5.6, scatter 2.8, gather 2.6, gtsv first pass 2.1) |
| `elastic.local` | 31% | 9.1 |
| `contact.local` | 20% | 5.7 |
| collide and the rest of the step | 5% | |

At 499 triplets every kernel is a single partial wave, so its time is launch latency plus one thread's serial chain.
Making the local kernel faster again would save at most 9 of the 31 µs per iteration.

### ADMM convergence (why the caps are always hit)

On `cable_pile` frame 20, one substep was run with the cap lifted, logging the residual every 10 iterations:

| Iterations | 10 | 100 | 300 | 610 | 1,010 | 1,810 |
| --- | --- | --- | --- | --- | --- | --- |
| Local residual (`stats[0]`) | 3.8e-3 | 2.8e-3 | 1.5e-3 | 7.4e-4 | 2.8e-4 | 2.2e-4 |
| q change per iteration (`stats[1]`) | 4.0e-4 | 4.0e-5 | 1.0e-5 | 9.3e-6 | 4.7e-6 | 2.3e-6 |

The tolerance is 1e-4. The residual falls smoothly, so ADMM is slow, not stuck at fp32 round-off, and 2,000
iterations still do not reach it. So the iteration caps, not the tolerance, decide both the cost and the accuracy of
every contact step.

### R3. Fewer ADMM iterations (the largest forward lever)

Every option below changes results and iteration counts, so gate each one on the end-to-end tests: `pytest tests/`,
the examples' `--test` runs, and the plectoneme theory checks.

1. **Split the residual first.** `stats[0]` is the `atomic_max` of the elastic (`triplet.py:437`) and contact
   (`contact.py:541`) residuals. Record them separately to learn which term holds convergence back.
2. **Over-relaxation** (`z` built from `α S q + (1 − α) z_prev`, with α ≈ 1.5–1.8). Nearly free: one line in each
   local kernel.
3. **Penalty balancing.** `rho` is fixed at construction (`rho_scale`, `contact_rho_scale`). Balancing the primal
   and dual residuals needs a refactorization of `H`. That is cheap for the tridiagonal solver, but not per
   iteration inside a graph. Do it per step, or every N steps, on the host.
4. **Anderson acceleration** on `(z, u)` or on `q`. It is the standard fix for linearly converging ADMM. It costs a
   few dot products per iteration and a small least-squares solve, which a warp-level kernel can do.

Measure iterations to reach `tol` on fixed snapshots of `cable_pile` (frame 20) and `plectoneme` (ply formed) before
and after each change. Even 2× fewer iterations at the same accuracy beats every kernel-level item below.

### R4. Fewer launches per ADMM iteration (the small-scene lever)

Each iteration launches six kernels: gather, gtsv first pass, gtsv loop, scatter, elastic local, contact local. `H`
is constant within a step, so its chain factors can be precomputed once.

- **A custom tridiagonal solve, 4 launches down to 1.** Precompute LDLᵀ (or cyclic-reduction) factors at
  `_factorize`. Write one kernel that gathers the right-hand side, solves, and scatters into `q`. Use one thread per
  chain for short chains (`cable_pile`: 400 chains of 41), and a block per chain with shared-memory cyclic reduction
  for long ones (`plectoneme`: chains of about 500). This saves about 10 of 31 µs per iteration in `plectoneme`, and
  about 9 of 37 µs in `cable_pile`.
- **Fuse the elastic and contact local steps** into one launch. Both read `q` and scatter into the same `rhs`.
- **Persistent kernel (the ceiling).** A 500-triplet scene fits in a few blocks, so a cooperative kernel could run all
  `check_every` iterations with grid syncs instead of launches. This is native-CUDA territory (the old handoff's
  option B: NVRTC on Warp's stream). Do it only after R3, because the iteration count multiplies everything.

### R5. Collision once per frame, not once per substep (`cable_pile`)

`collide` takes 29% of `cable_pile`, mostly the all-pairs broad phase over 4,000 capsules. SAP was measured at the
same cost (4.33 against 4.26 ms per frame), so changing the broad phase is not the fix. The pipeline already uses a
speculative gap (`dt=2 * sim_dt`, `speculative_contact_gap_max=2 * radius`). Colliding once per frame, or every k
substeps, with a gap that covers the frame's motion would cut this 4–8×. Check it against penetration in the
`--test` runs. This change belongs in the examples' `CableExample.simulate`, not in the solver.

## Order of work

| # | Item | Path | Expected effect | Effort |
| --- | --- | --- | --- | --- |
| 1 | cuDSS in fp32 for the factorization (experiment) | Reverse | Measures the fp64 penalty | Low |
| 2 | Banded per-rod solver (R1) | Reverse, and contact-free Newton forward | About 5–6× per fit evaluation (estimate) | Medium |
| 3 | Split the residual; over-relaxation (R3.1–2) | Forward | Unknown; potentially 2× fewer iterations | Low |
| 4 | Anderson acceleration or penalty balancing (R3.3–4) | Forward | Unknown; the largest potential gain | Medium |
| 5 | One-launch tridiagonal solve; fused local steps (R4) | Forward | About 30% per iteration in small scenes | Medium |
| 6 | Collision every frame or every k substeps (R5) | Forward (examples) | Up to 25% of `cable_pile` | Low |
| 7 | Reuse the forward factorization in the adjoint (R2) | Reverse | 1 of 3.2 factorizations per step | Low |
| 8 | Persistent cooperative ADMM kernel (R4) | Forward | Removes launch latency entirely | High |

## Tooling

No CUDA toolkit is installed (no `nsys` or `ncu`). Everything above came from CUPTI activity tracing:

- **Install.** `uv pip install --target <dir> cupti-python` (13.4). Append `<dir>` to `sys.path` *after* the venv,
  because it pulls in its own numpy. Preload `<dir>/nvidia/cu13/lib/libcupti.so.13` with
  `ctypes.CDLL(..., RTLD_GLOBAL)` before `from cupti import cupti`.
- **Usage.** Enable `CONCURRENT_KERNEL`, `MEMCPY`, `MEMSET`, `EXTERNAL_CORRELATION`, `DRIVER` and `RUNTIME`. Without
  the API kinds, no external correlation records are emitted. Tag phases by wrapping methods with
  `cupti.activity_push_external_correlation_id(CUSTOM0, id)` / `pop`, and map each kernel's `correlation_id` to a
  phase through those records. This captures cuDSS and cuSPARSE kernels, which `wp.ScopedTimer(cuda_filter=...)`
  misses.
- **Caveat.** Inside a graph with a conditional while-loop (`wp.capture_while`), CUPTI reports the loop body's
  kernels only once per graph launch. Use eager runs for kernel breakdowns, and graph runs only for wall time.
- **Late-frame snapshots.** `plectoneme` needs about 3,000 frames (about 3.5 minutes with graphs) to form its ply.
  Warm up with graph replay and switch to eager mode only for the traced frames (set `graph = None` and
  `capture = False` on the example for those frames).
- **Memory.** The machine has 15 GB of RAM. Run one compile-heavy job at a time, under
  `systemd-run --user --scope -p MemoryMax=5G` (the fit needs about 6 GB).

## Open questions

- How much of cuDSS's 6 ms factorization is the fp64 rate, and how much is per-component overhead (experiment 1)?
- Which term, elastic or contact, holds back ADMM convergence in each scene?
- Do the `plectoneme` theory checks (F, M and R within 3–5%) hold with fewer iterations, or do they rely on the
  200-iteration cap?
- Data-center GPUs (full-rate fp64) would shrink R1's gain from the fp64 rate. Re-measure on the target hardware.
- Carried over from the previous handoff, still open: `TripletTerm.begin_step` initializes `z` and `u` only once (a
  NaN poisons later runs), and `rho` does not follow parameter changes the way `refresh_mass` refreshes the masses.
