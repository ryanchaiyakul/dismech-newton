# Examples

Each example is one file, runnable on its own; `--viewer null --test` runs it headless and checks it.
Read them in this order: each adds one thing to the last.

| Example | What it adds |
| --- | --- |
| [`cable_twist.py`](cable_twist.py) | The basics: `add_rod` with stiffness and damping, `fix_segment`, the `DiSMechSolver` step, a clamped segment driven through its twist. |
| [`cable_pile.py`](cable_pile.py) | Many rods in one model, ground contact and friction. |
| [`cantilever.py`](cantilever.py) | ADMM on implicit Euler and `exact=True` Gauss-Newton on implicit midpoint (`theta=0.5`) beside Newton's VBD, against a beam formula: numerical damping against none. |
| [`overhand_knot.py`](overhand_knot.py) | Self-contact with friction, both ends clamped and pulled, a measured force against theory. |
| [`plectoneme.py`](plectoneme.py) | A rod from material constants (`newton.Rod`), frictionless self-contact under twist, against Clauvelin et al. (2009). |
| [`flagella.py`](flagella.py) | External forces on the nodes (`state.particle_f`: fluid drag) and contact between rods. |
| [`fit_buckling.py`](fit_buckling.py) | Gradients through the solver: `wp.Tape`, `tape.backward` in a batched model, `solver.restart` per window, L-BFGS on stiffness, density and damping from noisy 30 fps recordings and a gripper force sensor (positions alone only fix their ratios), windowed so every rollout stays on the recorded buckling branch. |
| [`fit_capture.py`](fit_capture.py) | The rod model itself from one RGB-D camera with simulated real-camera flaws: length, radius and held pose fitted as a chain of plain cylinders (analytic outline and depth), splats on it painted from the still; then gradients from pixels (gswarp in torch, the image loss handed back to `wp.Tape`): a coarse scan and L-BFGS on bend stiffness and damping from the same camera's video. Needs `--extra splat` and CUDA. |
| [`shell_cantilever.py`](shell_cantilever.py) | `cantilever.py` for midedge shells (`add_cloth_mesh`) beside VBD cloth from the same parameters, against plate theory; what Newton's `edge_ke` means as a plate stiffness. |
| [`shell_fold.py`](shell_fold.py) | A cloth strip flopped over onto itself: it falls, lands on its own lower half and settles into an open fold (an eyelet), held by self-contact with friction, its fall calmed by air drag on the nodes (`state.particle_f`), against the frictionless heavy elastica (height `0.907 (D / rho g)^(1/3)`). |

```bash
uv run examples/cable_twist.py
uv run examples/cable_twist.py --viewer null --test
```

## The pattern

Most examples subclass `CableExample` ([`utils/common.py`](utils/common.py)), Newton's example format:

```python
builder = newton.ModelBuilder()
bodies = DiSMechSolver.add_rod(builder, newton.Rod(points, radius=r), bend_stiffness=k)  # colliding
DiSMechSolver.fix_segment(builder, bodies[0])               # clamp the first segment
model = builder.finalize()
self.start(viewer, model, DiSMechSolver(model, friction=0.5))  # states; the solver finds its own contacts
self.spin = Drive(model, segment_dofs(model, bodies[0], twist_only=True))
self.drives = (self.spin,)                                   # prescribed every substep
```

`CableExample.step` then runs each frame's substeps (drive, solve) and replays them as one CUDA
graph after the first frame. Clamped DOFs move only through `state.dismech.q`, which `Drive` writes.

[`utils/`](utils) holds what the examples share and the package does not: the frame loop, drives, camera
framing, geometry checks, the theory plots, the fluid drag of `flagella.py` and `cached`, which stores
results that take minutes (`fit_buckling.py`'s recordings and fit) in `.cache/examples`, once per configuration.
[`utils/splat.py`](utils/splat.py) skins Gaussian splats to the rod's edges (differentiable, with USD export);
[`utils/gaussians.py`](utils/gaussians.py) renders them with gswarp for `fit_capture.py` and hands the image
loss's gradient back to Warp.
[`utils/reference.py`](utils/reference.py) records every example's trajectory (`record`) and compares a later run
against it (`compare`), to catch solver regressions.
