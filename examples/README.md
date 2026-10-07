# Examples

Each example is one file, runnable on its own; `--viewer null --test` runs it headless and checks it.
Read them in this order: each adds one thing to the last.

| Example | What it adds |
| --- | --- |
| [`cable_twist.py`](cable_twist.py) | The basics: `add_rod` with stiffness and damping, `fix_segment`, the `ADMMDiSMechSolver` step, a clamped segment driven through its twist. |
| [`cable_pile.py`](cable_pile.py) | Many rods in one model, ground contact and friction. |
| [`cantilever.py`](cantilever.py) | Both solvers (`ADMMDiSMechSolver`, `DiSMechSolver`) and their integrators (`theta`), against a beam formula. |
| [`overhand_knot.py`](overhand_knot.py) | Self-contact with friction, both ends clamped and pulled, a measured force against theory. |
| [`plectoneme.py`](plectoneme.py) | A rod from material constants (`newton.Rod`), frictionless self-contact under twist, against Clauvelin et al. (2009). |
| [`flagella.py`](flagella.py) | External forces on the nodes (`state.particle_f`: fluid drag) and contact between rods. |
| [`fit_buckling.py`](fit_buckling.py) | Gradients through the ADMM solver: `wp.Tape`, `tape.backward` in a batched model, `solver.reset` per window, L-BFGS on stiffness, density and damping from noisy 30 fps recordings and a gripper force sensor (positions alone only fix their ratios), windowed so every rollout stays on the recorded buckling branch. |
| [`fit_capture.py`](fit_capture.py) | The rod model itself from data: free 3D Gaussians from 24 photos (no rod in them), the rod's length, radius and pose read off them, each Gaussian bound to an edge (the inverse skin); then gradients from pixels (gswarp in torch, the image loss handed back to `wp.Tape`): L-BFGS on bend stiffness and damping from one camera's video. Needs `--extra splat` and CUDA. |
| [`experimental/franka_rod.py`](experimental/franka_rod.py) | Two-way coupling with a MuJoCo robot arm (`--extra mujoco`). |

```bash
uv run examples/cable_twist.py
uv run examples/cable_twist.py --viewer null --test
```

## The pattern

Most examples subclass `CableExample` ([`utils/common.py`](utils/common.py)), Newton's example format:

```python
builder = newton.ModelBuilder()
bodies = ADMMDiSMechSolver.add_rod(builder, newton.Rod(points, radius=r), bend_stiffness=k)
ADMMDiSMechSolver.fix_segment(builder, bodies[0])           # clamp the first segment
model = builder.finalize()
self.start(viewer, model, ADMMDiSMechSolver(model, friction=0.5), r)  # states, contact pipeline
self.spin = Drive(model, segment_dofs(model, bodies[0], twist_only=True))
self.drives = (self.spin,)                                   # prescribed every substep
```

`CableExample.step` then runs each frame's substeps (drive, collide, solve) and replays them as one CUDA
graph after the first frame. Clamped DOFs move only through `state.dismech.q`, which `Drive` writes.

[`utils/`](utils) holds what the examples share and the package does not: the frame loop, drives, camera
framing, geometry checks, the theory plots, the fluid drag of `flagella.py` and `cached`, which stores
results that take minutes (`fit_buckling.py`'s recordings and fit) in `.cache/examples`, once per configuration.
[`utils/gaussians.py`](utils/gaussians.py) renders Gaussian splats with gswarp for `fit_capture.py` and hands
the image loss's gradient back to Warp.
