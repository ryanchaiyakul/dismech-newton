# Examples

Each example is one file, runnable on its own; `--viewer null --test` runs it headless and checks it.
Read them in this order: each adds one thing to the last.

| Example | What it adds |
| --- | --- |
| [`cable_twist.py`](cable_twist.py) | The basics: `add_rod` with stiffness and damping, `fix_segment`, the `ADMMDiSMechSolver` step, a clamped segment driven through its twist. |
| [`cable_pile.py`](cable_pile.py) | Many rods in one model, ground contact and friction. |
| [`cantilever.py`](cantilever.py) | Both solvers (`ADMMDiSMechSolver`, `DiSMechSolver`) and their integrators (`theta`), against a beam formula. |
| [`overhand_knot.py`](overhand_knot.py) | Self-contact with friction, both ends clamped and pulled, a measured force against theory. |
| [`cable_plectoneme.py`](cable_plectoneme.py) | A rod from material constants (`newton.Rod`), frictionless self-contact under twist, against Clauvelin et al. (2009). |
| [`flagella.py`](flagella.py) | External forces on the nodes (`state.particle_f`: fluid drag) and contact between rods. |
| [`fit_stiffness.py`](fit_stiffness.py) | Gradients through the solver: `wp.Tape`, `tape.backward`, L-BFGS on the bending stiffness and damping. |
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
framing, geometry checks, the theory plots and the fluid drag of `flagella.py`.
