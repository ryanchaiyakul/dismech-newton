# Newton's examples on DiSMech

Newton's own cable and cloth examples (`python -m newton.examples <name>`, solved there by VBD) with
`DiSMechSolver`: the same geometry, material parameters, drives and timing, so the two run side by side. Each keeps
Newton's `test_final` and adds checks of its own (no pass-through, physical bounds). Where the scene had to change,
its docstring says how and why.

| Example | Newton's | What differs |
| --- | --- | --- |
| [`cable_twist.py`](cable_twist.py) | `cable_twist` | The basics: `add_rod` with stiffness and damping, `fix_segment`, the `DiSMechSolver` step, a clamped segment driven through its twist DOF. |
| [`cable_pile.py`](cable_pile.py) | `cable_pile` | Many rods in one model, ground contact and friction. |
| [`cable_plectoneme.py`](cable_plectoneme.py) | `cable_plectoneme` | Both end segments clamped and twisted in opposite senses: self-contact under gravity. The physics against theory is [`../plectoneme.py`](../plectoneme.py). |
| [`cable_bundle_hysteresis.py`](cable_bundle_hysteresis.py) | `cable_bundle_hysteresis` | Kinematic capsules moved by the example (pose and velocity into the state each substep). No Dahl bending hysteresis (Newton's `--no-dahl`): only Coulomb friction holds the bend, and the bundle springs back. |
| [`cloth_bending.py`](cloth_bending.py) | `cloth_bending` | A 12-triangle sheet dropped from 10 m. It lands on its edge without bouncing, grips, and rolls ~4 m over before it falls flat (converged in the step; VBD ends the landing at rest). |
| [`cloth_hanging.py`](cloth_hanging.py) | `cloth_hanging` (`--solver vbd`) | Newton's `add_cloth_grid` as a shell: `mass` per vertex becomes `density` per area. |
| [`cloth_twist.py`](cloth_twist.py) | `cloth_twist` | The sheet self-contact stress test: a square held along two edges, spun until it wrings into a coiling rope; the helicoid before contact against its symmetry. Compressed cloth: its steps end on the iteration budget (`solver.counts`). |

Not yet: [`cloth_rollers.py`](cloth_rollers.py) (Newton's `cloth_rollers`, rollers as kinematic cylinders
turning the cloth by friction) runs ~0.6 s, then its steps stop converging once the tail is pulled round roller 2;
it is not in the tests.

Not ported: `cable_y_junction` (rods sharing a node), `cable_cross_slide_table` and `cloth_poker_cards` (dynamic
rigid bodies pushed by the cable or cloth), `cloth_franka` (a robot in Featherstone coupled to the cloth),
`cloth_h1` and `cloth_style3d` (Style3D's anisotropic cloth and IK).

```bash
uv run examples/newton/cable_twist.py
uv run examples/newton/cable_twist.py --viewer null --test
```
