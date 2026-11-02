# DiSMech-Newton

[![PyPI](https://img.shields.io/pypi/v/dismech-newton)](https://pypi.org/project/dismech-newton/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://opensource.org/license/gpl-3.0)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![CUDA 13](https://img.shields.io/badge/CUDA-13-76B900)

Discrete elastic rods and midedge shells for [NVIDIA Newton](https://github.com/newton-physics/newton), inspired by
[Daviet (2023)](https://research.nvidia.com/labs/prl/admm_hair/).

## Examples

<table>
  <tr>
    <th width="25%">Cantilever</th>
    <th width="25%">Overhand Knot</th>
    <th width="25%">Plectoneme</th>
    <th width="25%">Flagella Bundling</th>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/ryanchaiyakul/dismech-newton/master/docs/cantilever.webp" alt="Cantilever" width="200"></td>
    <td><img src="https://raw.githubusercontent.com/ryanchaiyakul/dismech-newton/master/docs/knot.webp" alt="Overhand knot" width="200"></td>
    <td><img src="https://raw.githubusercontent.com/ryanchaiyakul/dismech-newton/master/docs/plectoneme.webp" alt="Plectoneme" width="200"></td>
    <td><img src="https://raw.githubusercontent.com/ryanchaiyakul/dismech-newton/master/docs/flagella.webp" alt="Flagella bundling" width="200"></td>
  </tr>
  <tr>
    <td valign="top">DER (top) and Newton's VBD (bottom) against Euler-Bernoulli.<br><code>uv run examples/cantilever.py</code></td>
    <td valign="top">A knot pulled tight with friction <a href="https://doi.org/10.1103/PhysRevLett.99.164301">Audoly et al. (2007)</a>.<br><code>uv run examples/overhand_knot.py</code></td>
    <td valign="top">A twisted rod coils into a plectoneme <a href="https://doi.org/10.1016/j.bpj.2009.02.032">Clauvelin et al. (2009)</a>.<br><code>uv run examples/plectoneme.py</code></td>
    <td valign="top">Flagella bundle in a viscous fluid <a href="https://arxiv.org/abs/2205.10309">Tong et al. (2022)</a>.<br><code>uv run examples/flagella.py --flagella 10</code></td>
  </tr>
  <tr>
    <th width="25%">Shell Cantilever</th>
    <th width="25%">Shell Fold</th>
    <th width="25%">Cloth Twist</th>
    <th width="25%">Franka Pick and Place</th>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/ryanchaiyakul/dismech-newton/master/docs/shell_cantilever.webp" alt="Shell cantilever" width="200"></td>
    <td><img src="https://raw.githubusercontent.com/ryanchaiyakul/dismech-newton/master/docs/shell_fold.webp" alt="A cloth strip folding onto itself" width="200"></td>
    <td><img src="https://raw.githubusercontent.com/ryanchaiyakul/dismech-newton/master/docs/cloth_twist.webp" alt="A cloth sheet wrung into a coiling rope" width="200"></td>
    <td><img src="https://raw.githubusercontent.com/ryanchaiyakul/dismech-newton/master/docs/franka.webp" alt="A Franka arm picking up a cable" width="200"></td>
  </tr>
  <tr>
    <td valign="top">Midedge shells beside Newton's VBD cloth against plate theory.<br><code>uv run examples/shell_cantilever.py</code></td>
    <td valign="top">A cloth strip folds onto itself with self-contact friction.<br><code>uv run examples/shell_fold.py</code></td>
    <td valign="top">A sheet wrung into a coiling rope by self-contact with friction.<br><code>uv run examples/newton/cloth_twist.py</code></td>
    <td valign="top">A MuJoCo Franka picks up a cable.<br><code>uv run --extra mujoco examples/newton/franka_cable_ik_pick_place.py</code></td>
  </tr>
</table>

Newton's own cable and cloth examples (pile, plectoneme, Y junction, hysteresis, cross slide table, Franka pick and
place, hanging and draped cloth, ...) run on `SolverDiSMech` in [`examples/newton/`](examples/newton). See
[`examples/`](examples) for what each example adds.

## Install

Requires Python 3.10+. Runs on an NVIDIA GPU with CUDA 13 (cuDSS through the `gpu` extra) or on the CPU (SciPy).

```bash
pip install "dismech-newton[gpu]"           # add `mujoco` for the coupled examples, `viewer` for the viewer
```

## Usage

```python
import newton
import numpy as np
from dismech_newton import SolverDiSMech

builder = newton.ModelBuilder()
points = np.linspace([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], 21)  # a 1 m rod, 20 segments
bodies, _ = SolverDiSMech.add_rod(builder, rod=newton.Rod(points, radius=0.01), bend_stiffness=1.0e-2)
SolverDiSMech.fix_segment(builder, bodies[0])  # clamp the first segment
model = builder.finalize()
solver = SolverDiSMech(model)
state_0, state_1 = model.state(), model.state()
for _ in range(100):
    solver.step(state_0, state_1, None, None, 1.0e-2)  # finds its own contacts
    state_0, state_1 = state_1, state_0
```

Shells come from `SolverDiSMech.add_cloth_mesh` with Newton's `add_cloth_mesh` arguments.

## Development

```bash
uv sync --extra gpu --extra mujoco --group dev
uv run pytest            # physical checks against theory and finite differences
uv run pytest -m slow    # the examples
```
