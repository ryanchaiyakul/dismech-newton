"""The S2015 taxel maps from exp/pads/tactile.py: per taxel its normal force (dot area) and shear (arrow, in the
sensor frame), press and pull; dots for the quantized readout's zero taxels."""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = os.environ.get("PAXINI", "../../../scratch/paxini")
r = np.load(sys.argv[1])
R = np.array(json.load(open(f"{D}/S2015.json"))["frames"])
xy = r["taxels"][:, :2] * 1e3
fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.2), sharey=True)
for ax, key, title in zip(axes, ("press", "pull"), ("Press", "Pull")):
    loc, raw = r[key], r[f"{key}_raw"]
    shear = np.einsum("tji,tj->ti", R, np.column_stack([loc[:, :2], np.zeros(len(loc))]))[:, :2]
    on = raw[:, 2] > 0
    ax.scatter(xy[~on, 0], xy[~on, 1], s=6, c="0.75", lw=0)
    sc = ax.scatter(xy[on, 0], xy[on, 1], s=400 * loc[on, 2] / max(loc[:, 2].max(), 1e-9), c=loc[on, 2],
                    cmap="viridis", vmin=0.0, vmax=loc[:, 2].max(), lw=0)
    ax.quiver(xy[:, 0], xy[:, 1], shear[:, 0], shear[:, 1], angles="xy", scale_units="xy",
              scale=max(np.linalg.norm(shear, axis=1).max(), 1e-9) / 3.0, width=0.006, color="C3")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ (mm)")
    fig.colorbar(sc, ax=ax, shrink=0.7, label=r"$F_z$ (N)")
axes[0].set_ylabel(r"$y$ (mm)")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=130)
