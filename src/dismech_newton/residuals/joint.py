"""Single-dataclass Stribeck + odd RFF joint torque residual."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp
from newton import Control, JointType, Model

from ..constants import ADJ_MODULE_OPTS
from .base import as_rff_weights, as_world_param


@wp.kernel(module="unique", module_options=ADJ_MODULE_OPTS)
def _joint_residual_kernel(
    q: wp.array[float],
    qd: wp.array[float],
    u_in: wp.array[float],
    dm: wp.array[wp.vec4i],
    fr: wp.array3d[float],
    wj: wp.array3d[float],
    cl: wp.array2d[float],
    st: wp.array2d[float],
    rff_dim: int,
    qs: float,
    vsc: float,
    usc: float,
    vs: float,
    eps: float,
    ov: int,
    cap: float,
    sc: float,
    out: wp.array[float],
):
    d = wp.tid()
    m = dm[d]
    uc = u_in[d]
    if m[3] == 0:
        out[d] = uc
        return

    w, l, qa, v = m[0], m[1], q[m[2]], qd[d]
    fc = cl[w if cl.shape[0] > 1 else 0, l]
    fs = st[w if st.shape[0] > 1 else 0, l]
    vr = v / vs
    t_st = -(fc + fs * wp.exp(-(vr * vr))) * (v / wp.sqrt(v * v + eps * eps))
    wi, qn, vn, un, tr = (w if wj.shape[0] > 1 else 0), qa / qs, v / vsc, uc / usc, float(0.0)
    for k in range(rff_dim):
        a0 = fr[l, k, 0] * qn + fr[l, k, 3]
        f1 = fr[l, k, 1] * vn
        f2 = fr[l, k, 2] * un
        dv = f1 + f2
        ev = wp.exp(-0.5 * (f1 * f1 + f2 * f2)) if ov != 0 else 0.0
        s2 = wp.sin(0.5 * dv)
        tr += wj[wi, l, k] * ((wp.sin(dv) - dv * ev) * wp.cos(a0) - 2.0 * s2 * s2 * wp.sin(a0))
    out[d] = uc + cap * wp.tanh((t_st + sc * tr) / cap)


@dataclass
class JointResidual:
    """Pre-allocated pure-Warp robot-owned joint Stribeck + odd-RFF residual."""

    rff_dim: int = 32
    rff_bandwidth: float = 1.5
    q_scale: float = 1.0
    qd_scale: float = 1.0
    u_scale: float = 10.0
    stribeck_v: float = 0.05
    eps_vel: float = 1.0e-3
    ortho_viscous: bool = True
    tau_cap: float = 50.0
    coulomb: float | np.ndarray | wp.array = 0.0
    stribeck: float | np.ndarray | wp.array = 0.0
    w_joint: np.ndarray | wp.array | None = None
    enabled_dofs: tuple[int, ...] | None = None
    seed: int = 17

    device: Any = field(default=None, init=False, repr=False)
    n_dof: int = field(default=0, init=False, repr=False)
    dof_map: wp.array | None = field(default=None, init=False, repr=False)
    rff_freq: wp.array | None = field(default=None, init=False, repr=False)
    joint_f_eff: wp.array | None = field(default=None, init=False, repr=False)
    params: tuple[wp.array, ...] = field(default=(), init=False, repr=False)
    _args: tuple[Any, ...] = field(default=(), init=False, repr=False)
    _ctrl: Control | None = field(default=None, init=False, repr=False)
    _zero_u: wp.array | None = field(default=None, init=False, repr=False)
    _none_ctrl: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _src: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _bound: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            self.rff_dim < 0
            or self.rff_bandwidth < 0.0
            or min(self.q_scale, self.qd_scale, self.u_scale, self.stribeck_v, self.eps_vel, self.tau_cap) <= 0.0
        ):
            raise ValueError("Invalid JointResidual parameters")

    def bind(self, model: Model, device=None) -> "JointResidual":
        """Allocate/bind joint residual buffers onto ``model`` on ``device``."""
        self._src = {"coulomb": self.coulomb, "stribeck": self.stribeck, "w_joint": self.w_joint}
        f = getattr(model, "_parent", model)
        dev = wp.get_device(f.device if device is None else device)
        self.device = dev
        self.n_dof = int(f.joint_dof_count)
        nq = int(f.joint_coord_count)
        nj = int(f.joint_count)

        jw = f.joint_world.numpy() if nj else np.zeros(0, np.int32)
        gm = bool(f.world_count > 1 and (jw < 0).any())
        dm = np.zeros((max(self.n_dof, 1), 4), np.int32)
        if self.n_dof and nj:
            qds = f.joint_qd_start.numpy()
            d = np.arange(self.n_dof, dtype=np.int32)
            ji = np.repeat(np.arange(nj, dtype=np.int32), np.diff(qds))
            w = jw[ji]
            jt = np.isin(f.joint_type.numpy()[ji], (int(JointType.PRISMATIC), int(JointType.REVOLUTE)))
            dl = d - np.where(w >= 0, f.joint_dof_world_start.numpy()[np.maximum(w, 0)], 0)
            dm[:, 0] = 0 if gm else np.maximum(w, 0)
            dm[:, 1] = dl
            dm[:, 2] = np.minimum(f.joint_q_start.numpy()[ji] + d - qds[ji], max(nq, 1) - 1)
            dm[:, 3] = jt if self.enabled_dofs is None else (jt & np.isin(dl, self.enabled_dofs))

        n_worlds = 1 if gm else max(1, int(f.world_count))
        n_rff = max(self.rff_dim, 1)
        rng = np.random.default_rng(self.seed)
        d_local = max(1, int(dm[:, 1].max()) + 1 if self.n_dof else 1)
        self.dof_map = wp.array(dm, dtype=wp.vec4i, device=dev)
        self.rff_freq = wp.array(
            np.concatenate(
                [rng.normal(0.0, self.rff_bandwidth, (d_local, n_rff, 3)), rng.uniform(0.0, 2.0 * np.pi, (d_local, n_rff, 1))],
                axis=-1,
            ).astype(np.float32),
            dtype=float,
            device=dev,
        )
        self.w_joint = as_rff_weights(self.w_joint, float, d_local, n_rff, n_worlds, dev, "w_joint")
        self.coulomb = as_world_param(self.coulomb, float, d_local, n_worlds, dev, "coulomb")
        self.stribeck = as_world_param(self.stribeck, float, d_local, n_worlds, dev, "stribeck")
        self._args = (
            int(self.rff_dim),
            float(self.q_scale),
            float(self.qd_scale),
            float(self.u_scale),
            float(self.stribeck_v),
            float(self.eps_vel),
            int(self.ortho_viscous),
            float(self.tau_cap),
            (2.0 / self.rff_dim) ** 0.5 if self.rff_dim else 0.0,
        )
        self.joint_f_eff = wp.zeros(self.n_dof, dtype=float, device=dev, requires_grad=True)
        self._ctrl = Control()
        self._zero_u = wp.zeros(self.n_dof, dtype=float, device=dev)
        self._none_ctrl = dict.fromkeys(self._ctrl.__dict__)
        self.params = (self.w_joint, self.coulomb, self.stribeck)
        self._bound = True
        return self

    def apply(self, state_in, control: Control | None) -> Control | None:
        """Evaluate joint residual into ``self.joint_f_eff`` and wrap in reusable ``Control``."""
        if self.n_dof:
            u = control.joint_f if (control is not None and control.joint_f is not None) else self._zero_u
            wp.launch(
                _joint_residual_kernel,
                self.n_dof,
                device=self.device,
                outputs=[self.joint_f_eff],
                inputs=[state_in.joint_q, state_in.joint_qd, u, self.dof_map, self.rff_freq, *self.grad_inputs(), *self._args],
            )
            self._ctrl.__dict__.clear()
            self._ctrl.__dict__.update(control.__dict__ if control is not None else self._none_ctrl)
            self._ctrl.joint_f = self.joint_f_eff
        return self._ctrl if self.n_dof else control

    def grad_inputs(self) -> list[wp.array]:
        """Differentiable parameter arrays passed to ``_joint_residual_kernel`` and tracked by adjoint."""
        return [self.w_joint, self.coulomb, self.stribeck]
