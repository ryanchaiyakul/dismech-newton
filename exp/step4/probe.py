"""Record what _lift_kernel lifts: import (pytest -p probe with PYTHONPATH=exp/step4) or run.py wrapper.
Per call kind (start detection / iterate detection), body kind (yielding proxy / other shape): lifted pairs,
deepest start gap in thicknesses (n.c / th; < 0: inside)."""
import atexit, sys
import warp as wp
from dismech_newton import contact as C

@wp.kernel
def _probe_kernel(q: wp.array[float], lift: float, s: C.Slots, arm: C.Arm, start: int,
                  n_out: wp.array[wp.int32], deep: wp.array[float],
                  info: wp.array[float], cnt: wp.array[wp.int32]):
    i = wp.tid()
    if s.active[i] == 0 or wp.min(s.w[i]) < 0.0:
        return
    c = C.contact_point(q, s.pair[i], s.w[i], s.anchor[i])
    b = C.arm_body(arm, s, i)
    if b >= 0 and q.shape[0] > arm.start:
        c -= C.arm_move(q, arm, b, s.anchor[i] - arm.com[b])
    g = wp.dot(s.normal[i], c) / s.thickness[i]
    k = 2 * (1 - start) + wp.where(b >= 0, 1, 0)  # 0 start/other 1 start/proxy 2 iter/other 3 iter/proxy
    if g < lift:
        wp.atomic_add(n_out, k, 1)
        if b >= 0 and q.shape[0] > arm.start:
            j = arm.base[b]
            wp.atomic_max(info, 0, wp.length(wp.vec3(q[j], q[j + 1], q[j + 2])))
            wp.atomic_max(info, 1, wp.length(wp.vec3(q[j + 3], q[j + 4], q[j + 5])))
            wp.atomic_max(info, 2, s.thickness[i])
            wp.atomic_min(info, 3, s.thickness[i])
            wp.atomic_add(cnt, 0, 1)
    wp.atomic_min(deep, k, g)

N = wp.zeros(4, dtype=wp.int32)
D = wp.full(4, 1.0e9, dtype=float)
I = wp.array([0.0, 0.0, 0.0, 1.0e9], dtype=float)
K = wp.zeros(1, dtype=wp.int32)
orig = C.BarrierBlock._launch

def _launch(self, kernel, inputs, dim=None):
    if kernel is C._lift_kernel and self.count:
        q, lift, s, arm = inputs
        start = int(abs(lift - float(C.LIFT)) < 1e-9)
        wp.launch(_probe_kernel, dim=self.count, inputs=[q, lift / 1.0 if False else lift, s, arm, start, N, D, I, K],
                  device=self.device)
    return orig(self, kernel, inputs, dim)

C.BarrierBlock._launch = _launch

@atexit.register
def report():
    n, d = N.numpy(), D.numpy()
    names = ["start/other", "start/proxy", "iter/other", "iter/proxy"]
    print("LIFT " + " ".join(sys.argv[:1]) + ": " + ", ".join(
        f"{nm} lifted {n[k]} deepest {d[k]:.3g}" for k, nm in enumerate(names) if d[k] < 1e9)
          + f" | proxy iter lifts {K.numpy()[0]}: max |dx| {I.numpy()[0]:.3g} max |dth| {I.numpy()[1]:.3g} th {I.numpy()[3]:.3g}-{I.numpy()[2]:.3g}", flush=True)
