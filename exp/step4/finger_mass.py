"""The finger: model mass vs the soft view's proxy mass (MuJoCo's effective), its joint's armature and gains."""
import os, sys
sys.path.insert(0, os.path.abspath("tests"))
from test_examples import _load
M = _load("newton/franka_cable_ik_pick_place")
M.Example.capture = False
ex = M.Example(None, None)
m, v = ex.model, ex.solver.view("soft")
g = [b for b, l in enumerate(m.body_label) if "finger" in l]
print("model mass", m.body_mass.numpy()[g], "view (proxy) mass", v.body_mass.numpy()[ex.fingers])
child = m.joint_child.numpy()
for b in g:
    j = int(list(child).index(b)); d = m.joint_qd_start.numpy()[j]
    print("finger", b, "joint type", m.joint_type.numpy()[j], "armature", m.joint_armature.numpy()[d],
          "ke", m.joint_target_ke.numpy()[d], "kd", m.joint_target_kd.numpy()[d], "h", ex.sim_dt)
