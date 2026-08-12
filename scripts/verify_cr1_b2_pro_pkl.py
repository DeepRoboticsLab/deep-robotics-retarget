"""Verify script: check bvh_to_robot.py --robot DR02_pro --format nokov output pkl content and joint order.

All verification is done by actually loading the model and running retargeting.
"""
import sys
import os
import pickle
import numpy as np
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import mujoco as mj
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.params import ROBOT_XML_DICT
from rich import print

ROBOT = "DR02_pro"
BVH_FORMAT = "nokov"

# 1) Select a BVH file
HERE = pathlib.Path(__file__).parent.parent
bvh_candidates = [
    HERE / "source_data" / "nokov_demo" / "getup.bvh",
    HERE / "source_data" / "nokov_demo" / "liedown.bvh",
    HERE / "source_data" / "nokov_demo" / "jugong.bvh",
]
bvh_file = None
for c in bvh_candidates:
    if c.exists():
        bvh_file = str(c)
        break
assert bvh_file is not None, "No BVH file found"
print(f"[Verify] Using BVH file: {bvh_file}")

# 2) Load robot model and print qpos layout (joint name -> qpos address)
xml_file = str(ROBOT_XML_DICT[ROBOT])
print(f"[Verify] Robot XML: {xml_file}")
model = mj.MjModel.from_xml_path(xml_file)

print("\n" + "=" * 70)
print("[Step 1] MuJoCo model qpos layout (joint name -> qpos address)")
print("=" * 70)
print(f"nq (qpos dim) = {model.nq}")
print(f"nv (qvel dim) = {model.nv}")
print(f"njnt (num joints) = {model.njnt}")
print()

qpos_joint_map = []  # (qpos_adr, joint_name, joint_type)
for jid in range(model.njnt):
    jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
    jtype = model.jnt_type[jid]
    qadr = model.jnt_qposadr[jid]
    jtype_str = {
        mj.mjtJoint.mjJNT_FREE: "free(7)",
        mj.mjtJoint.mjJNT_HINGE: "hinge(1)",
        mj.mjtJoint.mjJNT_SLIDE: "slide(1)",
        mj.mjtJoint.mjJNT_BALL: "ball(4)",
    }.get(jtype, str(jtype))
    qpos_joint_map.append((qadr, jname, jtype_str))
    print(f"  joint_id={jid:2d}  qpos_adr={qadr:2d}  type={jtype_str:10s}  name={jname}")

# 3) Run retargeting to generate a real pkl
print("\n" + "=" * 70)
print("[Step 2] Run retargeting to generate pkl")
print("=" * 70)

lafan1_data_frames, actual_human_height = load_bvh_file(bvh_file, format=BVH_FORMAT)
print(f"  Actual human height: {actual_human_height}")
print(f"  BVH frame count: {len(lafan1_data_frames)}")

retargeter = GMR(
    src_human=f"bvh_{BVH_FORMAT}",
    tgt_robot=ROBOT,
    actual_human_height=actual_human_height,
    verbose=False,
)

qpos_list = []
N_RUN = min(10, len(lafan1_data_frames))  # First 10 frames is enough for verification
for i in range(N_RUN):
    qpos = retargeter.retarget(lafan1_data_frames[i])
    qpos_list.append(qpos)

# Replicate the save logic from bvh_to_robot.py
root_pos = np.array([qpos[:3] for qpos in qpos_list])
root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])  # wxyz -> xyzw
dof_pos = np.array([qpos[7:] for qpos in qpos_list])

motion_data = {
    "fps": 200,
    "root_pos": root_pos,
    "root_rot": root_rot,
    "dof_pos": dof_pos,
    "local_body_pos": None,
    "link_body_list": None,
}

out_pkl = "/tmp/verify_DR02_pro.pkl"
with open(out_pkl, "wb") as f:
    pickle.dump(motion_data, f)
print(f"  Saved: {out_pkl}")

# 4) Load pkl and check content
print("\n" + "=" * 70)
print("[Step 3] Load pkl and check content")
print("=" * 70)
with open(out_pkl, "rb") as f:
    data = pickle.load(f)

print(f"  pkl keys: {list(data.keys())}")
for k, v in data.items():
    if isinstance(v, np.ndarray):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")

print(f"\n  root_pos example (frame 0): {data['root_pos'][0]}")
print(f"  root_rot example (frame 0): {data['root_rot'][0]}  (xyzw)")
print(f"  dof_pos example (frame 0): {data['dof_pos'][0]}")
print(f"  dof_pos columns (joint count): {data['dof_pos'].shape[1] if data['dof_pos'].ndim == 2 else 'N/A'}")

# 5) Map dof_pos columns to joint names
print("\n" + "=" * 70)
print("[Step 4] dof_pos column index -> joint name mapping")
print("=" * 70)

# Each scalar starting from qpos index 7 corresponds to a hinge joint (sorted by qpos_adr)
hinge_joints = []
for qadr, jname, jtype_str in qpos_joint_map:
    if jtype_str.startswith("hinge") or jtype_str.startswith("slide"):
        hinge_joints.append((qadr, jname))

hinge_joints.sort(key=lambda x: x[0])
print(f"  hinge/slide joint count: {len(hinge_joints)}")
print(f"  dof_pos column count: {data['dof_pos'].shape[1] if data['dof_pos'].ndim == 2 else 'N/A'}")
print()
print(f"  {'dof_pos_col':<14s} {'qpos_idx':<10s} {'joint_name'}")
print("  " + "-" * 60)
for i, (qadr, jname) in enumerate(hinge_joints):
    dof_idx = i  # dof_pos[i] == qpos[7 + i]
    qpos_idx = 7 + i
    match = "OK" if qadr == qpos_idx else f"MISMATCH(qpos_adr={qadr})"
    print(f"  {dof_idx:<14d} {qpos_idx:<10d} {jname}   [{match}]")

print("\n[Verify] Verification complete.")
