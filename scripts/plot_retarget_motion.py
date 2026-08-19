"""Visualize retargeted robot motion data.

Runs retargeting (without MuJoCo viewer) and collects per-frame:
  - Root position (base_link XYZ)
  - Root orientation (roll/pitch/yaw)
  - All joint angles
  - Key body world positions (wrists, feet, torso, etc.)

Then generates 4 sets of plots and saves as PNG.

Usage:
  python scripts/plot_retarget_motion.py --bvh_file source_data/nokov_demo/bow.bvh --format nokov
"""

import argparse
import pathlib
import os
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from rich import print

# Key body names for tracking world positions, grouped by body part
KEY_BODIES = {
    "Wrists": ["left_wrist_x_link", "right_wrist_x_link"],
    "Feet":   ["left_ankle_roll_link", "right_ankle_roll_link"],
    "Torso": ["base_link"],
    "Head": ["head_link"],
}

# Joint grouping keywords (match dof_name substrings)
JOINT_GROUPS = {
    "Left Arm": ["left_shoulder", "left_elbow", "left_wrist"],
    "Right Arm": ["right_shoulder", "right_elbow", "right_wrist"],
    "Waist": ["waist", "spine"],
    "Left Leg": ["left_hip", "left_knee", "left_ankle"],
    "Right Leg": ["right_hip", "right_knee", "right_ankle"],
    "Head": ["head", "neck"],
}


def run_retargeting(bvh_file, robot, fmt, collision_avoidance=False):
    """Run retargeting and collect per-frame data, return dict."""
    data_frames, actual_human_height = load_bvh_file(bvh_file, format=fmt)

    retargeter = GMR(
        src_human=f"bvh_{fmt}",
        tgt_robot=robot,
        actual_human_height=actual_human_height,
        use_collision_avoidance=collision_avoidance,
        verbose=False,
    )

    model = retargeter.model
    base_body_name = None
    from general_motion_retargeting.params import ROBOT_BASE_DICT
    base_body_name = ROBOT_BASE_DICT.get(robot, "base_link")

    # Get all body names
    all_body_names = []
    for i in range(model.nbody):
        bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        all_body_names.append(bname)

    # Determine key body IDs (filter non-existent ones)
    key_body_ids = {}
    for group, names in KEY_BODIES.items():
        ids = []
        for bn in names:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bn)
            if bid >= 0:
                ids.append((bn, bid))
        if ids:
            key_body_ids[group] = ids

    # Collect dof names
    dof_names = []
    for i in range(model.nv):
        dn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, model.dof_jntid[i])
        dof_names.append(dn if dn else f"dof_{i}")

    # Collect joint limits
    jnt_ranges = {}
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mj.mjtJoint.mjJNT_HINGE and model.jnt_limited[jid]:
            qadr = model.jnt_qposadr[jid]
            jnt_ranges[qadr] = (model.jnt_range[jid][0], model.jnt_range[jid][1])

    # Run retargeting
    root_pos_list = []
    root_rot_list = []   # wxyz quaternion
    dof_pos_list = []
    body_pos_dict = {}   # body_name -> list of [x,y,z]
    for group, items in key_body_ids.items():
        for bn, _ in items:
            body_pos_dict[bn] = []

    pbar = tqdm(total=len(data_frames), desc="Retargeting")
    for i, frame_data in enumerate(data_frames):
        qpos = retargeter.retarget(frame_data)

        # Root position and orientation
        root_pos_list.append(qpos[:3].copy())
        root_rot_list.append(qpos[3:7].copy())  # wxyz

        # qpos structure: [base_pos(3), base_quat(4), joint_angles(n_dof)]
        n_dof = model.nv - 6  # subtract base 6 DoF (free joint)
        # DR02_pro uses floating joint base, qpos = [3 pos, 4 quat, n_dof angles]
        dof_pos = qpos[7:].copy()
        dof_pos_list.append(dof_pos)

        # Key body world positions
        mj.mj_forward(model, retargeter.configuration.data)
        data = retargeter.configuration.data
        for group, items in key_body_ids.items():
            for bn, bid in items:
                body_pos_dict[bn].append(data.xpos[bid].copy())

        pbar.update(1)
    pbar.close()

    root_pos = np.array(root_pos_list)
    root_rot = np.array(root_rot_list)
    dof_pos = np.array(dof_pos_list)

    # body positions
    body_positions = {}
    for bn, lst in body_pos_dict.items():
        body_positions[bn] = np.array(lst)

    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "dof_names": dof_names,
        "body_positions": body_positions,
        "key_body_ids": key_body_ids,
        "jnt_ranges": jnt_ranges,
        "n_frames": len(data_frames),
        "retargeter": retargeter,
    }


def plot_root_position(root_pos, output_dir, n_frames):
    """Plot root position X/Y/Z over frames."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
    t = np.arange(n_frames)
    labels = ["X (m)", "Y (m)", "Z (m)"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i in range(3):
        axes[i].plot(t, root_pos[:, i], color=colors[i], linewidth=2.0)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
        axes[i].margins(y=0.05)
    axes[0].set_title("Base Link Position Over Time")
    axes[-1].set_xlabel("Frame")
    plt.tight_layout(pad=0.5)
    path = os.path.join(output_dir, "01_root_position.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_root_orientation(root_rot, output_dir, n_frames):
    """Plot root orientation (roll/pitch/yaw) over frames."""
    # root_rot is wxyz format, convert to euler (roll, pitch, yaw)
    # scipy Rotation.from_quat expects xyzw
    rots_xyzw = root_rot[:, [1, 2, 3, 0]]
    euler_angles = Rotation.from_quat(rots_xyzw).as_euler("xyz", degrees=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
    t = np.arange(n_frames)
    labels = ["Roll (deg)", "Pitch (deg)", "Yaw (deg)"]
    colors = ["#9b59b6", "#e67e22", "#1abc9c"]
    for i in range(3):
        axes[i].plot(t, euler_angles[:, i], color=colors[i], linewidth=2.0)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
        axes[i].margins(y=0.05)
    axes[0].set_title("Base Link Orientation (Euler Angles) Over Time")
    axes[-1].set_xlabel("Frame")
    plt.tight_layout(pad=0.5)
    path = os.path.join(output_dir, "02_root_orientation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_joint_angles(dof_pos, dof_names, jnt_ranges, output_dir, n_frames):
    """Plot joint angles over frames, grouped by body part."""
    n_dof = dof_pos.shape[1]
    t = np.arange(n_frames)

    # Assign each dof to a group
    dof_groups = {g: [] for g in JOINT_GROUPS}
    ungrouped = []
    for i in range(n_dof):
        name = dof_names[i] if i < len(dof_names) else f"dof_{i}"
        name_lower = name.lower()
        assigned = False
        for group, keywords in JOINT_GROUPS.items():
            if any(kw in name_lower for kw in keywords):
                dof_groups[group].append((i, name))
                assigned = True
                break
        if not assigned:
            ungrouped.append((i, name))

    # Add ungrouped joints as a separate group if any
    if ungrouped:
        dof_groups["Other"] = ungrouped

    # Filter empty groups
    active_groups = {g: items for g, items in dof_groups.items() if items}
    n_groups = len(active_groups)

    fig, axes = plt.subplots(n_groups, 1, figsize=(14, 3 * n_groups), sharex=True)
    if n_groups == 1:
        axes = [axes]

    # Color cycle
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    for ax_idx, (group, items) in enumerate(active_groups.items()):
        ax = axes[ax_idx]
        for idx, (dof_i, dof_name) in enumerate(items):
            color = colors[idx % len(colors)]
            # Convert radians to degrees
            values_deg = np.degrees(dof_pos[:, dof_i])
            ax.plot(t, values_deg, color=color, linewidth=1.8, label=dof_name)

            # Draw joint limit lines
            if dof_i in jnt_ranges:
                lo, hi = jnt_ranges[dof_i]
                ax.axhline(y=np.degrees(lo), color=color, linestyle=":", linewidth=1.2, alpha=0.5)
                ax.axhline(y=np.degrees(hi), color=color, linestyle=":", linewidth=1.2, alpha=0.5)

        ax.set_ylabel("Angle (deg)")
        ax.set_title(f"{group} ({len(items)} joints)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        # Limit legend entries for large groups
        if len(items) > 12:
            ax.legend(fontsize=6, loc="upper right", ncol=3)

    axes[-1].set_xlabel("Frame")
    fig.suptitle("Joint Angles Over Time (dotted = joint limits)", fontsize=14)
    plt.tight_layout(pad=0.5, h_pad=0.8)
    path = os.path.join(output_dir, "03_joint_angles.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize retargeted robot motion data")
    parser.add_argument("--bvh_file", required=True, type=str, help="Path to BVH motion file")
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--robot", default="DR02_pro", type=str)
    parser.add_argument("--output_dir", default="plots", type=str, help="Output directory")
    parser.add_argument("--collision_avoidance", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"BVH file: {args.bvh_file}")
    print(f"Robot: {args.robot}, format: {args.format}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Run retargeting
    results = run_retargeting(
        args.bvh_file, args.robot, args.format, args.collision_avoidance
    )

    n = results["n_frames"]
    print(f"\nRetargeting complete, {n} frames total")
    print(f"  root_pos shape: {results['root_pos'].shape}")
    print(f"  dof_pos shape:  {results['dof_pos'].shape}")
    print(f"  body_positions: {list(results['body_positions'].keys())}")
    print()

    # Generate plots
    print("Generating plots...")
    plot_root_position(results["root_pos"], args.output_dir, n)
    plot_root_orientation(results["root_rot"], args.output_dir, n)
    plot_joint_angles(
        results["dof_pos"], results["dof_names"], results["jnt_ranges"], args.output_dir, n
    )

    print(f"\nAll plots saved to {args.output_dir}/")



if __name__ == "__main__":
    main()
