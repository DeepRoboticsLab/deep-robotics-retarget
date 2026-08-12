"""Visualize scaling: show robot skeleton with original/scaled BVH human.

Compares robot and human skeletons at key frames to validate scaling factors.
Outputs PNG images showing skeleton overlays.

Usage:
    python scripts/visualize_scaling.py --bvh_file path/to/motion.bvh --format nokov
"""
import argparse
import pathlib
import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from rich import print
import json

HERE = pathlib.Path(__file__).parent

BVH_BONES = [
    ("Hips", "Spine2"),
    ("Spine2", "LeftArm"),
    ("Spine2", "RightArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFootMod"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFootMod"),
]

ROBOT_BONES = [
    ("base_link", "body"),
    ("body", "left_shoulder_z_link"),
    ("body", "right_shoulder_z_link"),
    ("left_shoulder_z_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_x_link"),
    ("right_shoulder_z_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_x_link"),
    ("base_link", "left_hip_z_link"),
    ("left_hip_z_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_x_link"),
    ("base_link", "right_hip_z_link"),
    ("right_hip_z_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_x_link"),
]

BVH_TO_ROBOT = {
    "Hips": "base_link", "Spine2": "body",
    "LeftArm": "left_shoulder_z_link", "RightArm": "right_shoulder_z_link",
    "LeftForeArm": "left_elbow_link", "RightForeArm": "right_elbow_link",
    "LeftHand": "left_wrist_x_link", "RightHand": "right_wrist_x_link",
    "LeftUpLeg": "left_hip_z_link", "RightUpLeg": "right_hip_z_link",
    "LeftLeg": "left_knee_link", "RightLeg": "right_knee_link",
    "LeftFootMod": "left_ankle_x_link", "RightFootMod": "right_ankle_x_link",
}


def scale_human_frame(human_frame, root_name, scale_table):
    root_pos = human_frame[root_name][0]
    scaled_root_pos = scale_table[root_name] * root_pos
    result = {root_name: (scaled_root_pos, human_frame[root_name][1])}
    for name in human_frame:
        if name not in scale_table or name == root_name:
            continue
        local = (human_frame[name][0] - root_pos) * scale_table[name]
        result[name] = (local + scaled_root_pos, human_frame[name][1])
    return result


def get_robot_skeleton(model, data):
    positions = {}
    for i in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if name and name in BVH_TO_ROBOT.values():
            positions[name] = data.xpos[i].copy()
    return positions


def draw_skeleton(ax, positions, bones, color, alpha=1.0, linewidth=2, scatter_size=30):
    for a, b in bones:
        if a in positions and b in positions:
            pa = positions[a]
            pb = positions[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                    color=color, alpha=alpha, linewidth=linewidth)
    for name, pos in positions.items():
        bone_names = set()
        for a, b in bones:
            bone_names.add(a)
            bone_names.add(b)
        if name in bone_names:
            ax.scatter(*pos, color=color, alpha=alpha, s=scatter_size, zorder=5)


def draw_frame(ax, frame_idx, bvh_frames, scale_eff, robot_skel, human_root,
               robot_max_reach, show_orig=True, show_scaled=True, show_robot=True,
               align_root=True):
    """Draw one frame on the given axes.
    
    If align_root=True, both skeletons' roots are placed at origin
    so we can compare skeleton shapes directly.
    """
    frame = bvh_frames[frame_idx]
    scaled = scale_human_frame(frame, human_root, scale_eff)

    # Robot root position (for alignment)
    robot_root = robot_skel.get('base_link', np.zeros(3))
    
    if align_root:
        # Shift robot to origin, and use local positions for human
        robot_draw = {k: v - robot_root for k, v in robot_skel.items()}
        
        # Human original: local positions (relative to Hips)
        raw_root = frame[human_root][0]
        human_orig = {n: frame[n][0] - raw_root for n in frame if n in BVH_TO_ROBOT}
        
        # Human scaled: local positions (relative to scaled root)
        # scale_human_data does: scaled_local = (pos - root) * scale, then + scaled_root
        # So scaled_local = scaled_pos - scaled_root = (pos - root) * scale
        human_scaled = {}
        for n in frame:
            if n not in scale_eff or n == human_root:
                continue
            if n in BVH_TO_ROBOT:
                local = (frame[n][0] - raw_root) * scale_eff[n]
                human_scaled[n] = local  # already relative to root=origin
    else:
        robot_draw = robot_skel
        human_orig = {n: frame[n][0] for n in frame if n in BVH_TO_ROBOT}
        human_scaled = {n: scaled[n][0] for n in scaled if n in BVH_TO_ROBOT}

    # Ground plane
    all_pts = []
    if show_robot:
        all_pts.extend(robot_draw.values())
    if show_scaled:
        all_pts.extend(human_scaled.values())
    if show_orig:
        all_pts.extend(human_orig.values())
    all_pts = np.array(list(all_pts))
    margin = 0.15
    xmin, xmax = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
    ymin, ymax = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin
    zmin = min(all_pts[:, 2].min() - margin, -0.1)
    zmax = all_pts[:, 2].max() + margin

    # Ground
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 5),
        np.linspace(ymin, ymax, 5)
    )
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.05, color='brown')

    # Robot
    if show_robot:
        draw_skeleton(ax, robot_draw, ROBOT_BONES, 'blue', 0.8, 2.5, 40)

    # Original human (gray, faded)
    if show_orig:
        draw_skeleton(ax, human_orig, BVH_BONES, 'gray', 0.3, 1.5, 20)

    # Scaled human (red)
    if show_scaled:
        draw_skeleton(ax, human_scaled, BVH_BONES, 'red', 0.9, 2.0, 30)
        # Connect scaled targets to robot joints
        for bvh_name, robot_name in BVH_TO_ROBOT.items():
            if bvh_name in human_scaled and robot_name in robot_draw:
                hp = human_scaled[bvh_name]
                rp = robot_draw[robot_name]
                ax.plot([hp[0], rp[0]], [hp[1], rp[1]], [hp[2], rp[2]],
                        'k--', alpha=0.15, linewidth=0.5)

    # Warnings - compute GLOBAL Z (scaled_root_z + local_z)
    raw_root_z = frame[human_root][0][2]
    scaled_root_z = raw_root_z * scale_eff[human_root]
    warnings = []
    for name in ['LeftHand', 'RightHand']:
        if name in human_scaled:
            local_z = human_scaled[name][2]  # local (relative to root, aligned at origin)
            global_z = scaled_root_z + local_z  # actual global Z
            if global_z < 0.05:
                warnings.append(f"{name} gZ={global_z*1000:.0f}mm")
            # Check reach (distance from root)
            d = np.linalg.norm(human_scaled[name])
            if d > robot_max_reach:
                warnings.append(f"{name} OVER")

    title = f"Frame {frame_idx}"
    if warnings:
        title += "  *** " + " ".join(warnings)
    ax.set_title(title, fontsize=9, color='red' if warnings else 'black')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", required=True, type=str)
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="nokov")
    parser.add_argument("--robot", default="DR02_pro")
    parser.add_argument("--output_dir", default="scaling_vis_output")
    parser.add_argument("--specific_frames", type=int, nargs='*', default=None)
    args = parser.parse_args()

    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    from general_motion_retargeting.params import ROBOT_XML_DICT, IK_CONFIG_DICT

    out_dir = HERE.parent / args.output_dir
    out_dir.mkdir(exist_ok=True)

    # Load
    data_frames, actual_h = load_bvh_file(args.bvh_file, format=args.format)
    with open(IK_CONFIG_DICT[f"bvh_{args.format}"][args.robot]) as f:
        ik_config = json.load(f)
    ratio = actual_h / ik_config["human_height_assumption"]
    scale_table = ik_config["human_scale_table"]
    root_name = ik_config["human_root_name"]
    scale_eff = {k: v * ratio for k, v in scale_table.items()}

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    robot_skel = get_robot_skeleton(model, data)

    # Compute robot max reach (quick, 5000 samples)
    print("Computing robot max reach...")
    np.random.seed(42)
    max_reach = 0
    base_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base_link")
    for _ in range(5000):
        qpos = model.qpos0.copy()
        for jid in range(model.njnt):
            if model.jnt_type[jid] == mj.mjtJoint.mjJNT_HINGE and model.jnt_limited[jid]:
                qadr = model.jnt_qposadr[jid]
                lo, hi = model.jnt_range[jid]
                qpos[qadr] = np.random.uniform(lo, hi)
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        for tip in ["left_wrist_x_link", "right_wrist_x_link"]:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, tip)
            if bid >= 0:
                d = np.linalg.norm(data.xpos[bid] - data.xpos[base_bid])
                if d > max_reach:
                    max_reach = d
    print(f"Robot max reach: {max_reach*1000:.1f}mm")

    print(f"ratio={ratio:.4f}")
    print(f"Scale table (raw -> eff):")
    for k, v in scale_table.items():
        print(f"  {k:20s}: {v:.3f} -> {v*ratio:.4f}")

    # Select key frames
    if args.specific_frames:
        key_frames = args.specific_frames
    else:
        # Find interesting frames automatically
        key_frames = []

        # Frame with max left hand reach
        max_reach_frame = 0
        max_reach_val = 0
        for i, frame in enumerate(data_frames):
            if 'Hips' in frame and 'LeftHand' in frame:
                d = np.linalg.norm(frame['LeftHand'][0] - frame['Hips'][0])
                if d > max_reach_val:
                    max_reach_val = d
                    max_reach_frame = i
        key_frames.append(max_reach_frame)

        # Frame with min left hand Z (the penetrating frame)
        min_z_frame = 0
        min_z_val = 999
        for i, frame in enumerate(data_frames):
            if 'LeftHand' in frame:
                z = frame['LeftHand'][0][2]
                if z < min_z_val:
                    min_z_val = z
                    min_z_frame = i
        key_frames.append(min_z_frame)

        # Frame with max right hand reach
        max_r_frame = 0
        max_r_val = 0
        for i, frame in enumerate(data_frames):
            if 'Hips' in frame and 'RightHand' in frame:
                d = np.linalg.norm(frame['RightHand'][0] - frame['Hips'][0])
                if d > max_r_val:
                    max_r_val = d
                    max_r_frame = i
        key_frames.append(max_r_frame)

        # Frame with min right hand Z
        min_rz_frame = 0
        min_rz_val = 999
        for i, frame in enumerate(data_frames):
            if 'RightHand' in frame:
                z = frame['RightHand'][0][2]
                if z < min_rz_val:
                    min_rz_val = z
                    min_rz_frame = i
        key_frames.append(min_rz_frame)

        # Standing frame (root Z near mean)
        root_zs = [f['Hips'][0][2] for f in data_frames if 'Hips' in f]
        mean_z = np.mean(root_zs)
        standing_frame = min(range(len(data_frames)),
                             key=lambda i: abs(data_frames[i]['Hips'][0][2] - mean_z))
        key_frames.append(standing_frame)

        # Squatting frame (min root Z)
        squat_frame = min(range(len(data_frames)),
                          key=lambda i: data_frames[i].get('Hips', [np.zeros(3)])[0][2])
        key_frames.append(squat_frame)

        # Jumping frame (max root Z)
        jump_frame = max(range(len(data_frames)),
                         key=lambda i: data_frames[i].get('Hips', [np.zeros(3)])[0][2])
        key_frames.append(jump_frame)

        # A few evenly spaced frames
        for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            idx = int(len(data_frames) * frac)
            key_frames.append(idx)

        # Deduplicate and sort
        key_frames = sorted(set(key_frames))

    print(f"\nKey frames: {key_frames}")
    print(f"Labels:")
    print(f"  Blue  = Robot skeleton (neutral pose)")
    print(f"  Red   = Scaled BVH human (IK targets)")
    print(f"  Gray  = Original BVH human (before scaling)")
    print(f"  Black dashed = target-to-robot-joint correspondence")
    print(f"  Red title = warnings (Z<50mm or overreach)")

    # Draw overview grid
    n_frames = len(key_frames)
    n_cols = min(4, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    for idx, fidx in enumerate(key_frames):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        draw_frame(ax, fidx, data_frames, scale_eff, robot_skel,
                   root_name, max_reach)

    fig.suptitle(
        f"Scaling Visualization  |  ratio={ratio:.4f}  |  "
        f"Blue=Robot  Red=Scaled Human  Gray=Original Human",
        fontsize=10
    )
    plt.tight_layout()
    overview_path = out_dir / "scaling_vis_overview.png"
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    print(f"\nOverview saved to: {overview_path}")

    # Also save individual frames
    for fidx in key_frames:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        draw_frame(ax, fidx, data_frames, scale_eff, robot_skel,
                   root_name, max_reach)
        plt.tight_layout()
        path = out_dir / f"scaling_vis_frame_{fidx:04d}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Individual frames saved to: {out_dir}/")
    plt.close('all')


if __name__ == "__main__":
    main()
