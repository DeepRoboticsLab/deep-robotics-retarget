"""Verify scaling factors by comparing robot link lengths vs human dimensions.

Computes robot chain lengths, human body dimensions from BVH, and validates
that scale factors produce correct proportions. Includes workspace coverage.

Usage:
    python scripts/verify_scaling.py --bvh_file path/to/motion.bvh --format nokov
"""
import argparse
import pathlib
import numpy as np
import mujoco as mj
from rich import print
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import ROBOT_XML_DICT, IK_CONFIG_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file
import json

HERE = pathlib.Path(__file__).parent


def get_robot_link_lengths(model):
    """Compute robot link lengths by measuring body positions in neutral pose."""
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    
    # Get body positions in world frame (neutral pose)
    body_pos = {}
    for i in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if name:
            body_pos[name] = data.xpos[i].copy()
    
    return body_pos, data


def compute_robot_chain_lengths(model):
    """Compute key chain lengths from the robot model."""
    body_pos, data = get_robot_link_lengths(model)
    
    # Define chains to measure
    chains = {
        # Left arm
        "left_arm_full": ("base_link", "left_wrist_x_link"),
        "left_shoulder_to_elbow": ("left_shoulder_z_link", "left_elbow_link"),
        "left_elbow_to_wrist": ("left_elbow_link", "left_wrist_x_link"),
        "left_wrist_to_hand_tip": ("left_wrist_x_link", "left_wrist_x_link"),  # mesh extends below
        
        # Right arm
        "right_arm_full": ("base_link", "right_wrist_x_link"),
        "right_shoulder_to_elbow": ("right_shoulder_z_link", "right_elbow_link"),
        "right_elbow_to_wrist": ("right_elbow_link", "right_wrist_x_link"),
        
        # Legs
        "left_leg_full": ("base_link", "left_ankle_x_link"),
        "left_hip_to_knee": ("left_hip_z_link", "left_knee_link"),
        "left_knee_to_ankle": ("left_knee_link", "left_ankle_x_link"),
        "right_leg_full": ("base_link", "right_ankle_x_link"),
        
        # Torso
        "torso": ("base_link", "body"),
        "base_to_floor": ("base_link", "base_link"),  # base_link z position
    }
    
    results = {}
    for name, (body_a, body_b) in chains.items():
        if body_a in body_pos and body_b in body_pos:
            dist = np.linalg.norm(body_pos[body_b] - body_pos[body_a])
            results[name] = dist
            # Also record z for floor reference
            if name == "base_to_floor":
                results[name] = body_pos[body_a][2]  # z height of base_link
    
    # Compute max arm reach: distance from shoulder to wrist when arm is fully extended
    # We need to find the max distance. Let's compute from shoulder_y (root of arm chain)
    arm_roots = {
        "left": "left_shoulder_y_link",
        "right": "right_shoulder_y_link",
    }
    arm_tips = {
        "left": "left_wrist_x_link",
        "right": "right_wrist_x_link",
    }
    
    for side in ["left", "right"]:
        root = arm_roots[side]
        tip = arm_tips[side]
        if root in body_pos and tip in body_pos:
            dist = np.linalg.norm(body_pos[tip] - body_pos[root])
            results[f"{side}_arm_shoulder_to_wrist"] = dist
    
    # Also compute MAX arm reach by sampling configurations
    max_reach = {"left": 0, "right": 0}
    np.random.seed(42)
    for _ in range(20000):
        qpos = model.qpos0.copy()
        for jid in range(model.njnt):
            if model.jnt_type[jid] == mj.mjtJoint.mjJNT_HINGE and model.jnt_limited[jid]:
                qadr = model.jnt_qposadr[jid]
                jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
                if "shoulder" in jname or "elbow" in jname or "wrist" in jname or "waist" in jname:
                    lo, hi = model.jnt_range[jid]
                    qpos[qadr] = np.random.uniform(lo, hi)
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        for side in ["left", "right"]:
            bid_b = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, arm_tips[side])
            bid_a = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base_link")
            if bid_a >= 0 and bid_b >= 0:
                dist = np.linalg.norm(data.xpos[bid_b] - data.xpos[bid_a])
                if dist > max_reach[side]:
                    max_reach[side] = dist
    
    for side in ["left", "right"]:
        results[f"{side}_arm_max_reach"] = max_reach[side]
    
    # Also compute base_link height (floor to base_link)
    results["base_link_height"] = body_pos.get("base_link", np.array([0,0,0]))[2]
    
    return results, body_pos


def compute_human_dimensions(data_frames, root_name="Hips"):
    """Compute human body dimensions from BVH data frames."""
    # Collect root-to-hand distances across all frames
    hand_names = ["LeftHand", "RightHand"]
    foot_names = ["LeftFootMod", "RightFootMod"]
    
    human_dims = {
        "left_arm_reach": [],   # root-to-left-hand distance
        "right_arm_reach": [],  # root-to-right-hand distance
        "left_foot_dist": [],   # root-to-left-foot distance
        "right_foot_dist": [],  # root-to-right-foot distance
        "root_z": [],            # root height
        "left_hand_z": [],
        "right_hand_z": [],
        "left_hand_local": [],   # hand position relative to root
        "right_hand_local": [],
    }
    
    for frame in data_frames:
        if root_name not in frame:
            continue
        root_pos = frame[root_name][0]
        human_dims["root_z"].append(root_pos[2])
        
        for hn, key_prefix in [("LeftHand", "left"), ("RightHand", "right")]:
            if hn in frame:
                hand_pos = frame[hn][0]
                dist = np.linalg.norm(hand_pos - root_pos)
                local = hand_pos - root_pos
                human_dims[f"{key_prefix}_arm_reach"].append(dist)
                human_dims[f"{key_prefix}_hand_z"].append(hand_pos[2])
                human_dims[f"{key_prefix}_hand_local"].append(local)
        
        for fn, key_prefix in [("LeftFootMod", "left"), ("RightFootMod", "right")]:
            if fn in frame:
                foot_pos = frame[fn][0]
                dist = np.linalg.norm(foot_pos - root_pos)
                human_dims[f"{key_prefix}_foot_dist"].append(dist)
    
    # Convert to arrays
    for k in human_dims:
        human_dims[k] = np.array(human_dims[k])
    
    return human_dims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", required=True, type=str)
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="nokov")
    parser.add_argument("--robot", default="DR02_pro")
    args = parser.parse_args()

    # Load BVH
    data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    print(f"Loaded {len(data_frames)} frames, actual_human_height={actual_human_height:.3f}m")

    # Load robot model
    xml_path = str(ROBOT_XML_DICT[args.robot])
    model = mj.MjModel.from_xml_path(xml_path)
    print(f"Robot model: {xml_path}")

    # Load IK config
    with open(IK_CONFIG_DICT[f"bvh_{args.format}"][args.robot]) as f:
        ik_config = json.load(f)
    
    human_height_assumption = ik_config["human_height_assumption"]
    scale_table = ik_config["human_scale_table"]
    root_name = ik_config["human_root_name"]
    
    # Compute scale ratio
    ratio = actual_human_height / human_height_assumption
    print(f"\nhuman_height_assumption = {human_height_assumption}m")
    print(f"actual_human_height = {actual_human_height:.3f}m")
    print(f"ratio = actual/assumption = {ratio:.4f}")
    print(f"\nScale table (raw → effective = raw * ratio):")
    for k, v in scale_table.items():
        print(f"  {k:20s}: {v:.3f} → {v * ratio:.4f}")

    # ============================================================
    # 1. Robot link lengths
    # ============================================================
    print("\n" + "=" * 80)
    print("[1] Robot Link Lengths (from MuJoCo model, neutral pose)")
    print("=" * 80)
    
    robot_lengths, body_pos = compute_robot_chain_lengths(model)
    for name, val in sorted(robot_lengths.items()):
        print(f"  {name:40s}: {val*1000:.1f} mm")
    
    # Robot arm reach (shoulder to wrist)
    robot_left_arm = robot_lengths.get("left_arm_shoulder_to_wrist", 0)
    robot_right_arm = robot_lengths.get("right_arm_shoulder_to_wrist", 0)
    robot_base_height = robot_lengths.get("base_link_height", 0)
    
    # Full arm reach: base_link to wrist
    robot_left_full = robot_lengths.get("left_arm_full", 0)
    robot_right_full = robot_lengths.get("right_arm_full", 0)
    
    # MAX arm reach (from random sampling)
    robot_left_max = robot_lengths.get("left_arm_max_reach", 0)
    robot_right_max = robot_lengths.get("right_arm_max_reach", 0)
    
    print(f"\n  Key dimensions:")
    print(f"    Robot base_link height:          {robot_base_height*1000:.1f} mm")
    print(f"    Robot left arm (base→wrist, neutral):  {robot_left_full*1000:.1f} mm")
    print(f"    Robot right arm (base→wrist, neutral): {robot_right_full*1000:.1f} mm")
    print(f"    Robot left arm (shoulder→wrist):       {robot_left_arm*1000:.1f} mm")
    print(f"    Robot right arm (shoulder→wrist):      {robot_right_arm*1000:.1f} mm")
    print(f"    Robot left arm MAX REACH:              {robot_left_max*1000:.1f} mm")
    print(f"    Robot right arm MAX REACH:             {robot_right_max*1000:.1f} mm")

    # ============================================================
    # 2. Human body dimensions from BVH
    # ============================================================
    print("\n" + "=" * 80)
    print("[2] Human Body Dimensions (from BVH data)")
    print("=" * 80)
    
    human_dims = compute_human_dimensions(data_frames, root_name)
    
    for side in ["left", "right"]:
        reach = human_dims[f"{side}_arm_reach"]
        print(f"\n  {side.capitalize()} arm (Hips→Hand distance):")
        print(f"    Min:    {reach.min()*1000:.1f} mm")
        print(f"    Max:    {reach.max()*1000:.1f} mm")
        print(f"    Mean:   {reach.mean()*1000:.1f} mm")
        print(f"    Median: {np.median(reach)*1000:.1f} mm")
    
    human_root_z = human_dims["root_z"]
    print(f"\n  Human root (Hips) Z height:")
    print(f"    Min:    {human_root_z.min()*1000:.1f} mm")
    print(f"    Max:    {human_root_z.max()*1000:.1f} mm")
    print(f"    Mean:   {human_root_z.mean()*1000:.1f} mm")

    # ============================================================
    # 3. Scale factor verification
    # ============================================================
    print("\n" + "=" * 80)
    print("[3] Scale Factor Verification")
    print("=" * 80)
    
    # The scale table scales local positions (relative to root)
    # So: scaled_local = human_local * scale_factor
    # The "correct" scale factor should be: robot_reach / human_reach
    
    for side in ["left", "right"]:
        human_reach_mean = human_dims[f"{side}_arm_reach"].mean()
        human_reach_max = human_dims[f"{side}_arm_reach"].max()
        
        # Robot arm reach from shoulder (not from base_link)
        # The human "arm reach" is from Hips to Hand
        # The robot equivalent is from base_link to wrist
        if side == "left":
            robot_max = robot_left_max
            robot_neutral = robot_left_full
        else:
            robot_max = robot_right_max
            robot_neutral = robot_right_full
        
        # Current scale factor (effective)
        hand_key = f"{side.capitalize()}Hand"
        arm_key = f"{side.capitalize()}Arm"
        forearm_key = f"{side.capitalize()}ForeArm"
        current_scale = scale_table.get(hand_key, 1.0) * ratio
        
        # Correct scale: robot_max_reach / human_reach
        correct_scale_mean = robot_max / human_reach_mean if human_reach_mean > 0 else 0
        correct_scale_max = robot_max / human_reach_max if human_reach_max > 0 else 0
        
        print(f"\n  {side.capitalize()} arm:")
        print(f"    Human mean reach (Hips→Hand):    {human_reach_mean*1000:.1f} mm")
        print(f"    Human max reach (Hips→Hand):     {human_reach_max*1000:.1f} mm")
        print(f"    Robot neutral reach (base→wrist): {robot_neutral*1000:.1f} mm")
        print(f"    Robot MAX reach (base→wrist):    {robot_max*1000:.1f} mm")
        print(f"    Correct scale (mean vs max):     {correct_scale_mean:.4f}  (= {robot_max*1000:.1f} / {human_reach_mean*1000:.1f})")
        print(f"    Correct scale (max vs max):      {correct_scale_max:.4f}  (= {robot_max*1000:.1f} / {human_reach_max*1000:.1f})")
        print(f"    Current effective scale:          {current_scale:.4f}")
        
        diff = abs(current_scale - correct_scale_mean) / correct_scale_mean * 100 if correct_scale_mean > 0 else 0
        print(f"    Current effective scale:          {current_scale:.4f}")
        print(f"    Difference (vs correct mean):    {diff:.1f}%")
        if diff > 5:
            direction = "OVER" if current_scale > correct_scale_mean else "UNDER"
            print(f"    *** WARNING: Scale mismatch > 5%!")
            print(f"        Current scale {direction}estimates by {diff:.1f}%")
            print(f"        Suggested raw scale (before ratio): {correct_scale_mean / ratio:.3f}")

    # ============================================================
    # 4. Root height comparison
    # ============================================================
    print(f"\n  Root height comparison:")
    print(f"    Human mean Hips height:    {human_root_z.mean()*1000:.1f} mm")
    print(f"    Robot base_link height:    {robot_base_height*1000:.1f} mm")
    root_scale = robot_base_height / human_root_z.mean() if human_root_z.mean() > 0 else 0
    current_root_scale = scale_table.get(root_name, 1.0) * ratio
    print(f"    Correct root scale:         {root_scale:.4f}")
    print(f"    Current effective root scale: {current_root_scale:.4f}")
    root_diff = abs(current_root_scale - root_scale) / root_scale * 100 if root_scale > 0 else 0
    print(f"    Difference:                 {root_diff:.1f}%")

    # ============================================================
    # 5. Workspace coverage analysis
    # ============================================================
    print("\n" + "=" * 80)
    print("[5] Workspace Coverage: How many scaled targets fall within robot reach?")
    print("=" * 80)
    
    for side in ["left", "right"]:
        hand_key = f"{side.capitalize()}Hand"
        if hand_key not in scale_table:
            continue
        
        scale = scale_table[hand_key] * ratio
        root_scale = scale_table.get(root_name, 1.0) * ratio
        
        # Scale the human local hand positions
        local_hands = human_dims[f"{side}_hand_local"]
        scaled_local = local_hands * scale
        
        # Scaled root positions
        scaled_root = human_dims["root_z"] * root_scale
        
        # Scaled global hand z = scaled_local_z + scaled_root_z
        scaled_hand_z = scaled_local[:, 2] + scaled_root
        
        # Robot max reach from base_link (for this side)
        if side == "left":
            robot_max = robot_left_max
        else:
            robot_max = robot_right_max
        
        # Scaled hand distance from root
        scaled_reach = np.linalg.norm(scaled_local, axis=1)
        
        within_reach = scaled_reach <= robot_max
        within_count = within_reach.sum()
        
        print(f"\n  {side.capitalize()} arm:")
        print(f"    Robot max reach (base→wrist):  {robot_max*1000:.1f} mm")
        print(f"    Scaled target reach (mean):    {scaled_reach.mean()*1000:.1f} mm")
        print(f"    Scaled target reach (max):     {scaled_reach.max()*1000:.1f} mm")
        print(f"    Scaled target reach (95th):    {np.percentile(scaled_reach, 95)*1000:.1f} mm")
        print(f"    Within reach: {within_count}/{len(scaled_reach)} ({within_count/len(scaled_reach):.1%})")
        print(f"    Scaled hand Z range: [{scaled_hand_z.min()*1000:.1f}, {scaled_hand_z.max()*1000:.1f}] mm")
        
        # How much overreach?
        overreach = scaled_reach[scaled_reach > robot_max] - robot_max
        if len(overreach) > 0:
            print(f"    Overreach frames: {len(overreach)} ({len(overreach)/len(scaled_reach):.1%})")
            print(f"    Overreach amount: mean={overreach.mean()*1000:.1f}mm, max={overreach.max()*1000:.1f}mm")
        
        # Suggested scale to fit 95% of targets
        target_95 = np.percentile(scaled_reach, 95)
        if target_95 > 0:
            suggested_scale = scale * robot_max / target_95
            print(f"    Suggested scale (fit 95%): {suggested_scale:.4f} (raw: {suggested_scale/ratio:.3f})")

    print("\n" + "=" * 80)
    print("Summary: The 'correct' scale factor = robot_reach / human_reach")
    print("  If current scale differs by >5%, adjust human_scale_table accordingly.")
    print("  Verify on real robot by measuring actual link lengths.")
    print("=" * 80)


if __name__ == "__main__":
    main()
