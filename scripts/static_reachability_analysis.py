"""Static reachability analysis for retargeted motion.

Solves IK from neutral pose with many iterations to determine if target
poses are kinematically reachable. Reports joint limit statistics.

Usage:
    python scripts/static_reachability_analysis.py --bvh_file path/to/motion.bvh --format nokov
"""
import argparse
import pathlib
import numpy as np
import mujoco as mj
import mink
from rich import print
from tqdm import tqdm
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from scipy.spatial.transform import Rotation as R

HERE = pathlib.Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", required=True, type=str)
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="nokov")
    parser.add_argument("--robot", default="DR02_pro")
    parser.add_argument("--max_iter", default=500, type=int)
    parser.add_argument("--tolerance", default=10.0, type=float,
                        help="Position tolerance in mm")
    args = parser.parse_args()

    # Load BVH data
    data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    print(f"Loaded {len(data_frames)} frames, human height={actual_human_height:.3f}m")

    # Create retargeter WITHOUT collision avoidance for pure kinematic analysis
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        use_collision_avoidance=False,
        use_velocity_limit=False,
        verbose=False,
    )

    # Get initial keyframe for reset
    model = retargeter.model
    initial_qpos = model.qpos0.copy()

    # Hand body names
    hand_bodies = {
        "left_wrist_x_link": "LeftHand",
        "right_wrist_x_link": "RightHand",
    }

    # Collect results
    results = []
    
    # Also check which joints hit limits during static solve
    joint_info = []  # list of (name, qpos_adr, lo, hi)
    for jid in range(model.njnt):
        jtype = model.jnt_type[jid]
        if jtype == mj.mjtJoint.mjJNT_HINGE and model.jnt_limited[jid]:
            jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
            lo, hi = model.jnt_range[jid]
            joint_info.append((jname, model.jnt_qposadr[jid], lo, hi))

    print(f"\nStarting static reachability analysis ({args.max_iter} iterations per frame)...")
    
    for i, frame_data in enumerate(tqdm(data_frames, desc="Static IK")):
        # Reset to neutral pose
        retargeter.configuration.data.qpos[:] = initial_qpos
        mj.mj_forward(model, retargeter.configuration.data)

        # Update targets (scaling + offsets)
        retargeter.update_targets(frame_data)

        # Get target positions for hands
        targets = {}
        for rb, hb in hand_bodies.items():
            if hb in retargeter.scaled_human_data:
                targets[rb] = retargeter.scaled_human_data[hb][0].copy()

        # Solve T1 then T2 with many iterations
        dt = model.opt.timestep

        # T1: orientation matching
        if retargeter.use_ik_match_table1:
            for _ in range(min(20, args.max_iter)):
                vel = mink.solve_ik(
                    retargeter.configuration, retargeter.tasks1, dt,
                    retargeter.solver, 0.1, retargeter.ik_limits
                )
                retargeter.configuration.integrate_inplace(vel, dt)

        # T2: position + orientation with many iterations
        if retargeter.use_ik_match_table2:
            prev_err = retargeter.error2()
            for it in range(args.max_iter):
                vel = mink.solve_ik(
                    retargeter.configuration, retargeter.tasks2, dt,
                    retargeter.solver, 0.1, retargeter.ik_limits
                )
                retargeter.configuration.integrate_inplace(vel, dt)
                curr_err = retargeter.error2()
                if abs(prev_err - curr_err) < 1e-8:
                    break
                prev_err = curr_err

        # Get final hand positions
        mj.mj_forward(model, retargeter.configuration.data)
        data = retargeter.configuration.data

        frame_result = {"frame": i, "reachable": True, "hands": {}}
        for rb, hb in hand_bodies.items():
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, rb)
            if bid >= 0 and rb in targets:
                actual_pos = data.xpos[bid].copy()
                target_pos = targets[rb]
                pos_err = np.linalg.norm(actual_pos - target_pos) * 1000  # mm
                z_err = (actual_pos[2] - target_pos[2]) * 1000  # mm

                # Check joint limits at final pose
                joints_at_limit = []
                for jname, qadr, lo, hi in joint_info:
                    if qadr < len(data.qpos):
                        val = data.qpos[qadr]
                        if abs(val - lo) < 0.005 or abs(val - hi) < 0.005:
                            joints_at_limit.append(jname)

                frame_result["hands"][rb] = {
                    "target_z_mm": target_pos[2] * 1000,
                    "actual_z_mm": actual_pos[2] * 1000,
                    "pos_err_mm": pos_err,
                    "z_err_mm": z_err,
                    "joints_at_limit": joints_at_limit,
                    "reachable": pos_err < args.tolerance,
                }
                if pos_err >= args.tolerance:
                    frame_result["reachable"] = False

        results.append(frame_result)

    # Summary
    total = len(results)
    reachable = sum(1 for r in results if r["reachable"])
    unreachable = total - reachable

    print("\n" + "=" * 80)
    print(f"[Static Reachability Analysis] ({args.max_iter} max iterations, tolerance={args.tolerance}mm)")
    print(f"  Total frames:      {total}")
    print(f"  Reachable:         {reachable}  ({reachable/total:.1%})")
    print(f"  Unreachable:       {unreachable}  ({unreachable/total:.1%})")

    # Per-hand analysis
    for rb, hb in hand_bodies.items():
        hand_results = [r["hands"][rb] for r in results if rb in r["hands"]]
        if not hand_results:
            continue
        pos_errs = np.array([h["pos_err_mm"] for h in hand_results])
        z_errs = np.array([h["z_err_mm"] for h in hand_results])
        target_zs = np.array([h["target_z_mm"] for h in hand_results])
        actual_zs = np.array([h["actual_z_mm"] for h in hand_results])

        reach_count = sum(1 for h in hand_results if h["reachable"])
        print(f"\n  --- {rb} ({hb}) ---")
        print(f"    Reachable:        {reach_count}/{len(hand_results)}  ({reach_count/len(hand_results):.1%})")
        print(f"    Position error:   min={pos_errs.min():.1f}mm  max={pos_errs.max():.1f}mm  mean={pos_errs.mean():.1f}mm  median={np.median(pos_errs):.1f}mm")
        print(f"    Z error:           min={z_errs.min():.1f}mm  max={z_errs.max():.1f}mm  mean={z_errs.mean():.1f}mm")
        print(f"    Target Z range:    [{target_zs.min():.1f}mm, {target_zs.max():.1f}mm]")
        print(f"    Actual Z range:    [{actual_zs.min():.1f}mm, {actual_zs.max():.1f}mm]")

        # Count how many frames have joints at limit
        at_limit_count = sum(1 for h in hand_results if h["joints_at_limit"])
        print(f"    Frames with joints at limit: {at_limit_count}/{len(hand_results)}  ({at_limit_count/len(hand_results):.1%})")

        # Show worst 10 unreachable frames
        unreachable_frames = [(i, h) for i, h in enumerate(hand_results) if not h["reachable"]]
        if unreachable_frames:
            unreachable_frames.sort(key=lambda x: x[1]["pos_err_mm"], reverse=True)
            print(f"\n    Top 10 worst unreachable frames:")
            print(f"      {'Frame':>6s} {'TargetZ':>10s} {'ActualZ':>10s} {'PosErr':>10s} {'ZErr':>10s}  Joints at limit")
            for idx, (fi, h) in enumerate(unreachable_frames[:10]):
                jl = ", ".join(h["joints_at_limit"][:5]) if h["joints_at_limit"] else "none"
                print(f"      {fi:6d} {h['target_z_mm']:10.1f} {h['actual_z_mm']:10.1f} {h['pos_err_mm']:10.1f} {h['z_err_mm']:10.1f}  {jl}")

        # Show worst 10 frames by z_err (even if "reachable" overall)
        worst_z = sorted(enumerate(hand_results), key=lambda x: x[1]["z_err_mm"])[:10]
        print(f"\n    Top 10 worst Z-error frames:")
        print(f"      {'Frame':>6s} {'TargetZ':>10s} {'ActualZ':>10s} {'PosErr':>10s} {'ZErr':>10s}  Joints at limit")
        for fi, h in worst_z:
            jl = ", ".join(h["joints_at_limit"][:5]) if h["joints_at_limit"] else "none"
            print(f"      {fi:6d} {h['target_z_mm']:10.1f} {h['actual_z_mm']:10.1f} {h['pos_err_mm']:10.1f} {h['z_err_mm']:10.1f}  {jl}")

    # Joint-at-limit frequency across all frames
    from collections import Counter
    joint_limit_counter = Counter()
    for r in results:
        for rb in r["hands"]:
            for jn in r["hands"][rb].get("joints_at_limit", []):
                joint_limit_counter[jn] += 1

    if joint_limit_counter:
        print(f"\n  --- Joints at limit during static IK (all frames, both hands) ---")
        for jn, cnt in joint_limit_counter.most_common():
            print(f"    {jn:40s}: {cnt:5d} frames ({cnt/total:.1%})")

    print("=" * 80)


if __name__ == "__main__":
    main()
