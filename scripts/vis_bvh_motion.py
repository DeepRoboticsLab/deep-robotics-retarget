"""Play a BVH skeleton with GMR's parser and the MuJoCo viewer."""

import argparse
import pathlib
import time

import glfw
import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


SCENE_XML = """
<mujoco model="bvh_skeleton">
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.20 0.25 1"/>
    <global azimuth="135" elevation="-20"/>
  </visual>
  <worldbody>
    <geom name="floor" type="plane" size="50 50 0.1" rgba="0.18 0.20 0.22 1"/>
  </worldbody>
</mujoco>
"""

HAND_POSE_PAIRS = (
    ("Left", "LeftForeArm", "LeftHand"),
    ("Right", "RightForeArm", "RightHand"),
)


def read_fps(bvh_path: pathlib.Path) -> float:
    with bvh_path.open("r", encoding="utf-8", errors="ignore") as bvh_file:
        for line in bvh_file:
            if line.lstrip().lower().startswith("frame time:"):
                frame_time = float(line.split(":", 1)[1].strip())
                if frame_time <= 0:
                    break
                return 1.0 / frame_time
    raise ValueError(f"Frame Time not found or invalid in {bvh_path}")


def load_motion(bvh_path: pathlib.Path, bvh_format: str):
    frames, _ = load_bvh_file(str(bvh_path), format=bvh_format)
    animation = read_bvh(str(bvh_path))
    edges = [
        (animation.bones[parent], animation.bones[index])
        for index, parent in enumerate(animation.parents)
        if parent >= 0
    ]
    return frames, animation.bones, edges


def relative_pose_series(frames, reference_joint, target_joint):
    """Return target positions and rotations expressed in the reference frame."""
    reference_positions = np.asarray([frame[reference_joint][0] for frame in frames])
    target_positions = np.asarray([frame[target_joint][0] for frame in frames])
    reference_rotations = R.from_quat(
        np.asarray([frame[reference_joint][1] for frame in frames]),
        scalar_first=True,
    )
    target_rotations = R.from_quat(
        np.asarray([frame[target_joint][1] for frame in frames]),
        scalar_first=True,
    )
    relative_positions = reference_rotations.inv().apply(
        target_positions - reference_positions
    )
    relative_rotations = reference_rotations.inv() * target_rotations
    return relative_positions, relative_rotations


def plot_hand_poses(frames, fps, output_path, euler_order):
    """Plot each hand 6-DoF pose in its elbow coordinate frame."""
    import matplotlib.pyplot as plt

    available_pairs = [
        pair
        for pair in HAND_POSE_PAIRS
        if all(pair[1] in frame and pair[2] in frame for frame in frames)
    ]
    if not available_pairs:
        expected = ", ".join(f"{elbow}/{hand}" for _, elbow, hand in HAND_POSE_PAIRS)
        raise ValueError(f"No hand/elbow joint pairs found; expected one of: {expected}")

    times = np.arange(len(frames), dtype=np.float64) / fps
    figure, axes = plt.subplots(
        2,
        len(available_pairs),
        figsize=(7 * len(available_pairs), 8),
        squeeze=False,
        sharex="col",
    )
    colors = ("tab:red", "tab:green", "tab:blue")

    for column, (side, elbow, hand) in enumerate(available_pairs):
        positions, rotations = relative_pose_series(frames, elbow, hand)
        euler_radians = rotations.as_euler(euler_order, degrees=False)
        euler_degrees = np.rad2deg(np.unwrap(euler_radians, axis=0))

        for axis, color in enumerate(colors):
            axis_name = "XYZ"[axis]
            axes[0, column].plot(
                times, positions[:, axis], color=color, label=f"{elbow} {axis_name}"
            )
            axes[1, column].plot(
                times,
                euler_degrees[:, axis],
                color=color,
                label=f"{euler_order[axis].upper()} angle",
            )

        axes[0, column].set_title(f"{side} hand relative to {elbow}")
        axes[0, column].set_ylabel("Position (m)")
        axes[1, column].set_ylabel("Unwrapped Euler angle (deg)")
        axes[1, column].set_xlabel("Time (s)")
        for row in range(2):
            axes[row, column].grid(True, alpha=0.3)
            axes[row, column].legend(loc="best")

    figure.suptitle(
        f"Hand poses in elbow frames (Euler order: {euler_order.upper()})"
    )
    figure.tight_layout()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def add_capsule(viewer, start, end, radius):
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        size=[radius, radius, radius],
        pos=np.zeros(3),
        mat=np.eye(3).ravel(),
        rgba=[0.72, 0.78, 0.86, 1.0],
    )
    mj.mjv_connector(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        width=radius,
        from_=np.asarray(start, dtype=np.float64),
        to=np.asarray(end, dtype=np.float64),
    )
    viewer.user_scn.ngeom += 1


def add_joint(viewer, name, position, radius, show_name):
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[radius, radius, radius],
        pos=np.asarray(position, dtype=np.float64),
        mat=np.eye(3).ravel(),
        rgba=[1.0, 0.48, 0.12, 1.0],
    )
    if show_name:
        geom.label = name
    viewer.user_scn.ngeom += 1


def add_orientation_axes(viewer, name, position, quaternion, length, show_name):
    """Draw the joint local X/Y/Z axes as red/green/blue arrows."""
    if viewer.user_scn.ngeom + 3 > viewer.user_scn.maxgeom:
        return
    rotation = R.from_quat(quaternion, scalar_first=True).as_matrix()
    colors = ([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0])
    for axis, color in enumerate(colors):
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(geom, type=mj.mjtGeom.mjGEOM_ARROW, size=[0.01, 0.01, 0.01], pos=np.asarray(position, dtype=np.float64), mat=rotation.ravel(), rgba=color)
        if show_name:
            geom.label = name + " " + "XYZ"[axis]
        mj.mjv_connector(geom, type=mj.mjtGeom.mjGEOM_ARROW, width=0.006, from_=np.asarray(position, dtype=np.float64), to=np.asarray(position, dtype=np.float64) + length * rotation[:, axis])
        viewer.user_scn.ngeom += 1


def draw_skeleton(viewer, frame, joint_names, edges, bone_radius, joint_radius, show_names, orientation_joints, orientation_length):
    viewer.user_scn.ngeom = 0
    for parent, child in edges:
        if parent in frame and child in frame:
            add_capsule(viewer, frame[parent][0], frame[child][0], bone_radius)
    for name in joint_names:
        if name in frame:
            add_joint(viewer, name, frame[name][0], joint_radius, show_names)
    for name in orientation_joints:
        if name in frame:
            add_orientation_axes(viewer, name, frame[name][0], frame[name][1], orientation_length, show_names)


def set_initial_camera(viewer, frames, joint_names):
    positions = np.asarray(
        [frame[name][0] for frame in frames for name in joint_names if name in frame],
        dtype=np.float64,
    )
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    center = (lower + upper) * 0.5
    extent = np.linalg.norm(upper - lower)
    viewer.cam.lookat[:] = center
    viewer.cam.distance = max(2.0, 1.25 * extent)
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -18


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh_file", required=True, type=pathlib.Path)
    parser.add_argument("--format", choices=("lafan1", "nokov"), default="nokov")
    parser.add_argument("--start", type=int, default=0, help="First frame to play")
    parser.add_argument("--end", type=int, default=None, help="Last frame to play, inclusive")
    parser.add_argument("--fps", type=float, default=None, help="Override the BVH frame rate")
    parser.add_argument("--speed", type=float, default=1.0, help="Initial playback speed")
    parser.add_argument(
        "--play-mode",
        choices=("once", "loop"),
        default="loop",
        help="Play once and pause at the end, or repeat continuously",
    )
    parser.add_argument("--no-loop", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--plot-hand-pose",
        nargs="?",
        type=pathlib.Path,
        const=pathlib.Path("hand_pose_relative_to_elbows.png"),
        default=None,
        metavar="PATH",
        help="Save hand poses relative to elbow frames; optionally set output PNG",
    )
    parser.add_argument(
        "--euler-order",
        choices=("xyz", "xzy", "yxz", "yzx", "zxy", "zyx"),
        default="xyz",
        help="Euler order used in the hand-pose plot",
    )
    parser.add_argument("--show-joint-names", action="store_true")
    parser.add_argument("--hand-joints", nargs="+", default=["LeftHand", "RightHand"], help="Joints where local XYZ orientation axes are drawn")
    parser.add_argument("--no-orientation", action="store_true", help="Hide local orientation axes")
    parser.add_argument("--orientation-length", type=float, default=0.12)
    parser.add_argument("--bone-radius", type=float, default=0.012)
    parser.add_argument("--joint-radius", type=float, default=0.022)
    return parser.parse_args()


def main():
    args = parse_args()
    bvh_path = args.bvh_file.expanduser().resolve()
    if not bvh_path.is_file():
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")
    if args.speed <= 0:
        raise ValueError("--speed must be greater than zero")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be greater than zero")
    if args.bone_radius <= 0 or args.joint_radius <= 0:
        raise ValueError("Skeleton radii must be greater than zero")
    if args.orientation_length <= 0:
        raise ValueError("--orientation-length must be greater than zero")

    frames, joint_names, edges = load_motion(bvh_path, args.format)
    if not frames:
        raise ValueError(f"BVH file contains no motion frames: {bvh_path}")

    start = max(0, args.start)
    end = len(frames) - 1 if args.end is None else min(args.end, len(frames) - 1)
    if start > end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}")

    fps = args.fps or read_fps(bvh_path)
    selected_frames = frames[start : end + 1]
    if args.plot_hand_pose is not None:
        plot_path = plot_hand_poses(
            selected_frames, fps, args.plot_hand_pose, args.euler_order
        )
        print(f"Saved hand-pose plot to {plot_path}")

    play_mode = "once" if args.no_loop else args.play_mode
    state = {"frame": start, "paused": False, "speed": args.speed, "seeked": False}

    def key_callback(keycode):
        if keycode == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
            state["seeked"] = True
        elif keycode == glfw.KEY_LEFT:
            state["frame"] = max(start, state["frame"] - 1)
            state["paused"] = True
            state["seeked"] = True
        elif keycode == glfw.KEY_RIGHT:
            state["frame"] = min(end, state["frame"] + 1)
            state["paused"] = True
            state["seeked"] = True
        elif keycode == glfw.KEY_HOME:
            state["frame"] = start
            state["paused"] = True
            state["seeked"] = True
        elif keycode == glfw.KEY_END:
            state["frame"] = end
            state["paused"] = True
            state["seeked"] = True
        elif keycode == glfw.KEY_UP:
            state["speed"] = min(8.0, state["speed"] * 1.25)
            state["seeked"] = True
        elif keycode == glfw.KEY_DOWN:
            state["speed"] = max(0.125, state["speed"] / 1.25)
            state["seeked"] = True

    model = mj.MjModel.from_xml_string(SCENE_XML)
    data = mj.MjData(model)

    print(f"Loaded {bvh_path.name}: {len(frames)} frames, {len(joint_names)} joints, {fps:.3f} FPS")
    print("Controls: Space play/pause, Left/Right step, Home/End jump, Up/Down speed")

    with mjv.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        set_initial_camera(viewer, selected_frames, joint_names)
        last_drawn_frame = None
        next_frame_time = time.perf_counter() + 1.0 / (fps * state["speed"])

        while viewer.is_running():
            if state["seeked"]:
                next_frame_time = time.perf_counter() + 1.0 / (fps * state["speed"])
                state["seeked"] = False

            if state["frame"] != last_drawn_frame:
                draw_skeleton(
                    viewer,
                    frames[state["frame"]],
                    joint_names,
                    edges,
                    args.bone_radius,
                    args.joint_radius,
                    args.show_joint_names,
                    () if args.no_orientation else args.hand_joints,
                    args.orientation_length,
                )
                last_drawn_frame = state["frame"]

            viewer.sync()
            now = time.perf_counter()
            if not state["paused"] and now >= next_frame_time:
                if state["frame"] >= end:
                    if play_mode == "once":
                        state["paused"] = True
                    else:
                        state["frame"] = start
                else:
                    state["frame"] += 1
                frame_period = 1.0 / (fps * state["speed"])
                next_frame_time = max(next_frame_time + frame_period, now)
            else:
                time.sleep(0.001)


if __name__ == "__main__":
    main()
