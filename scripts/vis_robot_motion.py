"""Visualize a single retargeted robot motion with playback control.

Loads a pickle motion file and displays it in MuJoCo viewer with
pause/resume using the Space key in the viewer.

Usage:
    python scripts/vis_robot_motion.py --robot_motion_path output/bow.pkl
"""

from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os

paused = False


def keyboard_callback(keycode):
    global paused
    if keycode == ord(' '):
        paused = not paused


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="DR02_pro")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path,
                            keyboard_callback=keyboard_callback)
    print("Viewer controls: Space = pause/resume.")

    frame_idx = 0
    try:
        while env.viewer.is_running():
            env.step(
                motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                rate_limit=True,
            )

            if not paused:
                frame_idx = (frame_idx + 1) % len(motion_root_pos)

    finally:
        env.close()
