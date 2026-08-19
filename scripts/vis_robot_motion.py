"""Visualize a single retargeted robot motion with playback control.

Loads a pickle motion file and displays it in MuJoCo viewer with
pause/resume and frame seeking via PlaybackController GUI.

Usage:
    python scripts/vis_robot_motion.py --robot_motion_path output/bow.pkl
"""

from general_motion_retargeting import PlaybackController, RobotMotionViewer, load_robot_motion
import argparse
import os

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
                            record_video=args.record_video, video_path=args.video_path)

    controller = PlaybackController(
        num_frames=len(motion_root_pos),
        title=f"Playback Control: {os.path.basename(robot_motion_path)}",
    )
    if not controller.is_available():
        print("Playback controller unavailable, fallback to continuous playback.")

    frame_idx = 0
    try:
        while True:
            if controller.is_available():
                controller.apply_pending_commands()
                if controller.is_closed():
                    break
                state = controller.get_state()
                frame_idx = state.frame_idx
                paused = state.paused
            else:
                paused = False

            env.step(
                motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                rate_limit=True,
            )

            if not paused:
                frame_idx = (frame_idx + 1) % len(motion_root_pos)

            if controller.is_available():
                controller.set_state(frame_idx, paused=paused)
    finally:
        controller.close()
        env.close()
