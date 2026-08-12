"""Visualize a dataset of retargeted robot motions with navigation.

Loads a folder of pickle motion files and allows browsing between them
using '[' and ']' keys. Supports pause/resume and frame seeking.

Usage:
    python scripts/vis_robot_motion_dataset.py --robot_motion_folder /path/to/motions
"""

from general_motion_retargeting import PlaybackController, RobotMotionViewer, load_robot_motion
import argparse
import os
from tqdm import tqdm

paused = False
motion_num = 0
motion_id = 0
current_motion_id = -1
controller = None

def keyboard_callback(keycode):
    global paused, motion_id, motion_num, controller
    if chr(keycode) == ' ':
        paused = not paused
        if controller is not None and controller.is_available():
            state = controller.get_state()
            controller.set_state(state.frame_idx, paused=paused)
    if chr(keycode) == '[':
        motion_id = (motion_id - 1) % motion_num
    if chr(keycode) == ']':
        motion_id = (motion_id + 1) % motion_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="DR02_pro")
                        
    parser.add_argument("--robot_motion_folder", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_folder = args.robot_motion_folder
    
    if not os.path.exists(robot_motion_folder):
        raise FileNotFoundError(f"Motion data dir {robot_motion_folder} does not exist.")
    
    motion_files = [f for f in os.listdir(robot_motion_folder) if f.endswith('.pkl')]
    motion_files = sorted(motion_files)
    motion_num = len(motion_files)
    print(f"Found {motion_num} motion files in {robot_motion_folder}, loading...")
    motion_dataset = []
    for motion_file in tqdm(motion_files):
        motion_path = os.path.join(robot_motion_folder, motion_file)
        motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(motion_path)
        motion_dataset.append({
            "motion_file": motion_file,
            "motion_data": motion_data,
            "motion_fps": motion_fps,
            "motion_root_pos": motion_root_pos,
            "motion_root_rot": motion_root_rot,
            "motion_dof_pos": motion_dof_pos,
            "motion_local_body_pos": motion_local_body_pos,
            "motion_link_body_list": motion_link_body_list,
        })
    print("Loading done.")
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path, 
                            keyboard_callback=keyboard_callback)
    
    frame_idx = 0
    try:
        while True:
            # get current motion
            if current_motion_id != motion_id:
                current_motion_id = motion_id
                frame_idx = 0
                motion_data = motion_dataset[motion_id]
                motion_file = motion_data["motion_file"]
                motion_fps = motion_data["motion_fps"]
                motion_root_pos = motion_data["motion_root_pos"]
                motion_root_rot = motion_data["motion_root_rot"]
                motion_dof_pos = motion_data["motion_dof_pos"]
                print(f"Switched to motion {motion_id}: {motion_file}, fps: {motion_fps}, num_frames: {len(motion_root_pos)}")
                if controller is not None:
                    controller.close()
                controller = PlaybackController(
                    num_frames=len(motion_root_pos),
                    title=f"Dataset Playback: {motion_file}",
                )
                if controller.is_available():
                    controller.set_state(0, paused=paused)
            
            if controller is not None and controller.is_available():
                controller.apply_pending_commands()
                if controller.is_closed():
                    break
                state = controller.get_state()
                frame_idx = state.frame_idx
                paused = state.paused

            env.step(
                motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                rate_limit=True,
            )

            if not paused:
                frame_idx += 1
                if frame_idx >= len(motion_root_pos):
                    frame_idx = 0

            if controller is not None and controller.is_available():
                controller.set_state(frame_idx, paused=paused)
    finally:
        if controller is not None:
            controller.close()
        env.close()
