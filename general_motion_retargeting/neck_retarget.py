"""Neck retargeting utilities for robot head control.

Extracts neck yaw and pitch angles from human head/spine orientation data
for robots with 2-DoF head joints.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def get_human_neck_orientation(head_pose=None):
    """Compute neck orientation from headset pose.

    Args:
        head_pose: Head pose array containing quaternion [x, y, z, w].

    Returns:
        tuple: (roll, pitch, yaw) in degrees.
    """
    quat_xyzw = np.array([head_pose[3], head_pose[4], head_pose[5], head_pose[6]])
    rotation = R.from_quat(quat_xyzw)
    roll, pitch, yaw = rotation.as_euler('xyz', degrees=True)
    return roll, pitch, yaw


def human_head_to_robot_neck(smplx_data=None):
    """Extract neck angle from smplx_data for the robot head.

    The robot head has 2 DoF:
        - dof 0: yaw
        - dof 1: pitch

    Args:
        smplx_data: Dict mapping body names to (position, quaternion) tuples.

    Returns:
        tuple: (neck_yaw, neck_pitch) in radians.
    """
    if smplx_data is None:
        return 0.0, 0.0

    spine_rotation = smplx_data['Spine3'][1]  # wxyz
    head_rotation = smplx_data['Head'][1]      # wxyz

    spine_rotation = R.from_quat(spine_rotation, scalar_first=True)
    head_rotation = R.from_quat(head_rotation, scalar_first=True)

    # Relative rotation: head relative to spine
    relative_rotation = spine_rotation.inv() * head_rotation
    roll, pitch, yaw = relative_rotation.as_euler('xyz', degrees=True)

    neck_yaw = -pitch
    neck_pitch = roll

    # Degree to radian
    neck_yaw = np.deg2rad(neck_yaw)
    neck_pitch = np.deg2rad(neck_pitch)

    return neck_yaw, neck_pitch
