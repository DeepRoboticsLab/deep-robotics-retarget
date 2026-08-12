"""General Motion Retargeting (GMR) for Humanoid Robots.

This package provides tools to retarget human motion data (from SMPLX, BVH, etc.)
to humanoid robot joint configurations using inverse kinematics (IK).

Core components:
    - GeneralMotionRetargeting: Main IK-based motion retargeting class.
    - RobotMotionViewer: MuJoCo-based robot motion visualizer.
    - KinematicsModel: Analytical forward kinematics model.
    - PlaybackController: GUI playback controller with pause/seek.
"""

from rich import print
from .params import IK_CONFIG_ROOT, ASSET_ROOT, ROBOT_XML_DICT, IK_CONFIG_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from .motion_retarget import GeneralMotionRetargeting
from .robot_motion_viewer import RobotMotionViewer, draw_frame
from .data_loader import load_robot_motion
from .playback_controller import PlaybackController
from .kinematics_model import KinematicsModel

from .neck_retarget import human_head_to_robot_neck

try:
    from .xrobot_utils import XRobotStreamer, XRobotRecorder
except ImportError:
    print("XRobotStreamer is not installed. Please install xrobotoolkit_sdk to use this feature.")
    XRobotStreamer = None
    XRobotRecorder = None
