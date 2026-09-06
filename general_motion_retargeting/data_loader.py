"""Data loading utilities for retargeted robot motion.

This module provides functions to load robot motion data from pickle files.
The data format includes root position/rotation, joint positions, and optional
local body positions for visualization and analysis.
"""

import importlib
import pickle
import sys

import numpy as np

if not hasattr(np, "_core"):
    # Pickles written under numpy 2.x reference the internal `numpy._core`
    # module (renamed from `numpy.core` in numpy 2). Register aliases so these
    # files can still be loaded under numpy 1.x.
    sys.modules.setdefault("numpy._core", np.core)
    for _name in ("multiarray", "_multiarray_umath", "umath", "numeric", "numerictypes"):
        try:
            sys.modules.setdefault(
                f"numpy._core.{_name}", importlib.import_module(f"numpy.core.{_name}")
            )
        except ImportError:
            pass

def load_robot_motion(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
        motion_fps = motion_data["fps"]
        motion_root_pos = motion_data["root_pos"]
        motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]] # from xyzw to wxyz
        motion_dof_pos = motion_data["dof_pos"]
        motion_local_body_pos = motion_data["local_body_pos"]
        motion_link_body_list = motion_data["link_body_list"]
    return motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list


