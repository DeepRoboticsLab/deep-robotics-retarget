"""Convert OMOMO dataset to SMPLX format.

This script converts OMOMO motion data to individual SMPLX-compatible pickle files.
It is provided as a reference for dataset conversion workflows.

Usage:
    python scripts/convert_omomo_to_smplx.py \\
        --motion_paths /path/to/omomo_train.p /path/to/omomo_test.p \\
        --target_dir /path/to/output
"""
import os
import joblib
import numpy as np
import pickle
import argparse
import importlib
import sys

if not hasattr(np, "_core"):
    # OMOMO pickles written under numpy 2.x reference the internal
    # `numpy._core` module (renamed from `numpy.core` in numpy 2). Register
    # aliases so they can still be loaded under numpy 1.x.
    sys.modules.setdefault("numpy._core", np.core)
    for _name in ("multiarray", "_multiarray_umath", "umath", "numeric", "numerictypes"):
        try:
            sys.modules.setdefault(
                f"numpy._core.{_name}", importlib.import_module(f"numpy.core.{_name}")
            )
        except ImportError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OMOMO dataset to SMPLX format.")
    parser.add_argument(
        "--motion_paths",
        nargs="+",
        required=True,
        help="Paths to OMOMO motion data pickle files.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Target directory to save converted SMPLX files.",
    )
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    for motion_path in args.motion_paths:
        all_motion_data = joblib.load(motion_path)
        for data_name in all_motion_data.keys():
            smpl_data = all_motion_data[data_name]
            seq_name = smpl_data['seq_name']
            num_frames = smpl_data["pose_body"].shape[0]
            mocap_frame_rate = 30
            poses = np.concatenate([smpl_data["pose_body"], 
                                    np.zeros((num_frames, 102))],
                                    axis=1)
            smpl_data["poses"] = poses
            smpl_data["mocap_frame_rate"] = np.array(mocap_frame_rate)
            with open(f"{args.target_dir}/{seq_name}.pkl", "wb") as f:
                pickle.dump(smpl_data, f)
            print(f"saved {seq_name}")