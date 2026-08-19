"""Convert retargeted PKL motion files to NPZ format.

Performs temporal resampling to a target FPS with linear interpolation
for positions and SLERP for rotations.

Usage:
    # Batch mode (folder)
    python scripts/pkl_to_npz.py --input_dir output/ --output_dir output/npz_file/

    # Single-file mode
    python scripts/pkl_to_npz.py --input output/bow.pkl --output output/npz_file/bow.npz
"""

import argparse
import glob
import os
import pickle

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


TARGET_FPS = 50


def convert_single(pkl_path: str, npz_path: str) -> None:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    fps = int(data["fps"])
    root_pos = np.asarray(data["root_pos"])          # (N, 3)
    root_rot = np.asarray(data["root_rot"])          # (N, 4) xyzw
    dof_pos = np.asarray(data["dof_pos"])            # (N, 21)

    N = root_pos.shape[0]
    duration = (N - 1) / fps
    M = int(np.floor(duration * TARGET_FPS)) + 1

    t_old = np.linspace(0.0, (N - 1) / fps, N)
    t_new = np.linspace(0.0, (M - 1) / TARGET_FPS, M)

    # 1. root_pos: linear interpolation
    f_pos = interp1d(t_old, root_pos, axis=0, kind="linear")
    root_pos_new = f_pos(t_new)

    # 2. root_rot: xyzw -> Slerp -> wxyz
    # scipy Rotation.from_quat expects xyzw
    rots = Rotation.from_quat(root_rot)
    slerp = Slerp(t_old, rots)
    rots_new = slerp(t_new)
    root_rot_new = rots_new.as_quat()[:, [3, 0, 1, 2]]  # convert to wxyz

    # 3. dof_pos: linear interpolation
    f_dof = interp1d(t_old, dof_pos, axis=0, kind="linear")
    dof_pos_new = f_dof(t_new)

    np.savez(
        npz_path,
        root_pos=root_pos_new.astype(np.float32),
        root_rot=root_rot_new.astype(np.float32),
        dof_pos=dof_pos_new.astype(np.float32),
        fps=TARGET_FPS,
    )
    print(f"Saved: {npz_path}  ({N} frames @ {fps}Hz -> {M} frames @ {TARGET_FPS}Hz)")


def main():
    parser = argparse.ArgumentParser(description="Convert retargeted PKL motion files to NPZ format.")
    parser.add_argument("--input_dir", help="Directory containing .pkl files (batch mode)")
    parser.add_argument("--output_dir", help="Directory to save .npz files (batch mode)")
    parser.add_argument("--input", help="Path to a single .pkl file (single-file mode)")
    parser.add_argument("--output", help="Path to save a single .npz file (single-file mode)")
    parser.add_argument("--override", action="store_true", default=False, help="覆盖已存在的输出文件，不询问")
    args = parser.parse_args()

    batch_mode = args.input_dir is not None or args.output_dir is not None
    single_mode = args.input is not None or args.output is not None

    if batch_mode and single_mode:
        parser.error("Cannot mix batch mode (--input_dir/--output_dir) and single-file mode (--input/--output).")
    if not batch_mode and not single_mode:
        parser.error("Must specify either --input_dir/--output_dir (batch) or --input/--output (single file).")

    if single_mode:
        if not args.input or not args.output:
            parser.error("Single-file mode requires both --input and --output.")
        if not os.path.isfile(args.input):
            parser.error(f"Input file not found: {args.input}")
        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(args.output) and not args.override:
            response = input(f"  文件已存在: {args.output}\n  是否覆盖? [y/N]: ").strip().lower()
            if response not in ("y", "yes"):
                print(f"  Skip: {args.output}")
                return
        convert_single(args.input, args.output)
        return

    # Batch mode
    if not args.input_dir or not args.output_dir:
        parser.error("Batch mode requires both --input_dir and --output_dir.")
    if not os.path.isdir(args.input_dir):
        parser.error(f"Input directory not found: {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    pkl_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pkl")))

    if not pkl_files:
        print(f"No .pkl files found in {args.input_dir}")
        return

    for pkl_path in pkl_files:
        basename = os.path.splitext(os.path.basename(pkl_path))[0]
        npz_path = os.path.join(args.output_dir, basename + ".npz")
        if os.path.exists(npz_path) and not args.override:
            response = input(f"  文件已存在: {npz_path}\n  是否覆盖? [y/N]: ").strip().lower()
            if response not in ("y", "yes"):
                print(f"  Skip: {npz_path}")
                continue
        convert_single(pkl_path, npz_path)


if __name__ == "__main__":
    main()
