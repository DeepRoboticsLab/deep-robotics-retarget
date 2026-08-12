# deep-robotics-retarget

[中文](README_cn.md)

## Overview

deep-robotics-retarget is a Python toolkit that uses inverse kinematics (IK) to retarget human motion data to humanoid robot joint configurations. It supports multiple human motion data sources and provides real-time visualization based on MuJoCo. Currently, the project supports the DR02 Pro robot model.

This project is modified from the [GMR](https://github.com/YanjieZe/GMR/tree/master) project.

## Features

- **Multiple Input Sources**: Supports SMPLX, BVH (Lafan1 / Nokov), and other motion capture formats
- **Two-Stage IK Solving**: Orientation-only tracking followed by combined position + orientation tracking
- **Collision Avoidance**: Optional self-collision and floor collision avoidance
- **Body Scaling**: Automatic mapping of human body proportions to the robot
- **Real-Time Visualization**: MuJoCo-based viewer with human motion overlay
- **Batch Processing**: Dataset-scale motion retargeting

## Supported Robots

| Robot | Status |
|-------|--------|
| DR02 Pro | ✅ Supported |

The framework is designed to be extensible. To add a new robot:

1. Add the robot's MuJoCo XML to `assets/`
2. Create an IK configuration JSON in `general_motion_retargeting/ik_configs/`
3. Register the robot in `general_motion_retargeting/params.py`

## Installation

### Requirements

- Python >= 3.10
- Linux (Ubuntu 22.04 / 24.04 recommended)

### Install Steps

> [!NOTE]
> This project has been tested on Ubuntu 24.04.

First, create a Conda environment:

```bash
conda create -n retarget python=3.10 -y
conda activate retarget
```

Then clone the repository and install:

```bash
# Clone the repository
git clone https://github.com/DeepRoboticsLab/deep-robotics-retarget.git
cd deep-robotics-retarget

# Update libstdc++
conda install -c conda-forge libstdcxx-ng -y

# Install the package
pip install -e .
```

Install PICO SDK:

- Download the [PICO package](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb) and install:
  `sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb`
  , or build from [source](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)

- Build the PICO PC Service SDK:
```
conda activate retargeting

git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git
cd XRoboToolkit-PC-Service-Pybind

mkdir -p tmp
cd tmp
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK
bash build.sh
cd ../../../..

mkdir -p lib
mkdir -p include
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/
cp -r tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/
# rm -rf tmp

# Build the project
conda install -c conda-forge pybind11
pip uninstall -y xrobotoolkit_sdk
python setup.py install
```

### Key Dependencies

- `mujoco` — Physics simulation and visualization
- `mink` — Inverse kinematics solver
- `smplx` — SMPL-X body model (installed from [GitHub](https://github.com/vchoutas/smplx))
- `qpsolvers[proxqp]` — QP solver for IK
- `redis[hiredis]` — Real-time streaming

## Data Preparation

### SMPLX Model

Download the body model from the [SMPL-X](https://smpl-x.is.tue.mpg.de/) official website, extract and place it in the `assets/body_models/smplx/` folder:

```
assets/body_models/smplx/
├── SMPLX_NEUTRAL.npz
├── SMPLX_FEMALE.npz
└── SMPLX_MALE.npz
```

> [!NOTE]
> This project uses `npz` model files by default. If you downloaded `pkl` format files from the SMPL-X website, you need to change the `ext` parameter in the `create()` function within the `smplx` library's `smplx/body_model.py` from the default value `npz` to `pkl`, or convert the `pkl` files to `npz` format.

### AMASS Data

Download raw data from [AMASS](https://amass.is.tue.mpg.de/) to any location. It is recommended to place it under `source_data/AMASS/`. For quick start, two sample files are provided in `source_data/AMASS_demo/`.

### LAFAN1 Data

Download raw BVH files from the [LAFAN repository](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) ([lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip)), extract and place them under `source_data/lafan1/`. For quick start, two sample files are provided in `source_data/lafan1_demo/`.

### Nokov Data

Capture the required data using Nokov motion capture equipment. For quick start, three sample files are provided in `source_data/nokov_demo/`.

## Quick Start

### SMPLX Motion Retargeting

#### Single Motion Retargeting

```bash
python scripts/smplx_to_robot.py \
    --smplx_file <path_to_smplx_data.npz> \
    --save_path <path_to_save_robot_data.pkl> \
    --rate_limit
```

By default, running this program will display the robot motion retargeting in real-time in a MuJoCo window.

Optional parameters:

- `--rate_limit`: Limit the retargeted robot motion data frequency to match human motion speed
- `--record_video`: Record a video of the robot motion retargeting, saved to the `videos/` folder
- `--robot`: Select the robot model for retargeting, default is `DR02_pro`

#### Batch Motion Retargeting

```bash
python scripts/smplx_to_robot_dataset.py \
    --src_folder <path_to_dir_of_smplx_data> \
    --tgt_folder <path_to_dir_to_save_robot_data>
```

By default, batch motion retargeting does not visualize the motion.

### BVH Motion Retargeting

#### Single Motion Retargeting

```bash
# Lafan1 format
python scripts/bvh_to_robot.py \
    --bvh_file <path_to_bvh_data> \
    --save_path <path_to_save_robot_data.pkl> \
    --format lafan1

# Nokov format
python scripts/bvh_to_robot.py \
    --bvh_file <path_to_bvh_data> \
    --save_path <path_to_save_robot_data.pkl> \
    --format nokov
```

#### Batch Motion Retargeting

```bash
# Lafan1 format
python scripts/bvh_to_robot_dataset.py \
    --src_folder <path_to_dir_of_bvh_data> \
    --tgt_folder <path_to_dir_to_save_robot_data> \
    --format lafan1

# Nokov format
python scripts/bvh_to_robot_dataset.py \
    --src_folder <path_to_dir_of_bvh_data> \
    --tgt_folder <path_to_dir_to_save_robot_data> \
    --format nokov
```

### Utility Tools

#### Data Transfer

```bash
# Single file
python scripts/pkl_to_npz.py \
    --input <path_to_pkl_data> \
    --output <path_to_npz_data> \

# Batch convert
python scripts/pkl_to_npz.py \
    --input_dir <path_to_dir_of_pkl_data> \
    --output_dir <path_to_dir_of_npz_data> \

```

#### Visualize Retargeted Motion

```bash
# Single motion
python scripts/vis_robot_motion.py \
    --robot_motion_path path/to/retargeted.pkl

# Dataset browsing (use '[' and ']' to switch motions)
python scripts/vis_robot_motion_dataset.py \
    --robot_motion_folder /path/to/retargeted_motions
```

#### Plot Retargeting Results

```bash
python scripts/plot_retarget_motion.py \
    --bvh_file <path_to_bvh_data> \
    --format nokov
```

Results are saved in the `plots/` folder, including plots of root position, root orientation, and joint angles over time.

## Python API

```python
from general_motion_retargeting import GeneralMotionRetargeting, RobotMotionViewer

# Initialize retargeting
retarget = GeneralMotionRetargeting(
    src_human="bvh_nokov",       # Source data format
    tgt_robot="DR02_pro",        # Target robot
    actual_human_height=1.75,    # Optional: actual human height
)

# Retarget a single frame
# human_data: dict, format is {body_name: (position, quaternion)}
qpos = retarget.retarget(human_data)

# Extract joint information
root_pos = qpos[:3]
root_rot = qpos[3:7]             # quaternion (wxyz)
joint_angles = qpos[7:]
```

## Project Structure

```
deep-robotics-retarget/
├── general_motion_retargeting/     # Core package
│   ├── motion_retarget.py          # Main IK retargeting class
│   ├── robot_motion_viewer.py      # MuJoCo visualization
│   ├── kinematics_model.py         # Analytical forward kinematics
│   ├── playback_controller.py      # GUI playback control
│   ├── neck_retarget.py            # Head/neck retargeting
│   ├── data_loader.py              # Motion data I/O
│   ├── params.py                   # Robot/config registry
│   ├── rot_utils.py                # Rotation utilities
│   ├── torch_utils.py              # PyTorch rotation ops
│   ├── xrobot_utils.py             # XRobot SDK integration
│   ├── ik_configs/                 # IK configuration JSONs
│   └── utils/                      # Data format loaders
│       ├── smpl.py                 # SMPL/SMPLX loading
│       └── lafan1.py               # BVH (Lafan1/Nokov) loading
├── scripts/                        # Example scripts
│   ├── smplx_to_robot.py           # SMPLX retargeting
│   ├── bvh_to_robot.py             # BVH retargeting
│   ├── bvh_to_robot_dataset.py     # Batch BVH processing
│   ├── vis_robot_motion.py         # Motion visualization
│   ├── plot_retarget_motion.py     # Retargeting result plotting
│   └── ...                         # Other utilities
├── assets/                         # Robot models and body models
│   ├── DR02/                       # DR02 Pro robot model
│   └── body_models/smplx/          # SMPL-X body models
└── source_data/                    # Sample motion data
```

## License

This project is licensed under the MIT License.
