# deep-robotics-retarget

[English](README.md)

## 概述

deep-robotics-retarget 是一个 Python 工具包，使用逆运动学（IK）将人体运动数据重定向到人形机器人关节配置。支持多种人体运动数据来源，并提供基于 MuJoCo 的实时可视化，目前该项目支持 DR02 Pro 型号机器人。

本项目基于[GMR](https://github.com/YanjieZe/GMR/tree/master)项目修改。

## 功能特性

- **多数据源输入**：支持 SMPLX、BVH（Lafan1 / Nokov）等多种动捕格式
- **两阶段 IK 求解**：先进行纯姿态跟踪，再进行位置 + 姿态联合跟踪
- **碰撞避免**：可选的自碰撞与地面碰撞避免
- **身体比例缩放**：自动将人体比例映射到机器人
- **实时可视化**：基于 MuJoCo 的查看器，支持人体运动叠加显示
- **批量处理**：支持数据集级别的批量运动重定向

## 支持的机器人

| 机器人 | 状态 |
|--------|------|
| DR02 Pro | ✅ 已支持 |

本框架设计上支持扩展。添加新机器人的步骤：

1. 将机器人的 MuJoCo XML 放入 `assets/`
2. 在 `general_motion_retargeting/ik_configs/` 中创建 IK 配置 JSON
3. 在 `general_motion_retargeting/params.py` 中注册机器人

## 安装

### 系统要求

- Python >= 3.10
- Linux（推荐 Ubuntu 22.04 / 24.04）

### 安装步骤

> [!NOTE]
> 本项目已在 Ubuntu 24.04 操作系统上完成测试。

首先创建 Conda 环境：

```bash
conda create -n retarget python=3.10 -y
conda activate retarget
```

然后克隆仓库并安装：

```bash
# 克隆仓库
git clone [to do]
cd deep-robotics-retarget

# 更新 libstdc++
conda install -c conda-forge libstdcxx-ng -y

# 安装包
pip install -e .
```

安装PICO SDK：

- 下载[PICO安装包](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb)并安装：
`sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
`，或者通过[源码](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)编译

- 构建 PICO PC Service SDK
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
### 核心依赖

- `mujoco` — 物理仿真与可视化
- `mink` — 逆运动学求解器
- `smplx` — SMPL-X 人体模型（从 [GitHub](https://github.com/vchoutas/smplx) 安装）
- `qpsolvers[proxqp]` — IK 使用的 QP 求解器
- `redis[hiredis]` — 实时流式传输

## 数据准备

### SMPLX 模型

从 [SMPL-X](https://smpl-x.is.tue.mpg.de/) 官网下载人体模型，解压后存放到 `assets/body_models/smplx/` 文件夹下：

```
assets/body_models/smplx/
├── SMPLX_NEUTRAL.npz
├── SMPLX_FEMALE.npz
└── SMPLX_MALE.npz
```

> [!NOTE]
> 本项目默认使用 `npz` 格式的模型文件。如果你从 SMPL-X 官网下载的是 `pkl` 格式，需要将 `smplx` 库的 `smplx/body_model.py` 中 `create()` 函数的 `ext` 参数从默认值 `npz` 修改为 `pkl`，或者将 `pkl` 文件转换为 `npz` 格式。

### AMASS 数据

从 [AMASS](https://amass.is.tue.mpg.de/) 下载原始数据到任意位置，推荐放在 `source_data/AMASS/` 下。为方便快速上手，`source_data/AMASS_demo/` 下已提供两条示例数据。

### LAFAN1 数据

从 [LAFAN 仓库](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) 下载原始 BVH 文件（[lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip)），解压后推荐放在 `source_data/lafan1/` 路径下。为方便快速上手，`source_data/lafan1_demo/` 下已提供两条示例数据。

### Nokov 数据

通过 Nokov 动作捕捉设备采集所需数据。为方便快速上手，`source_data/nokov_demo/` 下已提供三条示例数据。

## 快速开始

### SMPLX 运动数据重定向

#### 单个动作重定向

```bash
python scripts/smplx_to_robot.py \
    --smplx_file <path_to_smplx_data.npz> \
    --save_path <path_to_save_robot_data.pkl> \
    --rate_limit
```

默认情况下，运行该程序会在 MuJoCo 窗口中实时展示机器人运动重定向效果。

可选参数：

- `--rate_limit`：限制重定向后机器人运动数据频率，使其动作速度与人类保持一致
- `--record_video`：录制机器人运动重定向效果视频，保存至 `videos/` 文件夹
- `--robot`：选择重定向的机器人型号，默认为 `DR02_pro`

#### 批量运动重定向

```bash
python scripts/smplx_to_robot_dataset.py \
    --src_folder <path_to_dir_of_smplx_data> \
    --tgt_folder <path_to_dir_to_save_robot_data>
```

默认情况下，批量运动重定向不会可视化运动效果。

### BVH 运动数据重定向

#### 单个动作重定向

```bash
# Lafan1 格式
python scripts/bvh_to_robot.py \
    --bvh_file <path_to_bvh_data> \
    --save_path <path_to_save_robot_data.pkl> \
    --format lafan1 \
    --rate_limit


# Nokov 格式
python scripts/bvh_to_robot.py \
    --bvh_file <path_to_bvh_data> \
    --save_path <path_to_save_robot_data.pkl> \
    --format nokov \
    --rate_limit
```

#### 批量运动重定向

```bash
# Lafan1 格式
python scripts/bvh_to_robot_dataset.py \
    --src_folder <path_to_dir_of_bvh_data> \
    --tgt_folder <path_to_dir_to_save_robot_data> \
    --format lafan1

# Nokov 格式
python scripts/bvh_to_robot_dataset.py \
    --src_folder <path_to_dir_of_bvh_data> \
    --tgt_folder <path_to_dir_to_save_robot_data> \
    --format nokov
```

### 辅助工具

#### 数据转换

```bash
# 单个文件
python scripts/pkl_to_npz.py \
    --input <path_to_pkl_data> \
    --output <path_to_npz_data> \

# 批量处理
python scripts/pkl_to_npz.py \
    --input_dir <path_to_dir_of_pkl_data> \
    --output_dir <path_to_dir_of_npz_data> \

```

#### 可视化重定向运动

```bash
# 单个运动
python scripts/vis_robot_motion.py \
    --robot_motion_path <path_to_pkl_data>

# 数据集浏览（使用 '[' 和 ']' 切换运动）
python scripts/vis_robot_motion_dataset.py \
    --robot_motion_folder <path_to_dir_of_pkl_data>
```

#### 重定向结果绘图

```bash
python scripts/plot_retarget_motion.py \
    --bvh_file <path_to_bvh_data> \
    --format nokov
```

结果输出在 `plots/` 文件夹下，包括根节点位置、根节点姿态以及各关节角度随时间变化的曲线图。

## Python API

```python
from general_motion_retargeting import GeneralMotionRetargeting, RobotMotionViewer

# 初始化重定向
retarget = GeneralMotionRetargeting(
    src_human="bvh_nokov",       # 源数据格式
    tgt_robot="DR02_pro",        # 目标机器人
    actual_human_height=1.75,    # 可选：实际人体身高
)

# 重定向单帧
# human_data: dict, 格式为 {body_name: (position, quaternion)}
qpos = retarget.retarget(human_data)

# 提取关节信息
root_pos = qpos[:3]
root_rot = qpos[3:7]             # 四元数 (wxyz)
joint_angles = qpos[7:]
```

## 项目结构

```
deep-robotics-retarget/
├── general_motion_retargeting/     # 核心包
│   ├── motion_retarget.py          # 主 IK 重定向类
│   ├── robot_motion_viewer.py      # MuJoCo 可视化
│   ├── kinematics_model.py         # 解析正向运动学
│   ├── playback_controller.py      # GUI 播放控制
│   ├── neck_retarget.py            # 头部 / 颈部重定向
│   ├── data_loader.py              # 运动数据 I/O
│   ├── params.py                   # 机器人 / 配置注册表
│   ├── rot_utils.py                # 旋转工具
│   ├── torch_utils.py              # PyTorch 旋转操作
│   ├── xrobot_utils.py             # XRobot SDK 集成
│   ├── ik_configs/                 # IK 配置 JSON
│   └── utils/                      # 数据格式加载器
│       ├── smpl.py                 # SMPL / SMPLX 加载
│       └── lafan1.py               # BVH（Lafan1 / Nokov）加载
├── scripts/                        # 示例脚本
│   ├── smplx_to_robot.py           # SMPLX 重定向
│   ├── bvh_to_robot.py             # BVH 重定向
│   ├── bvh_to_robot_dataset.py     # 批量 BVH 处理
│   ├── vis_robot_motion.py         # 运动可视化
│   ├── plot_retarget_motion.py     # 重定向结果绘图
│   └── ...                         # 其他工具
├── assets/                         # 机器人模型与人体模型
│   ├── DR02/                       # DR02 Pro 机器人模型
│   └── body_models/smplx/          # SMPL-X 人体模型
└── source_data/                    # 示例运动数据
```

## 许可证

本项目基于 MIT 许可证开源。
