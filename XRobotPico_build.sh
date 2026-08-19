#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="${SCRIPT_DIR}/third_party"

echo "=========================================="
echo "  XRobotPico Build Script"
echo "=========================================="

# 1. 安装 PC Service (deb)
echo ""
echo "[Step 1] 安装 XRoboToolkit PC Service (deb)..."
DEB_URL="https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb"
DEB_FILE="/tmp/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb"

if [ ! -f "${DEB_FILE}" ]; then
    wget -O "${DEB_FILE}" "${DEB_URL}"
fi
sudo dpkg -i "${DEB_FILE}"

# 2. 安装编译依赖
echo ""
echo "[Step 2] 安装编译依赖..."
sudo apt install -y cmake build-essential python3-dev pybind11-dev

# 3. 克隆 XRoboToolkit-PC-Service-Pybind 到 third_party
echo ""
echo "[Step 3] 克隆 XRoboToolkit-PC-Service-Pybind..."
PYBIND_DIR="${THIRD_PARTY_DIR}/XRoboToolkit-PC-Service-Pybind"
if [ ! -d "${PYBIND_DIR}" ]; then
    git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git "${PYBIND_DIR}"
else
    echo "  目录已存在，跳过克隆"
fi

# 4. 克隆 XRoboToolkit-PC-Service 作为 SDK 源码
echo ""
echo "[Step 4] 克隆 XRoboToolkit-PC-Service..."
SDK_DIR="${PYBIND_DIR}/tmp/XRoboToolkit-PC-Service"
if [ ! -d "${SDK_DIR}" ]; then
    mkdir -p "${PYBIND_DIR}/tmp"
    git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git "${SDK_DIR}"
else
    echo "  目录已存在，跳过克隆"
fi

# 5. 编译 SDK
echo ""
echo "[Step 5] 编译 PXREARobotSDK..."
cd "${SDK_DIR}/RoboticsService/PXREARobotSDK"
bash build.sh

# 6. 拷贝头文件和库文件到 Pybind 项目
echo ""
echo "[Step 6] 拷贝头文件和库文件..."
cd "${PYBIND_DIR}"
mkdir -p lib include
cp "${SDK_DIR}/RoboticsService/PXREARobotSDK/PXREARobotSDK.h" include/
cp -r "${SDK_DIR}/RoboticsService/PXREARobotSDK/nlohmann" include/nlohmann/
cp "${SDK_DIR}/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so" lib/

# 7. 安装 Python 包
echo ""
echo "[Step 7] 安装 Python SDK..."
python setup.py install

echo ""
echo "=========================================="
echo "  构建完成！"
echo "=========================================="
