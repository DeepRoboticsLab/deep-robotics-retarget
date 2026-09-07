#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="${SCRIPT_DIR}/third_party"
PYTHON_BIN="$(command -v python)" || {
    echo "Activate the deep-robotics-humanoid Conda environment first." >&2
    exit 1
}
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
echo "Using Python: ${PYTHON_BIN}"

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
sudo apt install -y cmake build-essential
# Ubuntu 22.04's pybind11 2.9.1 is incompatible with Python 3.11.
# Use headers and CMake configuration from the active Python environment.
"${PYTHON_BIN}" -m pip install setuptools==80.9.0 wheel pybind11==2.13.6
PYBIND11_CMAKE_DIR="$("${PYTHON_BIN}" -m pybind11 --cmakedir)"
export CMAKE_PREFIX_PATH="${PYBIND11_CMAKE_DIR}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

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
mkdir -p include/nlohmann
cp -r "${SDK_DIR}/RoboticsService/PXREARobotSDK/nlohmann/." include/nlohmann/
cp "${SDK_DIR}/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so" lib/

# 7. 安装 Python 包
echo ""
echo "[Step 7] 安装 Python SDK..."
# Clear generated artifacts so CMake cannot reuse Python libraries or headers
# from another Conda environment (both environments can have Python 3.11).
rm -rf "${PYBIND_DIR}/build"
"${PYTHON_BIN}" -m pip install --no-build-isolation --no-deps .
"${PYTHON_BIN}" -c 'import xrobotoolkit_sdk; print("SDK import OK:", xrobotoolkit_sdk.__file__)'

echo ""
echo "=========================================="
echo "  构建完成！"
echo "=========================================="
