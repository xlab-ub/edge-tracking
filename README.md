# PTZ Camera Human Tracking and Fall Detection System

A real-time computer vision system that uses a PTZ (Pan-Tilt-Zoom) camera to track humans and detect falls, with automatic WhatsApp alerts for fall detection.

## Features

- **Real-time Human Detection & Tracking**: Uses YOLOv11 for person detection with persistent ID tracking
- **Automatic PTZ Control**: Camera automatically follows selected target person
- **Action Classification**: Detects 4 human actions:
  - Standing
  - Walking
  - Sitting
  - Falling (with automatic alerts)
- **Fall Detection Alerts**: Sends WhatsApp notifications with images when falls are detected
- **Re-identification**: Can recognize previously seen persons even after they leave and return to the frame
- **Multi-person Tracking**: Maintains unique IDs for multiple people simultaneously

## Prerequisites

### Hardware
- PTZ camera with V4L2 support (tested with USB PTZ cameras)
- NVIDIA GPU (recommended for TensorRT acceleration)

### Software
- Python 3.8+
- CUDA and cuDNN (for GPU acceleration)
- TensorRT (optional, for optimized inference)

## Installation Steps

### 1. Python Dependencies Setup

First, install the required Python packages in the correct order:

```bash
# Install specific setuptools version to avoid compatibility issues
pip install "setuptools<66.0.0"

# Install OpenCV for computer vision functionality
pip install opencv-python

# Upgrade setuptools after OpenCV installation
pip install --upgrade setuptools

# Install V4L2 for video device access
pip install v4l2

# Fix V4L2 compatibility issue with Python 3.10+
sed -i 's/range(\([^)]*\)) + \[\([^]]*\)\]/list(range(\1)) + [\2]/g' /usr/local/lib/python3.10/dist-packages/v4l2.py

# Install specific NumPy version for compatibility
pip install --upgrade numpy==1.24.3

# Install Twilio for communication features
pip install twilio

# Install python-dotenv for environment variable management
pip install dotenv
```

### 2. X11 Display Setup (If Having XCB Problems)

If you encounter XCB (X11 Connection Block) related issues, configure the display environment:

```bash
# Set display variable
export DISPLAY=:1

# Allow X11 connections (run this on the Jetson host)
DISPLAY=:1 xhost +
```

**Note:** The `xhost +` command should be run on the Jetson device itself, not inside the Docker container.

### 3. Docker Container Setup

Pull and run the Edge Tracking Docker container:

```bash
# Pull the edge-tracking Docker image
docker pull xlabub/edge-tracking

# Run the container with all necessary configurations
docker run -it \
  --privileged \
  --ipc=host \
  --runtime=nvidia \
  -e DISPLAY=${DISPLAY} \
  -e LD_LIBRARY_PATH="/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:${LD_LIBRARY_PATH}" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /var/lock:/var/lock \
  -v /opt/nvidia/nsight-systems/2024.5.4:/opt/nvidia/nsight-systems/2024.5.4 \
  -v /usr/share/doc/nsight-compute-2025.1.1:/usr/share/doc/nsight-compute-2025.1.1 \
  -v /usr/local/cuda:/usr/local/cuda \
  -v /usr/local/cuda-12.6:/usr/local/cuda-12.6 \
  --network=host \
  --device=/dev/video0:/dev/video0 \
  xlabub/edge-tracking
```

## Docker Run Command Explanation

| Parameter | Purpose |
|-----------|---------|
| `--privileged` | Grants extended privileges to the container |
| `--ipc=host` | Uses host IPC namespace for shared memory |
| `--runtime=nvidia` | Uses NVIDIA container runtime for GPU access |
| `-e DISPLAY=${DISPLAY}` | Passes display environment for GUI applications |
| `-e LD_LIBRARY_PATH=...` | Sets library paths for CUDA and system libraries |
| `-e NVIDIA_VISIBLE_DEVICES=all` | Makes all NVIDIA GPUs available |
| `-e NVIDIA_DRIVER_CAPABILITIES=all` | Enables all NVIDIA driver capabilities |
| `-v /tmp/.X11-unix:/tmp/.X11-unix` | Mounts X11 socket for GUI display |
| `-v /var/lock:/var/lock` | Mounts system lock directory |
| `-v /opt/nvidia/...` | Mounts NVIDIA profiling tools |
| `-v /usr/local/cuda...` | Mounts CUDA installation directories |
| `--network=host` | Uses host network stack |
| `--device=/dev/video0:/dev/video0` | Provides access to camera device |

## Troubleshooting

### Common Issues

1. **XCB Connection Issues**
   - Ensure X11 server is running
   - Run `DISPLAY=:1 xhost +` on the Jetson host
   - Verify the DISPLAY environment variable is set correctly

2. **Camera Access Problems**
   - Check if camera is connected and detected: `ls /dev/video*`
   - Ensure camera permissions are correct
   - Verify camera is not being used by another process

3. **CUDA/GPU Issues**
   - Verify NVIDIA runtime is installed: `docker info | grep nvidia`
   - Check GPU availability: `nvidia-smi`
   - Ensure CUDA paths are correctly mounted

4. **Python Package Conflicts**
   - Follow the exact installation order provided
   - The setuptools version constraint is critical for compatibility

## Notes

- The specific NumPy version (1.24.3) is required for compatibility with the edge tracking system
- The sed command fixes a Python 3.10 compatibility issue in the v4l2 package
- All CUDA and system library paths are configured for aarch64 (ARM64) architecture
- The container runs with host networking to simplify communication setup