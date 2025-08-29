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

## Quick Start

### 1. Pull the Docker Image

```bash
docker pull xlabub/edge-tracking
```

### 2. Run the Container

```bash
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

### 3. Configure Display (On Jetson Host)

Before running the application, configure the X11 display:

```bash
export DISPLAY=:1
DISPLAY=:1 xhost +
```

**Note**: The `xhost +` command disables access control for X11. For security purposes, consider using `xhost +local:docker` instead.

### 4. Run the Edge Tracking Application

Inside the container:

```bash
cd edge-tracking
python main.py
```

## Configuration Details

### Docker Run Parameters Explained

- `--privileged`: Grants extended privileges to the container
- `--ipc=host`: Uses the host's IPC namespace for shared memory
- `--runtime=nvidia`: Uses the NVIDIA container runtime for GPU access
- `--network=host`: Uses the host's network stack
- `--device=/dev/video0:/dev/video0`: Maps the camera device into the container

### Environment Variables

- `DISPLAY`: X11 display configuration
- `LD_LIBRARY_PATH`: Library search paths for CUDA and Tegra libraries
- `NVIDIA_VISIBLE_DEVICES=all`: Makes all NVIDIA GPUs visible to the container
- `NVIDIA_DRIVER_CAPABILITIES=all`: Enables all NVIDIA driver capabilities

### Volume Mounts

The container mounts several important directories:

- `/tmp/.X11-unix`: X11 socket for GUI applications
- `/var/lock`: System lock files
- `/opt/nvidia/nsight-systems/2024.5.4`: NVIDIA Nsight Systems profiler
- `/usr/share/doc/nsight-compute-2025.1.1`: NVIDIA Nsight Compute documentation
- `/usr/local/cuda` and `/usr/local/cuda-12.6`: CUDA runtime and libraries