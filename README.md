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

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ptz-camera-tracking
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download YOLOv11 models**
```bash
# The models will be downloaded automatically on first run
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n-pose.pt
```

5. **Set up environment variables**

Create a `.env` file in the project root:
```env
# Twilio credentials for WhatsApp alerts
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886  # Twilio sandbox number
YOUR_WHATSAPP_NUMBER=whatsapp:+1234567890     # Your WhatsApp number

# FreeImage.host API key for image hosting
FREEIMAGE_API_KEY=your_freeimage_api_key
```

## Configuration

### Camera Settings
Edit `PTZ_camera.py` to adjust camera parameters:
```python
# Resolution
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Frame rate
self.cap.set(cv2.CAP_PROP_FPS, 30)
```

### Tracking Parameters
In `person_tracker.py`:
```python
max_disappeared = 15  # Frames before person is considered lost
max_distance = 100    # Max pixel distance for matching
active_matching_threshold = 0.95
reidentification_threshold = 0.065
```

### Fall Detection Settings
In `action_classifier.py`:
```python
fall_alert_interval = 60  # Seconds between fall alerts
```

## Usage

### Basic Operation

1. **Run the main program**
```bash
python main.py
```

2. **Keyboard Controls**
- `t` - Toggle tracking mode on/off
- `c` - Select closest person to center as tracking target
- `←/→` - Select previous/next person as tracking target
- `r` - Reset camera position
- `q` - Quit

3. **GUI Controls**
- **Pan Speed**: 0=Left, 1=Stop, 2=Right
- **Tilt Speed**: 0=Down, 1=Stop, 2=Up
- **Zoom**: Slide to adjust zoom level

### Tracking Modes

1. **Manual Mode** (default)
   - Camera stays stationary
   - All detected persons are shown with IDs
   - Actions are classified and displayed

2. **Tracking Mode** (press 't')
   - Camera automatically follows selected target
   - If target is lost, camera performs search sweep
   - Can manually select different targets with arrow keys

## System Architecture

### Main Components

1. **main.py**
   - Entry point and main loop
   - Handles UI and user input
   - Coordinates all components

2. **PTZ_camera.py**
   - Camera control interface
   - Implements tracking logic
   - Manages pan/tilt/zoom operations

3. **person_tracker.py**
   - YOLOv11-based person detection
   - Feature extraction for re-identification
   - ID assignment and tracking logic

4. **action_classifier.py**
   - Pose estimation using YOLOv11-pose
   - Action classification logic
   - Fall detection algorithm

5. **twilio_whatsapp.py**
   - WhatsApp alert functionality
   - Image upload to cloud service

### Detection Pipeline

1. **Person Detection**: YOLOv11 detects all persons in frame
2. **Feature Extraction**: HOG + color histograms for each person
3. **ID Assignment**: Match with existing tracked persons or assign new ID
4. **Pose Estimation**: Extract keypoints for each person
5. **Action Classification**: Analyze keypoints to determine action
6. **Fall Detection**: Special checks for horizontal body position
7. **Alert System**: Send WhatsApp alert if fall detected

## Troubleshooting

### Common Issues

1. **Camera not found**
   - Check camera is connected: `ls /dev/video*`
   - Verify camera permissions: `sudo chmod 666 /dev/video0`

2. **Low FPS**
   - Reduce resolution in PTZ_camera.py
   - Enable TensorRT optimization (automatic on first run)
   - Check GPU is being used: `nvidia-smi`

3. **WhatsApp alerts not working**
   - Verify Twilio credentials in .env file
   - Ensure phone number is verified with Twilio
   - Check internet connection

4. **Tracking loses person frequently**
   - Adjust `max_disappeared` parameter
   - Increase `active_matching_threshold`
   - Improve lighting conditions

### Debug Mode

Enable verbose output by modifying print statements in the code or adding debug flags.

## Performance Optimization

1. **TensorRT Acceleration**
   - Models are automatically exported to TensorRT on first run
   - Creates `.engine` files for faster inference

2. **Resolution vs Speed Trade-off**
   - 640x480: ~30 FPS on modern GPU
   - 1280x720: ~15-20 FPS
   - 1920x1080: ~8-12 FPS

3. **Feature Extraction Optimization**
   - Adjust histogram bins in person_tracker.py
   - Reduce HOG feature dimensions

## Future Improvements

- [ ] Add support for multiple camera switching
- [ ] Implement zone-based alerts
- [ ] Add person counting statistics
- [ ] Create web interface for remote monitoring
- [ ] Support for additional action types
- [ ] Integration with home automation systems

## License

[Specify your license here]

## Acknowledgments

- YOLOv11 by Ultralytics
- OpenCV community
- Twilio for WhatsApp API