# Task D: Multiple Human Tracking using YOLO11

Real-time multiple human tracking system using **YOLO11** with persistent object IDs.

## 📚 Overview

This implementation uses state-of-the-art YOLO11 (You Only Look Once) object detection model with built-in tracking capabilities to perform real-time tracking of multiple people in video streams. The system assigns unique IDs to each detected person and maintains these IDs consistently across video frames, even when people temporarily disappear or are occluded.

## ✨ Features

- ✅ Real-time human detection and tracking
- ✅ Persistent object IDs across frames
- ✅ Bounding box visualization with track IDs
- ✅ Video output with tracking annotations
- ✅ Easy-to-use command-line interface
- ✅ Automatic GPU/CPU detection
- ✅ High accuracy person detection
- ✅ Handles occlusions and re-identification

## 🚀 Quick Start

### 1. Create Virtual Environment (Recommended)

Using a virtual environment isolates project dependencies and prevents conflicts.

**On Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` prefix in your terminal after activation.

### 2. Install Dependencies

**With virtual environment activated:**

```bash
pip install -r requirements.txt
```

**Note:** If you have an NVIDIA GPU and want GPU acceleration:

```bash
# For CUDA 12.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Then install other dependencies:
pip install -r requirements.txt
```

### 3. Run Human Tracking

**Basic usage:**

```bash
python main.py <path_to_video>
```

**Example:**

```bash
python main.py test/sample_video.mp4
```

**Output:**

- Processed video saved to `output/` directory with same filename
- Real-time preview window (press `q` to quit early)

**Tip:** To deactivate the virtual environment when done:

```bash
deactivate
```

## 🔧 How It Works

The tracking system uses a three-component pipeline:

### 1. YOLO11n - Object Detection

- Lightweight YOLO11 nano model for person detection
- Pre-trained on COCO dataset
- Fast inference (~5.6 MB model size)
- High accuracy person detection (class 0 in COCO)

### 2. Built-in Tracking

- YOLO11's built-in tracking algorithm
- Assigns unique IDs to each detected person
- Maintains IDs across frames with persistence
- Handles occlusions and re-identification

### 3. Visualization & Export

- Draws bounding boxes around each person
- Displays unique track ID above each box
- Exports annotated video to `output/` folder

### Processing Pipeline

```
Input Video → Frame Extraction → YOLO Detection (Person Class) 
→ Track ID Assignment → Annotation → Output Video
```

The model automatically:

- Detects all people in each frame
- Assigns unique IDs to each person
- Maintains IDs across frames even with occlusions
- Draws bounding boxes with track IDs
- Saves annotated video with same resolution and FPS

## 📦 Dependencies

Main libraries used in this project:

| Library | Version | Purpose |
|---------|---------|---------|
| ultralytics | 8.3.235 | YOLO11 model and tracking |
| opencv-python | 4.11.0.86 | Video processing and display |
| torch | 2.5.1 | Deep learning backend |
| torchvision | 0.20.1 | Vision utilities |
| numpy | 2.2.4 | Array operations |
| pillow | 11.2.0 | Image processing |
| scipy | 1.16.0 | Scientific computing |
| matplotlib | 3.10.7 | Plotting utilities |

See `requirements.txt` for the complete list of dependencies.

## 📊 Model Information

**Model**: YOLO11n (Nano)

- **Architecture**: YOLOv11 Nano variant
- **Size**: ~5.6 MB
- **Speed**: Real-time on CPU/GPU
- **Pre-training**: COCO dataset (80 classes)
- **Detection Class**: Person (class 0)
- **Input Format**: Video files (MP4, AVI, MOV, etc.)
- **Output Format**: MP4 video with H.264 codec

**Performance Characteristics**:

- Fast inference speed suitable for real-time applications
- Lightweight model suitable for edge devices
- High accuracy for person detection
- Supports both CPU and GPU inference

## 📂 Project Structure

```bash
multiple-human-tracking/
├── test/                   # Test videos
│   └── sample_video.mp4    # Example test video
├── output/                 # Tracking results
│   └── *.mp4              # Processed videos
├── main.py                 # Main tracking script
├── requirements.txt        # Python dependencies
├── yolo11n.pt             # YOLO11 pretrained model
└── README.md              # This file
```

## 📈 Output Files

After processing, find your results in the `output/` directory:

- **Filename**: Same as input video (e.g., `sample_video.mp4`)
- **Format**: MP4 with H.264 codec
- **Resolution**: Same as input video
- **FPS**: Same as input video
- **Annotations**: Bounding boxes with track IDs

## 📚 Resources

- **YOLO11 Documentation**: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **YOLO Tracking Guide**: [https://docs.ultralytics.com/modes/track/](https://docs.ultralytics.com/modes/track/)
- **OpenCV Documentation**: [https://docs.opencv.org/](https://docs.opencv.org/)
- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)

## 👨‍💻 Implementation Details

### Code Overview

The `main.py` script implements the following workflow:

1. **Argument Parsing**: Accepts video path as command-line argument
2. **Model Loading**: Loads YOLO11n pre-trained model
3. **Video Capture**: Opens input video and extracts properties (FPS, resolution)
4. **Video Writer Setup**: Creates output video writer with same properties
5. **Frame Processing Loop**:
   - Read frame from video
   - Run YOLO tracking with person class filter
   - Annotate frame with bounding boxes and IDs
   - Write to output video
   - Display in preview window
6. **Cleanup**: Release resources and save output

### Key Code Components

```python
# Model initialization
model = YOLO("yolo11n.pt")

# Tracking with persistence and class filtering
results = model.track(frame, persist=True, classes=[0])

# Visualization
annotated = result.plot()
```
