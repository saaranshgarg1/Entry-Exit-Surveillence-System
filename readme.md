# Real-time Person Tracking System with Door Entry/Exit Detection

A sophisticated computer vision system that combines person detection, face recognition, and depth-based door detection to track people entering and exiting through doorways.

NOTE: This uses a modified version of InsightFace library. Please refer to [this section](#patching-insightface-library-for-optimal-performance)
## Core Features

1. **Person Detection & Tracking**
   - Uses YOLO (You Only Look Once) for real-time person detection
   - Tracks individuals across frames using IoU (Intersection over Union)
   - Maintains persistent person IDs for continuous tracking

2. **Face Recognition**
   - Leverages InsightFace for robust face detection and recognition
   - Matches detected faces against a pre-loaded database
   - Associates faces with detected persons using spatial overlap

3. **Door Detection**
   - Uses DepthAnything V2 model for depth estimation
   - Detects door frames using depth map analysis
   - Supports both fully visible and partially visible door frames

4. **Entry/Exit Monitoring**
   - Tracks people's positions relative to detected doors
   - Uses depth comparison to determine if someone is entering or exiting
   - Logs all entry/exit events with confidence scores

## System Architecture

### Components

1. **AsyncDepthProcessor**
   - Handles depth estimation asynchronously
   - Uses a worker thread to prevent blocking the main loop
   - Maintains frame queues for smooth processing

2. **PersonTracker**
   - Manages person state tracking
   - Records entry/exit events
   - Maintains history of person movements
   - Logs activities to CSV file

3. **Face Management**
   - Loads face database from images
   - Performs real-time face matching
   - Associates faces with detected persons

### Key Functions

- `associate_face_person()`: Links detected faces with person bounding boxes
- `detect_door_from_depth()`: Processes depth maps to identify door frames
- `visualize_depth_heatmap()`: Creates visual representation of depth data
- `IOU_tracker()`: Tracks person identities between frames
- `match_face()`: Performs face recognition against database

## Data Flow

1. Frame Capture → Person Detection → Face Detection
2. Face-Person Association → Identity Matching
3. Door Detection → Depth Analysis
4. State Tracking → Event Logging

## Dependencies

- OpenCV (cv2)
- PyTorch
- Ultralytics YOLO
- InsightFace
- NumPy
- DepthAnything V2

## Configuration

### Required Models
- YOLO model (`yolo11n.pt`)
- Depth estimation models:
  - `depth_anything_v2_vitl.pth`
  - `depth_anything_v2_vits.pth`

### Directory Structure
```
project/   
├── faces/ # Face database images  
├── yolo11n.pt  
├── depth_anything_v2_vits.pth  
├── depth_anything_v2_vitl.pth  
└── person_tracking.csv
```


## Output Files

1. `person_tracking.csv`: Entry/exit event log with timestamps
   - Timestamp
   - Person Identity
   - Action (Entered/Exited)
   - Confidence Score

## Performance Considerations

- Async depth processing to maintain frame rate
- IoU-based tracking for performance
- Configurable frame skip for face recognition
- GPU acceleration support for all deep learning models

## Implementation Details

1. **Initialization**
   - Loads all required models
   - Initializes tracking systems
   - Performs initial door detection

2. **Main Loop**
   - Captures frame
   - Detects persons and faces
   - Updates tracking information
   - Monitors door activity
   - Updates display and logs

3. **State Management**
   - Tracks person locations
   - Manages identity associations
   - Handles disappearances
   - Updates entry/exit states

4. **Visualization**
   - Bounding boxes for persons and faces
   - Association lines
   - Identity labels with confidence
   - Inside/Outside status
   - FPS counter

## Error Handling

- Graceful handling of frame capture failures
- Model initialization timeout protection
- Recovery from tracking losses
- Robust face recognition matching

## Usage

1. Place face images in `faces/` directory
2. Ensure all model files are present
3. Run the main script:
```bash
python live2.py
```
4. System will automatically:
    - Detect doors in the first frame
    - Begin tracking people
    - Log all entry/exit events
  
  ## Patching `InsightFace` library for optimal performance
  Direct usage of `InsightFace` library was slightly inneficient, therefore I added a method in the `FaceAnalysis` class to use it more optimally.
  ```python
  def get2(self, img, bboxes, kpss):
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret
   ```

