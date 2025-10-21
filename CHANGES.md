# Vehicle Detection System Refactoring - Change Summary

## Overview
This document summarizes the changes made to update the vehicle detection system to use YOLOv8n and DepthAI's built-in ObjectTracker node, along with improvements to the line crossing logic.

## Changes Made

### 1. YOLOv8n Integration (pedestrian_vehicle_detector.py)

#### Before:
- Used YOLOv7-tiny with anchor-based detection
- Required manual anchor configuration with specific anchor values and masks
- Model path: `models/yolov7-tiny_416x416.blob`

#### After:
- Uses YOLOv8n with anchor-free detection
- Removed all anchor and anchor mask configurations (lines 29-33 removed)
- Model path updated to: `models/yolov8n_416x416.blob`
- Added DepthAI ObjectTracker node to the pipeline

**Key Code Changes:**
```python
# Removed anchor configuration:
# self.detectionNetwork.setAnchors([...])
# self.detectionNetwork.setAnchorMasks({...})

# Added ObjectTracker node:
self.objectTracker = self.pipeline.create(dai.node.ObjectTracker)
self.objectTracker.setDetectionLabelsToTrack([0, 2, 3, 5, 7])
self.objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
self.objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
```

**Pipeline Architecture:**
```
Input Frame → YoloDetectionNetwork → ObjectTracker → Output Tracklets
                         ↓
                   Passthrough Frame
                         ↓
                   ObjectTracker
```

### 2. Object Tracker Refactoring (object_tracker.py)

#### Before:
- Custom SORT (Simple Online and Realtime Tracking) implementation
- Used Kalman filters for prediction
- Hungarian algorithm for data association
- Manual tracking ID assignment
- Dependencies: scipy, filterpy

#### After:
- Uses DepthAI's built-in ObjectTracker node
- Tracking handled by hardware/firmware
- Simplified software implementation
- Reduced dependencies (removed scipy, filterpy)
- Direct tracklet processing from DepthAI pipeline

**Key Removals:**
- `KalmanBoxTracker` class (lines 48-107)
- `bbox_iou()` function
- `convert_bbox_to_z()` and `convert_z_to_bbox()` functions
- `associate_detections_to_trackers()` method with Hungarian algorithm

**New Implementation:**
```python
class DepthAIObjectTracker:
    """Object Tracker using DepthAI's built-in ObjectTracker node"""
    def update(self, detections, timestamp):
        # Process tracklets directly from DepthAI
        for det in detections:
            track_id = det.get('track_id', None)  # ID from DepthAI
            # Update or create TrackedObject
```

### 3. Line Crossing Logic Enhancement (line_crossing.py)

#### Changes:
- Added comprehensive documentation to the `_intersects()` method
- Clarified the counter-clockwise (CCW) test algorithm
- No functional changes - logic was already correct per specifications

**Algorithm Verification:**
The line intersection algorithm correctly implements the CCW test:
1. Checks if endpoints of trajectory segment are on opposite sides of crossing line
2. Checks if endpoints of crossing line are on opposite sides of trajectory segment
3. Both conditions must be true for intersection

This matches the requirements in `specs correct.pdf` for detecting when an object's trajectory crosses a predefined line.

### 4. Configuration Updates (utils/config.py)

```python
# Before:
self.model_path = "models/yolov7-tiny_416x416.blob"

# After:
self.model_path = "models/yolov8n_416x416.blob"
```

### 5. Main Application Updates (main.py)

```python
# Before:
from tracking.object_tracker import SORTTracker
tracker = SORTTracker(config)

# After:
from tracking.object_tracker import DepthAIObjectTracker
tracker = DepthAIObjectTracker(config)
```

### 6. Build Hygiene (.gitignore)

Added comprehensive .gitignore to exclude:
- Python cache files (`__pycache__/`)
- Build artifacts
- Log files and detection images
- Model files (*.blob)
- Video files

## Technical Benefits

1. **Performance**: Hardware-accelerated tracking in DepthAI ObjectTracker is more efficient than software SORT
2. **Accuracy**: YOLOv8n provides better detection accuracy than YOLOv7-tiny
3. **Simplicity**: Reduced code complexity by ~150 lines
4. **Dependencies**: Removed scipy and filterpy dependencies
5. **Maintainability**: Less custom tracking code to maintain

## Compatibility

- **Maintained**: All TrackedObject methods for logging system
- **Maintained**: Line crossing detection interface
- **Maintained**: Log file format per Italian specification
- **Changed**: Detection pipeline now outputs tracklets with IDs
- **Changed**: Model file requirement (yolov8n instead of yolov7-tiny)

## Testing Requirements

1. **Model File**: Ensure `models/yolov8n_416x416.blob` exists
2. **Video Input**: Test with sample video at `videos/people.mp4`
3. **Line Crossing**: Verify crossing detection with configured lines
4. **Log Output**: Validate log format matches specification
5. **Performance**: Check frame rate and tracking stability

## Migration Notes

To migrate from the old system:
1. Replace model file: `yolov7-tiny_416x416.blob` → `yolov8n_416x416.blob`
2. No changes needed to configuration files (crossing lines, etc.)
3. No changes needed to logging system
4. Existing log files remain compatible

## Security

- CodeQL analysis: **0 vulnerabilities found**
- No new external dependencies introduced
- Removed complex mathematical libraries (scipy, filterpy)

## Specification Compliance

All changes comply with requirements in `specs correct.pdf`:
- ✅ Object detection and tracking
- ✅ Unique ID assignment (P for pedestrians, V for vehicles)
- ✅ Position tracking with coordinates
- ✅ Speed calculation
- ✅ Line crossing detection
- ✅ Log file format (rilevazione_yyyy-mm-dd.tt)
- ✅ Crossing image capture (fg_yyyy-mm-dd folder)
- ✅ Italian vehicle classification codes
