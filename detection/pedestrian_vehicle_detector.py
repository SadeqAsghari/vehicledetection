import cv2
import depthai as dai
import numpy as np
import time

class PedestrianVehicleDetector:
    def __init__(self, config):
        self.config = config
        self.label_map = config.label_map
        self.model_path = config.model_path
        self.video_path = config.test_video_path
        self.cap = cv2.VideoCapture(self.video_path)
        
        # Build pipeline for YOLOv8n (anchor-free) with ObjectTracker
        self.pipeline = dai.Pipeline()
        
        self.xinFrame = self.pipeline.create(dai.node.XLinkIn)
        self.xinFrame.setStreamName("inFrame")
        
        self.detectionNetwork = self.pipeline.create(dai.node.YoloDetectionNetwork)
        self.detectionNetwork.setBlobPath(self.model_path)
        self.detectionNetwork.setConfidenceThreshold(0.5)
        self.detectionNetwork.setNumClasses(80)
        self.detectionNetwork.setCoordinateSize(4)
        self.detectionNetwork.setIouThreshold(0.5)
        
        # YOLOv8n is anchor-free, no anchors or anchor masks needed
        
        # Create ObjectTracker node for tracking
        self.objectTracker = self.pipeline.create(dai.node.ObjectTracker)
        self.objectTracker.setDetectionLabelsToTrack([0, 2, 3, 5, 7])  # person, car, motorcycle, bus, truck
        # Set tracker type to SHORT_TERM_KCF or ZERO_TERM_COLOR_HISTOGRAM
        self.objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        # Set tracking threshold
        self.objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
        
        self.xinFrame.out.link(self.detectionNetwork.input)
        self.detectionNetwork.passthrough.link(self.objectTracker.inputTrackerFrame)
        self.detectionNetwork.out.link(self.objectTracker.inputDetectionFrame)
        self.detectionNetwork.passthrough.link(self.objectTracker.inputTrackerFrame)
        
        self.xoutTrack = self.pipeline.create(dai.node.XLinkOut)
        self.xoutTrack.setStreamName("tracklets")
        self.objectTracker.out.link(self.xoutTrack.input)
        
        self.device = dai.Device(self.pipeline)
        self.trackQueue = self.device.getOutputQueue("tracklets", maxSize=4, blocking=False)
        self.inputQueue = self.device.getInputQueue("inFrame")
        
        self.frame_count = 0
        self.frame = None
        
    def pipeline_context(self):
        class DummyContext:
            def __enter__(self_):
                return self
            def __exit__(self_, *a):
                pass
        return DummyContext()
        
    def get_frame_and_detections(self, device):
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            print("End of video or invalid frame.")
            exit()
            
        # Resize to 416x416 for YOLOv8n
        resized = cv2.resize(frame, (416, 416))
        frame_nn = dai.ImgFrame()
        frame_nn.setData(self.to_planar(resized))
        frame_nn.setTimestamp(time.monotonic())
        frame_nn.setWidth(416)
        frame_nn.setHeight(416)
        frame_nn.setType(dai.ImgFrame.Type.BGR888p)
        
        self.inputQueue.send(frame_nn)
        
        # Get tracklets from ObjectTracker
        track_data = self.trackQueue.get()
        detections = []
        timestamp = time.time()
        
        for tracklet in track_data.tracklets:
            label_id = tracklet.label
            label = self.label_map.get(label_id)
            if not label:
                continue
            
            # Get ROI (region of interest) from tracklet
            roi = tracklet.roi.denormalized(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0]))
            x2 = max(0, min(x2, frame.shape[1]))
            y2 = max(0, min(y2, frame.shape[0]))
            
            detections.append({
                "track_id": tracklet.id,
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "confidence": tracklet.status.name,  # TRACKED, LOST, NEW
                "timestamp": timestamp,
                "status": tracklet.status
            })
            
        self.frame = frame
        return type("Frame", (), {"frame": frame, "timestamp": timestamp}), detections
        
    def to_planar(self, frame):
        """Convert BGR frame to planar format for DepthAI"""
        return frame.transpose(2, 0, 1).flatten().tolist()
        
    def display(self, frame, tracked_objects):
        """Display tracked objects on frame"""
        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {obj.track_id} {obj.label}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                       
        cv2.imshow("Video Detection with DepthAI", frame)
        if cv2.waitKey(1) == ord('q'):
            exit()