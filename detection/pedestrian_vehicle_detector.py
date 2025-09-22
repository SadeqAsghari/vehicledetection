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
        
        # Build pipeline for YOLOv7-tiny
        self.pipeline = dai.Pipeline()
        
        self.xinFrame = self.pipeline.create(dai.node.XLinkIn)
        self.xinFrame.setStreamName("inFrame")
        
        self.detectionNetwork = self.pipeline.create(dai.node.YoloDetectionNetwork)
        self.detectionNetwork.setBlobPath(self.model_path)
        self.detectionNetwork.setConfidenceThreshold(0.5)
        self.detectionNetwork.setNumClasses(80)
        self.detectionNetwork.setCoordinateSize(4)
        self.detectionNetwork.setIouThreshold(0.5)
        
        # YOLOv7-tiny uses 416x416 input
        # Note: anchors might need adjustment based on your specific model
        self.detectionNetwork.setAnchors([10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326])
        self.detectionNetwork.setAnchorMasks({
            "side26": [0, 1, 2],  # 26x26 grid
            "side13": [3, 4, 5],  # 13x13 grid
        })
        
        self.xinFrame.out.link(self.detectionNetwork.input)
        
        self.xoutDet = self.pipeline.create(dai.node.XLinkOut)
        self.xoutDet.setStreamName("detections")
        self.detectionNetwork.out.link(self.xoutDet.input)
        
        self.device = dai.Device(self.pipeline)
        self.detectionQueue = self.device.getOutputQueue("detections", maxSize=4, blocking=False)
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
            
        # Resize to 416x416 for YOLOv7-tiny
        resized = cv2.resize(frame, (416, 416))
        frame_nn = dai.ImgFrame()
        frame_nn.setData(self.to_planar(resized))
        frame_nn.setTimestamp(time.monotonic())
        frame_nn.setWidth(416)
        frame_nn.setHeight(416)
        frame_nn.setType(dai.ImgFrame.Type.BGR888p)
        
        self.inputQueue.send(frame_nn)
        
        # Get detections
        inDet = self.detectionQueue.get()
        detections = []
        timestamp = time.time()
        
        for det in inDet.detections:
            label_id = det.label
            label = self.label_map.get(label_id)
            if not label:
                continue
                
            # Scale coordinates back to original frame size
            x1 = int(det.xmin * frame.shape[1])
            y1 = int(det.ymin * frame.shape[0])
            x2 = int(det.xmax * frame.shape[1])
            y2 = int(det.ymax * frame.shape[0])
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0]))
            x2 = max(0, min(x2, frame.shape[1]))
            y2 = max(0, min(y2, frame.shape[0]))
            
            detections.append({
                "id": len(detections),
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "confidence": det.confidence,
                "timestamp": timestamp
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