import cv2
import datetime
from detection.pedestrian_vehicle_detector import PedestrianVehicleDetector
from tracking.object_tracker import SORTTracker
from tracking.line_crossing import LineCrossingDetector
from logs.logger import DetectionLogger
from utils.config import Config

# Initialize system components
config = Config()
detector = PedestrianVehicleDetector(config)
tracker = SORTTracker(config)  # Now using SORT tracker
crossing_detector = LineCrossingDetector(config)
logger = DetectionLogger(config)

# Prepare logging environment
logger.initialize_log_file()
logger.create_image_folder()

# Track mouse position for debugging
mouse_pos = (0, 0)

def mouse_callback(event, x, y, flags, param):
    global mouse_pos
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)

cv2.namedWindow("Video Detection with DepthAI")
cv2.setMouseCallback("Video Detection with DepthAI", mouse_callback)

# Main processing loop
with detector.pipeline_context() as device:
    while True:
        frame_wrapper, detections = detector.get_frame_and_detections(device)
        raw_frame = frame_wrapper.frame
        timestamp = frame_wrapper.timestamp
        
        # Update tracker with new detections
        tracked_objects = tracker.update(detections, timestamp)
        
        # Process each tracked object
        for obj in tracked_objects:
            # Check for line crossings
            crossing_code = crossing_detector.check_crossing(obj)
            
            # Log new objects
            if obj.track_id not in logger.logged_objects:
                logger.log_new_object(obj)
                logger.logged_objects.add(obj.track_id)
                
            # Log position
            logger.log_position(obj)
            
            # Handle line crossing
            if crossing_code:
                logger.log_crossing(obj, crossing_code)
                logger.save_crossing_frame(obj, raw_frame)
                logger.log_crossing_debug(obj, crossing_code, timestamp)
                
                # Set blink counter for the crossed line
                for line in config.crossing_lines:
                    if line["code"] == crossing_code:
                        line["blink_counter"] = 10  # Blink for 10 frames
                        
        # Draw debug info
        tracker.draw_debug_info(raw_frame)
        
        # Draw crossing lines with blinking effect
        for line in config.crossing_lines:
            p1, p2 = line["p1"], line["p2"]
            
            # Determine line color (red if blinking, green otherwise)
            if line["blink_counter"] > 0:
                color = (0, 0, 255)  # Red when recently crossed
                line["blink_counter"] -= 1  # Decrement counter
            else:
                color = (0, 255, 0)  # Green normally
                
            cv2.line(raw_frame, p1, p2, color, 2)
            cv2.putText(raw_frame, line["code"], 
                       (p1[0], p1[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw time information
        video_seconds = int(timestamp)
        cv2.putText(raw_frame, f"Time: {video_seconds}s",
                   (raw_frame.shape[1] - 160, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw mouse coordinates
        cv2.putText(raw_frame, f"X: {mouse_pos[0]}, Y: {mouse_pos[1]}",
                   (10, raw_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame with tracked objects
        detector.display(raw_frame, tracked_objects)