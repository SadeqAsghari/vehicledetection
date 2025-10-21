# object_tracker.py - Using DepthAI's ObjectTracker node
import numpy as np
from collections import deque
import time
import depthai as dai

def get_centroid(bbox):
    """Get centroid of bounding box"""
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

class TrackedObject:
    """Maintains compatibility with existing logging system"""
    def __init__(self, object_id, label, bbox, timestamp):
        self.track_id = object_id
        self.label = label
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.positions = [(get_centroid(bbox), timestamp)]
        self.start_time = timestamp
        self.trail = deque(maxlen=30)
        self.trail.append(get_centroid(bbox))
        self.last_seen = timestamp
        self.type = 'P' if label == 'person' else 'V'
        self.license_plate = 'XXXXXXX '  # 9 characters space-padded
        self.nationality = 'I  ' if self.type == 'V' else ''  # 3 characters space-padded
        
        # Italian vehicle type codes
        vehicle_type_map = {
            'car': 'A',      # autoveicolo
            'motorcycle': 'M', # motoveicolo  
            'bus': 'F',      # filobus
            'truck': 'R',    # rimorchio
            'bicycle': 'V'   # velocipede
        }
        self.vehicle_type_code = vehicle_type_map.get(label, 'X') if self.type == 'V' else ''
        
    def update(self, bbox, timestamp):
        self.bbox = bbox
        centroid = get_centroid(bbox)
        self.positions.append((centroid, timestamp))
        self.trail.append(centroid)
        self.last_seen = timestamp
        
    def get_latest_position(self):
        return (*get_centroid(self.bbox), self.last_seen)
        
    def get_speed(self):
        """Calculate speed in hundredths of km/h"""
        if len(self.positions) < 2:
            return 0
        
        (x1, y1), t1 = self.positions[-2]
        (x2, y2), t2 = self.positions[-1]
        dt = max(t2 - t1, 0.001)
        
        dist_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        # Assuming 1 pixel = 1 cm (adjust based on camera calibration)
        dist_cm = dist_px
        dist_km = dist_cm / 100000.0
        speed_kmh = (dist_km / dt) * 3600
        return int(speed_kmh * 100)  # Return in hundredths
        
    def get_total_distance_cm(self):
        """Calculate total distance in centimeters"""
        dist = 0
        for i in range(1, len(self.positions)):
            p1 = self.positions[i-1][0]
            p2 = self.positions[i][0]
            dist += np.linalg.norm(np.array(p2) - np.array(p1))
        return int(dist)
        
    def get_elapsed_time_s(self):
        """Get elapsed time in seconds"""
        return int(self.last_seen - self.start_time)
        
    def get_average_speed(self):
        """Calculate average speed in km/h"""
        if len(self.positions) < 2:
            return 0
        total_time = self.get_elapsed_time_s()
        if total_time == 0:
            return 0
        total_distance = self.get_total_distance_cm() / 100000  # cm to km
        avg_speed = total_distance / (total_time / 3600)  # km/h
        return round(avg_speed, 2)

class DepthAIObjectTracker:
    """Object Tracker using DepthAI's built-in ObjectTracker node"""
    def __init__(self, config):
        self.config = config
        self.tracked_objects = {}
        self.frame_count = 0
        self.total_detected = 0
        self.next_id = 0
        
    def update(self, detections, timestamp):
        """Update tracker with new detections from DepthAI"""
        self.frame_count += 1
        
        # Process detections and update tracked objects
        current_ids = set()
        
        for det in detections:
            # Use track_id from detection if available, otherwise assign new ID
            track_id = det.get('track_id', None)
            
            if track_id is None:
                # Assign new ID for untracked detection
                track_id = self.next_id
                self.next_id += 1
                
            current_ids.add(track_id)
            bbox = det['bbox']
            label = det['label']
            
            # Update or create TrackedObject
            if track_id not in self.tracked_objects:
                obj = TrackedObject(track_id, label, bbox, timestamp)
                self.tracked_objects[track_id] = obj
                self.total_detected += 1
            else:
                self.tracked_objects[track_id].update(bbox, timestamp)
        
        # Remove stale tracked objects (not seen for multiple frames)
        max_age = 30  # frames
        stale_ids = []
        for track_id, obj in self.tracked_objects.items():
            if track_id not in current_ids:
                age = self.frame_count - (obj.last_seen if hasattr(obj, 'last_frame') else 0)
                if not hasattr(obj, 'last_frame'):
                    obj.last_frame = self.frame_count
                else:
                    obj.last_frame = obj.last_frame if track_id not in current_ids else self.frame_count
                    
                if self.frame_count - obj.last_frame > max_age:
                    stale_ids.append(track_id)
            else:
                obj.last_frame = self.frame_count
                    
        for track_id in stale_ids:
            del self.tracked_objects[track_id]
                    
        return list(self.tracked_objects.values())
        
    def draw_debug_info(self, frame):
        """Draw debug information on frame"""
        import cv2
        cv2.putText(frame, f"Total Tracked: {len(self.tracked_objects)}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Total Detected: {self.total_detected}", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                   
        for obj in self.tracked_objects.values():
            for i in range(1, len(obj.trail)):
                cv2.line(frame, obj.trail[i - 1], obj.trail[i], (255, 0, 0), 2)

# For backward compatibility
ObjectTracker = DepthAIObjectTracker
SORTTracker = DepthAIObjectTracker