# object_tracker.py - COMPLETELY REWRITTEN WITH SORT-BASED TRACKING
import numpy as np
from collections import deque
import time
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def bbox_iou(boxA, boxB):
    """Calculate Intersection over Union between two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

def get_centroid(bbox):
    """Get centroid of bounding box"""
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

def convert_bbox_to_z(bbox):
    """Convert [x1, y1, x2, y2] to [x, y, s, r] where x,y is center, s is scale, r is aspect ratio"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is area
    r = w / float(h) if h > 0 else 1.0  # aspect ratio
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_z_to_bbox(z):
    """Convert [x, y, s, r] back to [x1, y1, x2, y2]"""
    w = np.sqrt(z[2] * z[3])
    h = z[2] / w
    return np.array([z[0] - w/2., z[1] - h/2., z[0] + w/2., z[1] + h/2.]).flatten()

class KalmanBoxTracker:
    """Individual tracker using Kalman Filter for one object"""
    count = 0
    
    def __init__(self, bbox, label):
        # Initialize Kalman filter with constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        
        self.kf.R[2:, 2:] *= 10.  # Measurement noise
        self.kf.P[4:, 4:] *= 1000.  # Initial uncertainty in velocity
        self.kf.P *= 10.  # Initial uncertainty
        self.kf.Q[-1, -1] *= 0.01  # Process noise
        self.kf.Q[4:, 4:] *= 0.01
        
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.label = label
        
    def update(self, bbox):
        """Update the state with observed bbox"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        
    def predict(self):
        """Predict next state"""
        if (self.kf.x[2] + self.kf.x[6]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_z_to_bbox(self.kf.x))
        return self.history[-1]
        
    def get_state(self):
        """Return current bbox estimate"""
        return convert_z_to_bbox(self.kf.x).reshape((1, 4))[0]

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

class SORTTracker:
    """SORT-based Multi-Object Tracker"""
    def __init__(self, config, max_age=5, min_hits=3, iou_threshold=0.3):
        self.config = config
        self.max_age = max_age  # Max frames to keep alive without detection
        self.min_hits = min_hits  # Min hits before track is confirmed
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.tracked_objects = {}
        self.frame_count = 0
        self.total_detected = 0
        
    def update(self, detections, timestamp):
        """Update tracker with new detections"""
        self.frame_count += 1
        
        # Get predictions from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:4] = pos
            trk[4] = 0
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Remove invalid trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Extract detection info
        dets = []
        det_labels = []
        for det in detections:
            bbox = det['bbox']
            dets.append(bbox + [det.get('confidence', 0.9)])
            det_labels.append(det['label'])
            
        dets = np.array(dets) if dets else np.empty((0, 5))
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            dets, trks, self.iou_threshold)
        
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])
            
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            label = det_labels[i] if i < len(det_labels) else 'unknown'
            trk = KalmanBoxTracker(dets[i, :4], label)
            self.trackers.append(trk)
            self.total_detected += 1
            
        # Update tracked objects
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            
            # Update or create TrackedObject
            if trk.id not in self.tracked_objects:
                if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                    obj = TrackedObject(trk.id, trk.label, d, timestamp)
                    self.tracked_objects[trk.id] = obj
            else:
                self.tracked_objects[trk.id].update(d, timestamp)
                
            i -= 1
            # Remove dead tracklets
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                if trk.id in self.tracked_objects:
                    del self.tracked_objects[trk.id]
                    
        return list(self.tracked_objects.values())
        
    def associate_detections_to_trackers(self, detections, trackers, iou_threshold):
        """Associate detections to tracked objects using Hungarian algorithm"""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
            
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = bbox_iou(det[:4], trk[:4])
                
        if min(iou_matrix.shape) > 0:
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))
        else:
            matched_indices = np.empty(shape=(0, 2))
            
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
                
        # Filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
                
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
            
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
        
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

# For backward compatibility, alias SORTTracker as ObjectTracker
ObjectTracker = SORTTracker