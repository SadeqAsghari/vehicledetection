# logger.py - COMPLETE IMPLEMENTATION

import os
import datetime
from pathlib import Path
from PIL import Image
import cv2

class DetectionLogger:
    def __init__(self, config):
        self.config = config
        self.today = datetime.date.today().strftime("%Y-%m-%d")
        self.log_dir = os.path.join(config.log_root, f"rilevazione_{self.today}.tt")
        self.image_dir = os.path.join(config.log_root, f"fg_{self.today}")
        self.logged_objects = set()

    def initialize_log_file(self):
        """Initialize the daily log file"""
        Path(self.config.log_root).mkdir(parents=True, exist_ok=True)
        with open(self.log_dir, 'w') as f:
            f.write("")  # Reset contents

    def create_image_folder(self):
        """Create the daily image folder for crossing frames"""
        os.makedirs(self.image_dir, exist_ok=True)

    def log_new_object(self, obj):
        """Log new object identification with proper formatting
        
        Format: pedoneVeicolo(1) + Progressivo(7) + [tipoVeicolo(1) + Nazionalita(3) + Targa(9)]
        Example for person: P0000001
        Example for vehicle: V0000002AI AA000AA
        """
        prog = str(obj.track_id).zfill(7)  # 7-digit progressive, zero-padded
        line = f"{obj.type}{prog}"
        
        if obj.type == 'V':
            # Vehicle fields: type code + nationality + license plate
            vehicle_type = obj.vehicle_type_code if obj.vehicle_type_code else 'X'
            
            # Nationality: 3 characters, space-padded if shorter
            nationality = obj.nationality if obj.nationality else 'I'
            nationality = nationality[:3].ljust(3)  # Truncate to 3 chars and pad with spaces
            
            # License plate: 9 characters, space-padded if shorter
            license_plate = obj.license_plate if obj.license_plate else 'XXXXXXX'
            license_plate = license_plate[:9].ljust(9)  # Truncate to 9 chars and pad with spaces
            
            line += f"{vehicle_type}{nationality}{license_plate}"
        
        with open(self.log_dir, 'a') as f:
            f.write(line + "\n")

    def log_position(self, obj):
        """Log position update with proper formatting
        
        Format: pedoneVeicolo(1) + Progressivo(7) + numPosizione(3) + coordinataX(5) + 
                coordinataY(5) + velocitàAttuale(5) + distanzaTotale(6) + tempoTotale(5)
        Example: V 0040001 023 00800 00800 00400 000250 00600
        """
        prog = str(obj.track_id).zfill(7)  # 7-digit progressive
        pos_num = str(len(obj.positions)).zfill(3)  # 3-digit position number
        
        # Get current position coordinates
        x, y, _ = obj.get_latest_position()
        x_str = str(int(x)).zfill(5)  # 5-digit X coordinate
        y_str = str(int(y)).zfill(5)  # 5-digit Y coordinate
        
        # Speed in hundredths of km/h (5 digits)
        speed = str(obj.get_speed()).zfill(5)
        
        # Total distance in centimeters (6 digits)
        dist = str(obj.get_total_distance_cm()).zfill(6)
        
        # Total elapsed time in seconds (5 digits)
        time_s = str(obj.get_elapsed_time_s()).zfill(5)
        
        # Format: type + space + prog + space + pos_num + space + x + space + y + space + speed + space + dist + space + time
        line = f"{obj.type} {prog} {pos_num} {x_str} {y_str} {speed} {dist} {time_s}"
        
        with open(self.log_dir, 'a') as f:
            f.write(line + "\n")

    def log_crossing(self, obj, line_code):
        """Log line crossing event with proper formatting
        
        Format: tipoRiga(1) + progressivo(7) + istante(10) + codiceLinea(5)
        Example: A 0040001 1731944191 AC001
        """
        prog = str(obj.track_id).zfill(7)  # 7-digit progressive
        timestamp = str(int(datetime.datetime.now().timestamp())).zfill(10)  # 10-digit timestamp
        code = line_code[:5].ljust(5)  # 5-character line code, space-padded
        
        # Format: A + space + prog + space + timestamp + space + code
        line = f"A {prog} {timestamp} {code}"
        
        with open(self.log_dir, 'a') as f:
            f.write(line + "\n")

    def save_crossing_frame(self, obj, frame):
        """Save frame when object crosses a line
        
        Filename format: progressive_instant.jpeg
        Example: 0040001_1731944191.jpeg
        """
        timestamp = str(int(datetime.datetime.now().timestamp()))
        filename = f"{str(obj.track_id).zfill(7)}_{timestamp}.jpeg"
        filepath = os.path.join(self.image_dir, filename)
        
        # Ensure image directory exists
        os.makedirs(self.image_dir, exist_ok=True)
        
        if isinstance(frame, Image.Image):
            # If frame is already a PIL Image
            frame.save(filepath)
        else:
            # If frame is a numpy array (OpenCV format)
            cv2.imwrite(filepath, frame)

    def log_object_summary(self, obj):
        """Log object summary for debugging (optional)"""
        prog = str(obj.track_id).zfill(7)
        duration = obj.get_elapsed_time_s()
        distance = obj.get_total_distance_cm()
        avg_speed = obj.get_average_speed()
        
        line = f"{obj.type} {prog} duration={duration}s distance={distance}cm avg_speed={avg_speed}km/h\n"
        
        summary_path = os.path.join(self.config.log_root, "summary.txt")
        with open(summary_path, 'a') as f:
            f.write(line)

    def log_crossing_debug(self, obj, line_code, timestamp):
        """Log crossing debug information (optional)"""
        debug_path = os.path.join(self.config.log_root, "line_crossing_debug.txt")
        with open(debug_path, 'a') as f:
            f.write(f"Track ID: {obj.track_id}, Line: {line_code}, Time: {int(timestamp)}\n")