class Config:
    def __init__(self):
        self.label_map = {
            0: 'person',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Updated for YOLOv7-tiny 416x416
        self.model_path = "models/yolov7-tiny_416x416.blob"
        
        self.pixel_to_cm_ratio = 1.0  # 1 px = 1 cm (needs calibration)
        
        self.crossing_lines = [
            {"code": "AC001", "p1": (900, 750), "p2": (1400, 750), "blink_counter": 0},
            {"code": "AC002", "p1": (200, 300), "p2": (700, 300), "blink_counter": 0}
        ]
        
        self.log_root = "logs"
        self.reset_days = 7
        self.test_video_path = "videos/people.mp4"