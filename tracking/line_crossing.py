class LineCrossingDetector:
    def __init__(self, config):
        self.lines = config.crossing_lines
        self.crossed = set()
        
    def check_crossing(self, tracked_obj):
        """Check if tracked object crossed any configured line"""
        if len(tracked_obj.positions) < 2:
            return None
            
        (x1, y1), _ = tracked_obj.positions[-2]
        (x2, y2), _ = tracked_obj.positions[-1]
        
        for line in self.lines:
            p1, p2 = line["p1"], line["p2"]
            
            if self._intersects((x1, y1), (x2, y2), p1, p2):
                key = (tracked_obj.track_id, line["code"])
                if key not in self.crossed:
                    self.crossed.add(key)
                    # Don't set blink_counter here, let main.py handle it
                    return line["code"]
        return None
        
    def _intersects(self, a1, a2, b1, b2):
        """Check if two line segments intersect"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)