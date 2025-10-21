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
        """Check if two line segments intersect using proper line segment intersection algorithm
        
        Args:
            a1, a2: Points defining the first line segment (object trajectory)
            b1, b2: Points defining the second line segment (crossing line)
        
        Returns:
            True if the line segments intersect, False otherwise
        """
        def ccw(A, B, C):
            """Check if three points are in counter-clockwise order"""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        # Two line segments intersect if the endpoints of each segment are on opposite sides of the other segment
        return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)