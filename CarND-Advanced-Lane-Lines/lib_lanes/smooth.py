from lib_lanes.lane_detector import Polynomial

class SmoothedPolynomial:

    def __init__(self, history_size, curvature_th = 1500):
        self.history = []
        self.history_size = history_size
        self.curvature_th = curvature_th
        
    def polynomial(self, w, h, current):
        next = self.history + [current]
        if len(next) > self.history_size:
            next = next[1:]
        smooth = Polynomial.from_history(next)        
        if smooth.real_world(h, w)[0] < self.curvature_th or len(next) < self.history_size:
            self.history = next
            return smooth
        else:
            return Polynomial.from_history(self.history)
