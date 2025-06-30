import scipy
import numpy as np

class InvertedPendulum:
    # Point mass attached to a massless rigid rod
    
    def __init__(self, mass, len, gravity):
        self.MASS = mass
        self.LEN = len
        self.GRAVITY = gravity
        self.angular_pos = 0        # 0 is pointing straight up, positive clockwise
        self.angular_vel = 0        # Positive velocity is clockwise
    
    