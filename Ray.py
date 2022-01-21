import numpy as np

from util import vec3d


class Ray:
    def __init__(self, pos:vec3d, heading:vec3d):
        self.pos = pos
        self.heading = heading/np.linalg.norm(heading)

    def at(self, k):
        return self.pos+k*self.heading
