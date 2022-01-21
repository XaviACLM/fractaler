from util import vec3d
import numpy as np
from functools import cached_property

from Ray import Ray


class Camera:
    def __init__(self, pos: vec3d, heading: vec3d, horizontal_fov: float, vertical_fov: float):
        self.pos = pos
        self.heading = heading/np.linalg.norm(heading)
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov

    def screen_to_viewplane_ray(self, x: float, y: float, resolution):
        raise NotImplementedError()


class PointCamera(Camera):

    @cached_property
    def horizontal_stretch(self):
        v = np.asarray((self.heading[1], -self.heading[0], 0))
        v /= np.linalg.norm(v)
        # v *= self.horizontal_fov
        return v

    @cached_property
    def vertical_stretch(self):
        v = self.heading + np.asarray((0, 0, 1))
        v -= self.heading*np.dot(v, self.heading)
        v /= np.linalg.norm(v)
        # v *= self.vertical_fov
        return v

    def screen_to_viewplane_ray(self, x: float, y: float, resolution):
        return Ray(self.pos, self.heading+self.horizontal_stretch*x/resolution+self.vertical_stretch*y/resolution)
