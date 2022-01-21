from Ray import Ray
from typing import Union
from util import vec3d, normalize
import numpy as np


class HitInfo:
    def __init__(self,
                 pos: vec3d,
                 normal: Union[type(None), vec3d],
                 ambient_complexity: float):
        self.pos = pos
        self.normal = normal
        self.ambient_complexity = ambient_complexity


class VisibleObject:
    def hit(self, ray: Ray, compute_normal=True) -> Union[type(None), HitInfo]:
        raise NotImplementedError()


class Sphere(VisibleObject):
    def __init__(self, center: vec3d, radius: float):
        self.center = center
        self.radius = radius

    def hit(self, ray: Ray, compute_normal=True) -> Union[type(None), HitInfo]:
        b = 2*np.dot(ray.heading, ray.pos-self.center)
        c = np.dot(ray.pos-self.center,ray.pos-self.center)-self.radius*self.radius
        discriminant = b*b-4*c
        if discriminant < 0: return None
        k = (-b - np.sqrt(discriminant))/2
        hit_pos = ray.at(k)
        return HitInfo(hit_pos, normalize(hit_pos-self.center), 1)