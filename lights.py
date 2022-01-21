from util import vec3d, normalize, norm
from Ray import Ray


class Light:
    def __init__(self, strength: float):
        self.strength = strength

    def get_ray_to(self, point: vec3d) -> Ray:
        raise NotImplementedError()

    def get_intensity_at(self, point: vec3d) -> float:
        raise NotImplementedError()


class PointLight(Light):
    def __init__(self, pos: vec3d, strength: float):
        super().__init__(strength)
        self.pos = pos

    def get_ray_to(self, point: vec3d) -> Ray:
        difference = point-self.pos
        return Ray(self.pos, normalize(difference))

    def get_intensity_at(self, point: vec3d) -> float:
        distance = norm(point-self.pos)
        return self.strength/(distance*distance)


class DirectionalLight(Light):
    big_number = 1e3

    def __init__(self, direction: vec3d, strength: float):
        super().__init__(strength)
        self.direction = normalize(direction)

    def get_ray_to(self, point: vec3d) -> Ray:
        return Ray(point-self.big_number*self.direction, self.direction)

    def get_intensity_at(self, point: vec3d) -> float:
        return self.strength
