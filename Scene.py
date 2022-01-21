from lights import Light
from visible_objects import VisibleObject
from util import norm
import numpy as np


class Scene:
    def __init__(self, light: Light, visible_object: VisibleObject):
        self.light = light
        self.visible_object = visible_object

    # TODO: multiobject, multilight
    """ 
    def add_light(self, light: Light):
        self.lights.append(light)
        
    def add_visible_object(self, visible_object: VisibleObject):
        self.visible_objects.append(VisibleObject)
    """

    def render(self, camera, resolution: float, lighting_tolerance = 1e-3):
        screen_width = int(camera.horizontal_fov*resolution)
        screen_height = int(camera.vertical_fov*resolution)

        im = np.zeros((screen_width*2+1,screen_height*2+1))

        for screen_x in range(-screen_width, screen_width + 1):
            print(screen_x,screen_width)
            for screen_y in range(-screen_height, screen_height + 1):
                ray = camera.screen_to_viewplane_ray(screen_x, screen_y, resolution)
                hit_info = self.visible_object.hit(ray)

                if hit_info is None:
                    continue
                    # TODO: maybe background

                light_ray = self.light.get_ray_to(hit_info.pos)
                lighting_info = self.visible_object.hit(light_ray, compute_normal=False)

                if lighting_info is None or norm(lighting_info.pos-hit_info.pos) > lighting_tolerance:
                    continue

                lighting_intensity = self.light.get_intensity_at(hit_info.pos)

                if hit_info.normal is not None:
                    light_angle = np.dot(hit_info.normal, -light_ray.heading)
                    lighting_intensity *= light_angle

                if lighting_intensity < 0:
                    continue

                # TODO: more fancier
                # intensity /= hit_info.ambient_complexity
                # intensity *= lighting_info.path_occlusion

                im[screen_x, screen_y] = lighting_intensity
        im = np.roll(im, shift=(screen_width, screen_height), axis=(0,1))

        return im.T
