from typing import Union

import numpy as np

from Ray import Ray
from simplices import StackedSimplices
from visible_objects import VisibleObject, HitInfo
from util import vec3d, normalize, norm, complete_base, make_vec


class SDFFractal(VisibleObject):
    convergence_tol = 1e-5
    max_iters = 1000
    runoff_tolerance = 100

    def hit(self, ray: Ray, compute_normal=True) -> Union[type(None), HitInfo]:
        converged = False
        k=0
        distance = 0
        qt_iters = 0
        min_dist = float("inf")
        while not converged:
            qt_iters += 1
            k += distance
            point = ray.at(k)
            distance = self.signed_distance(point)
            min_dist = min(distance, min_dist)
            converged = (distance <= self.convergence_tol)
            if qt_iters>self.max_iters or distance-min_dist>self.runoff_tolerance:
                return None
        return HitInfo(point, self.get_normal(point), qt_iters)

    def get_normal(self, point: vec3d) -> Union[type(None), vec3d]:
        return None

    def signed_distance(self, point: vec3d) -> float:
        raise NotImplementedError


class MarchedSphere(SDFFractal):
    def __init__(self, pos: vec3d, radius: float):
        self.pos = pos
        self.radius = radius

    def get_normal(self, point: vec3d) -> Union[type(None), vec3d]:
        return normalize(point-self.pos)

    def signed_distance(self, point: vec3d) -> float:
        return norm(point-self.pos)-self.radius


class RuggedSphere(SDFFractal):
    def __init__(self, pos: vec3d, radius: float, simplex: StackedSimplices):
        self.pos = pos
        self.radius = radius
        self.simplex = simplex

    def get_normal(self, point: vec3d) -> Union[type(None), vec3d]:
        sea_normal = normalize(point-self.pos)
        sea_normal, v1, v2 = complete_base([sea_normal])

        surface = sea_normal*self.radius
        tolerance = 0.000001
        epsilon = tolerance/self.simplex.highest_frequency
        pos1p = normalize(surface+epsilon*v1)*self.radius
        pos1m = normalize(surface-epsilon*v1)*self.radius
        pos2p = normalize(surface+epsilon*v2)*self.radius
        pos2m = normalize(surface-epsilon*v2)*self.radius
        height1p = self.simplex.at(*pos1p)
        height1m = self.simplex.at(*pos1m)
        height2p = self.simplex.at(*pos2p)
        height2m = self.simplex.at(*pos2m)
        pos1p *= 1+height1p/self.radius
        pos1m *= 1+height1m/self.radius
        pos2p *= 1+height2p/self.radius
        pos2m *= 1+height2m/self.radius

        d1 = pos1p-pos1m
        d2 = pos2p-pos2m
        d3 = normalize(complete_base([d1, d2])[-1])

        if np.dot(d3, sea_normal) > 0:
            return d3
        else:
            return -d3

    def signed_distance(self, point: vec3d) -> float:
        centered = point-self.pos
        elevation = norm(centered)-self.radius
        margin = elevation - self.simplex.amplitude

        if margin > self.simplex.amplitude/2:
            return margin

        surface = normalize(centered)*self.radius
        terrain_height = self.simplex.at(*surface)
        height = elevation-terrain_height

        looseness = 0
        l_const = self.simplex.lipschitz_bound
        l_const *= 1-looseness
        return height/np.sqrt(1+l_const*l_const)


class CoordinateRuggedTerrain(SDFFractal):
    def __init__(self, height: float, simplex: StackedSimplices):
        self.height = height
        self.simplex = simplex

    def get_normal(self, point: vec3d) -> Union[type(None), vec3d]:
        tolerance = 0.000001
        epsilon = tolerance/self.simplex.highest_frequency

        surface = make_vec(point[0], self.height, point[2])
        v1, v2 = make_vec(1, 0, 0), make_vec(0, 0, 1)

        pos1p = surface+epsilon*v1
        pos1m = surface-epsilon*v1
        pos2p = surface+epsilon*v2
        pos2m = surface-epsilon*v2
        height1p = self.simplex.at(*pos1p)
        height1m = self.simplex.at(*pos1m)
        height2p = self.simplex.at(*pos2p)
        height2m = self.simplex.at(*pos2m)
        pos1p[1] += height1p
        pos1m[1] += height1m
        pos2p[1] += height2p
        pos2m[1] += height2m

        d1 = pos1p-pos1m
        d2 = pos2p-pos2m
        d3 = make_vec(d1[1]*d2[2]-d1[2]*d2[1],d1[2]*d2[0]-d1[0]*d2[2],d1[0]*d2[1]-d1[1]*d2[0])#normalize(complete_base([d1, d2])[-1])

        if d3[1] > 0:
            return d3
        else:
            return -d3

    def signed_distance(self, point: vec3d) -> float:
        elevation = point[1]-self.height
        margin = elevation - self.simplex.amplitude

        if margin > self.simplex.amplitude/2:
            return margin

        terrain_height = self.simplex.at(point[0], self.height, point[2])
        height = elevation-terrain_height

        looseness = 0
        l_const = self.simplex.lipschitz_bound
        l_const *= 1-looseness
        return height/np.sqrt(1+l_const*l_const)


class SphereLattice(SDFFractal):
    def __init__(self, pos: vec3d, radius: float, box_size: float):
        self.pos = pos
        self.radius = radius
        self.box_size = box_size

    def get_normal(self, point: vec3d) -> Union[type(None), vec3d]:
        point -= self.pos
        point = (np.mod((point+self.box_size/2),self.box_size))-self.box_size/2
        return normalize(point)

    def signed_distance(self, point: vec3d) -> float:
        point -= self.pos
        point = (np.mod((point+self.box_size/2),self.box_size))-self.box_size/2
        return norm(point)-self.radius
