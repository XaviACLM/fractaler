from Camera import PointCamera
from simplices import StackedSimplices
from visible_objects import Sphere
from fractals import MarchedSphere, SphereLattice, RuggedSphere, CoordinateRuggedTerrain
from lights import PointLight, DirectionalLight
from Scene import Scene
from util import make_vec
from matplotlib import pyplot as plt
import numpy as np




if __name__=="__main__" and 0:

    camera = PointCamera(make_vec(0, 3, 0),make_vec(0, -1, -10), 1, 1)
    simplex = StackedSimplices.functional_amplitude(1, [2.5 ** i for i in range(1)], lambda x: 0.2 / x)
    simplex = StackedSimplices.functional_amplitude(3, [0.25,1,4], lambda x: 0.7 / x)
    print(simplex.amplitude)

    terrain = CoordinateRuggedTerrain(0, simplex)
    #light = PointLight(make_vec(10, 0, 0), 1)
    light = PointLight(make_vec(10, 3, 10), 100)
    #light = DirectionalLight(make_vec(-1, -1, 0), 1)
    light.big_number = 1
    #light = DirectionalLight(make_vec(-1, 0, 0), 1)

    scene = Scene(light, terrain)
    im = scene.render(camera, 80)
    plt.imshow(im)
    plt.show()

if __name__=="__main__":

    camera = PointCamera(make_vec(0, 2, 0),make_vec(0, -1, 0), 1,1)
    simplex = StackedSimplices.functional_amplitude(1, [2.5 ** i for i in range(1)], lambda x: 0.2 / x)
    simplex = StackedSimplices.functional_amplitude(3, [1,7,20], lambda x: 0.1 / x)
    #simplex = StackedSimplices.functional_amplitude(1, [4], lambda x: 0.1 / x)
    print(simplex.amplitude)

    np.random.seed(415)
    pts = np.random.uniform(-1,1,(3,1))
    def fun(x,y,z):
        x = np.asarray((x, y, z))
        ds = np.linalg.norm(pts-x[:, np.newaxis], axis=0)
        hs = np.maximum(0.1-np.abs(ds-0.8),0)
        return np.max(hs)

    #simplex.at = fun

    sphere = RuggedSphere(make_vec(0, 0, 0), 1, simplex)
    #sphere = SphereLattice(make_vec(0, 0, 0), 1, 5)
    #light = PointLight(make_vec(10, 0, 0), 1)
    light = PointLight(make_vec(20, 5, -10), 100)
    #light = DirectionalLight(make_vec(-1, -1, 0), 1)
    light.big_number = 1
    #light = DirectionalLight(make_vec(-1, 0, 0), 1)

    scene = Scene(light, sphere)
    im = scene.render(camera, 200)
    plt.imshow(im)
    plt.show()