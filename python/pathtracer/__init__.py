"""PathTracer – light transport on trimmed NURBS geometry (C++ kernel).

Light tracing, path tracing, bidirectional path tracing and photon mapping as estimators of the
same quantities: the irradiance on faces (a texture per face) and the radiance seen by a camera.
Lengths in mm, power in W. The geometry is given as the dicts of the .nfcad format.

    import pathtracer as pt

    mesh = pt.build_mesh(entities, 0.01)
    scene = pt.Scene(mesh, materials, entity_material, sources, closed)
    sensor = pt.TexelSensor(scene, faces, 64)
    out = pt.render(scene, sensor, "bidir", samples=500_000, depth=8)
"""

from ._core import (KINDS, METHODS, Camera, Distribution, Mesh, Scene, Sensor, TexelSensor, build_mesh, light_paths,
                    render, v_lambda)

__version__ = "0.1.0"
__all__ = ["KINDS", "METHODS", "Camera", "Distribution", "Mesh", "Scene", "Sensor", "TexelSensor", "build_mesh",
           "light_paths", "render", "v_lambda"]
