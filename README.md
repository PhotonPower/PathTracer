# PathTracer

Light transport on trimmed NURBS geometry, in C++ with a Python package: **light tracing, path
tracing, bidirectional path tracing and photon mapping** as estimators of the same quantities –
the irradiance on faces (a texture per face) and the radiance seen by a camera.

It is the C++ twin of the NumPy kernel of the *Path Tracer* plugin of Raytatouille Fusion
(`python/plugins/pathtracer/pt_*.py` there): the same model and the same estimators, written one
path at a time instead of arrays of paths, and run on all processor cores. The plugin's guide
describes the model; this file describes the library.

## What is in it

| Part | File | |
|---|---|---|
| NURBS | `src/core.cpp` | curves and surfaces of the `.nfcad` format, points and first derivatives |
| Mesh | `src/mesh.cpp` | every face tessellated in its surface parameters to a chordal tolerance; trimming as a mask over (u, v) tested at every hit; face and texel areas over the same mask; the exact surface normal where a ray meets a curved face (Newton) |
| BVH | `src/bvh.cpp`, header | surface area heuristic; nearest hit and any hit with a filter |
| Materials | `src/material.cpp` | absorber, diffuse (with transmission), mirror, glossy, glass, detector, metal and frosted (GGX); by wavelength: index (Abbe number), reflectance, absorption |
| Sources | `src/scene.cpp`, `src/photometry.cpp` | surfaces (cone or a measured distribution), points, curves (thin wires); line spectra; luminaire files (IES LM-63, EULUMDAT) |
| Scene | `src/scene.cpp` | closed bodies oriented by counting crossings; glass bodies that touch or lie inside one another |
| Sensors | `src/sensor.cpp` | textures on faces, pinhole and lens camera; shares of the samples per texel / pixel that adapt |
| Estimators | `src/integrator.cpp`, `src/photon.cpp` | subpaths, connections, multiple importance sampling; photon mapping; batches on threads, standard errors; the light that leaves the scene |
| Python | `bindings/module.cpp`, `python/pathtracer` | geometry as `.nfcad` dicts in, NumPy arrays out |

Lengths in mm, power in W: radiance in W/(mm² sr), irradiance = flux per texel / texel area.

## Build

Needs a C++20 compiler, CMake ≥ 3.25, Python ≥ 3.10 with NumPy and pybind11.

```bash
pip install .                      # the package pathtracer
# or, in the source tree (the module lands in python/pathtracer):
cmake -S . -B build -G Ninja -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build
python tests/test_kernel.py
```

## Use

```python
import pathtracer as pt

mesh = pt.build_mesh(entities, 0.01)             # dicts {"id", "geometry"}: surface, trimmed_surface, shell
scene = pt.Scene(mesh,
                 [{"name": "white", "kind": "diffuse", "reflectance": 0.8},
                  {"name": "glass", "kind": "glass", "ior": 1.5}],
                 entity_material=[0, 1],          # one material number per entity
                 sources=[{"name": "lamp", "flux": 1.0, "points": [(0, 0, 50)]}],
                 closed=[1])                      # entities that are closed bodies
sensor = pt.TexelSensor(scene, faces=[0], long_side=64)
out = pt.render(scene, sensor, "bidir", samples=1_000_000, depth=8, seed=1)
flux = out["values"]                              # W per texel; out["error"]: its standard error
image = pt.render(scene, pt.Camera((300, -300, 200), (0, 0, 0), (0, 0, 1), 40, 320, 240), "path",
                  samples=2_000_000, depth=8)["values"].reshape(240, 320)
```

`render(scene, sensor, method, samples, depth, seed=0, threads=0, radius=0, adaptive=False,
weights=None, batch=0, progress=None, cancelled=None)` – methods `light`, `path`, `bidir`,
`photon`; `threads` 0: all cores; the result is the same for any number of threads up to rounding.

`escapes(scene, n, depth, seed=0, index=0, threads=0)` – the light that leaves the scene (a far
field, e.g. the luminous intensity of a luminaire): n light subpaths of up to `depth` segments;
for every one that leaves, its direction `d`, its `flux` (the fluxes of one call add up to the
escaping flux), its wavelength's `channel`, its number of surface events `bounces` (0: straight
from the source) and the entities of its `first` and `last` surface (-1: none). Batches of one
seed are independent by `index`; the result does not depend on the number of threads.

## Tests

`tests/test_kernel.py` (about 10 s): the closed forms and cross-checks of the plugin's test –
point source over a plane, closed diffuse box, glass plates (also absorbing, dispersing,
touching, immersed), detectors, wires, measured distributions, cameras with and without a lens,
photon mapping through glass, microfacet sampling against its evaluation, standard errors,
adaptive sampling, threads, cancel.

## Licence and sources

MIT, see [LICENSE](LICENSE). No third-party code is included; the package is built with
[pybind11](https://github.com/pybind/pybind11) (BSD-3-Clause) and uses NumPy (BSD-3-Clause) at run
time. The methods follow the literature:

- B-spline basis functions: Piegl, Tiller, *The NURBS Book*, algorithm A2.2
- ray–triangle intersection: Möller, Trumbore (1997)
- path space, bidirectional path tracing, multiple importance sampling: Veach (1997)
- microfacet reflection and refraction through rough surfaces: Walter, Marschner, Li, Torrance (2007);
  sampling the visible normals of GGX: Heitz (2018)
- progressive photon mapping with a shrinking radius: Knaus, Zwicker (2011)
- random numbers: xoshiro256++ and SplitMix64 (Blackman, Vigna; public domain)
- photopic luminous efficiency V(λ): CIE 1924
- luminaire files: IES LM-63 and EULUMDAT, read from their format descriptions

## Not here (yet)

Sources and cameras inside glass; polarisation; coatings; volumes.
