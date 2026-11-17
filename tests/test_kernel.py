"""The C++ kernel against closed forms and against itself across the methods (the tests of the
NumPy kernel of the Path Tracer plugin, through the Python module). Run: python tests/test_kernel.py"""

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

if not os.environ.get("PT_TEST_INSTALLED"):  # the wheel test (cibuildwheel): the installed package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import pathtracer as pt  # noqa: E402

SEED = int(os.environ.get("PT_TEST_SEED", "1"))
METHODS = ("light", "path", "bidir")


def rect(p0, du, dv):
    """A flat face: corner and two edges; the normal is du x dv."""
    p0, du, dv = (np.asarray(v, dtype=float) for v in (p0, du, dv))
    pts = [p0, p0 + dv, p0 + du, p0 + du + dv]                       # index = i_u * num_v + i_v
    return {"type": "surface", "degree_u": 1, "degree_v": 1, "knots_u": [0, 0, 1, 1], "knots_v": [0, 0, 1, 1],
            "num_u": 2, "num_v": 2, "points": [float(c) for p in pts for c in p]}


def box(lo, hi):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    ex, ey, ez = np.diag(hi - lo)
    faces = [rect(lo, ey, ex), rect(lo + ez, ex, ey), rect(lo, ex, ez), rect(lo + ey, ez, ex),
             rect(lo, ez, ey), rect(lo + ex, ey, ez)]                # normals outwards
    return {"type": "shell", "faces": [{"surface": f} for f in faces]}


def polyline(points):
    n = len(points)
    return {"degree": 1, "knots": [0.0] + [i / (n - 1) for i in range(n)] + [1.0],
            "points": [float(c) for p in points for c in (p[0], p[1], 0.0)]}


def sphere_patch(radius):
    """A quarter circle (rational, degree 2) revolved by a quarter turn: an eighth of a sphere."""
    w = math.sqrt(0.5)
    arc = [(1, 0), (1, 1), (0, 1)]
    pts, weights = [], []
    for (cx, cz), wu in zip(arc, (1, w, 1)):
        for (ax, ay), wv in zip(arc, (1, w, 1)):
            pts.append((radius * cx * ax, radius * cx * ay, radius * cz))
            weights.append(wu * wv)
    return {"type": "surface", "degree_u": 2, "degree_v": 2, "knots_u": [0, 0, 0, 1, 1, 1],
            "knots_v": [0, 0, 0, 1, 1, 1], "num_u": 3, "num_v": 3, "points": [float(c) for p in pts for c in p],
            "weights": weights}


MATERIALS = [
    {"name": "black", "kind": "absorber"},
    {"name": "white", "kind": "diffuse", "reflectance": 0.6},
    {"name": "mirror", "kind": "mirror", "reflectance": 0.9},
    {"name": "glass", "kind": "glass", "ior": 1.5},
    {"name": "glossy", "kind": "glossy", "reflectance": 0.7, "roughness": 0.3},
    {"name": "tinted", "kind": "glass", "ior": 1.5, "absorption": 0.05},
    {"name": "sheet", "kind": "diffuse", "reflectance": 0.3, "transmittance": 0.6},
    {"name": "detector", "kind": "detector"},
    {"name": "metal", "kind": "metal", "reflectance": 0.9, "roughness": 0.5},
    {"name": "frosted", "kind": "frosted", "ior": 1.5, "roughness": 0.5},
    {"name": "fine", "kind": "frosted", "ior": 1.5, "roughness": 0.02},
    {"name": "flint", "kind": "glass", "ior": 1.7},
    {"name": "water", "kind": "glass", "ior": 1.33},
    {"name": "dyed", "kind": "glass", "ior": 1.33, "absorption": 0.01},
]
BLACK, WHITE, MIRROR, GLASS, GLOSSY, TINTED, SHEET, DETECTOR, METAL, FROSTED, FINE, FLINT, WATER, DYED = range(14)


def scene_of(geometries, materials, sources, closed=(), tol=1e-3, rows=MATERIALS):
    mesh = pt.build_mesh([{"id": k + 1, "geometry": g} for k, g in enumerate(geometries)], tol)
    return pt.Scene(mesh, rows, list(materials), sources, list(closed))


def values(scene, sensor, method, samples, depth=6, **more):
    return pt.render(scene, sensor, method, samples, depth, seed=SEED, **more)["values"]


def solid_angle(a, b, height):
    """Of the rectangle |x| < a, |y| < b seen from the point at `height` above its centre."""
    return 4 * math.atan(a * b / (height * math.sqrt(a * a + b * b + height * height)))


def point(position, flux=1.0, **more):
    return {"name": "p", "flux": flux, "points": [position], **more}


started = time.time()

# ---- mesh: a sphere patch within the tolerance, its area; exact normals at hits
patch = sphere_patch(2.0)
mesh = pt.build_mesh([{"id": 7, "geometry": patch}], 1e-3)
corners = np.concatenate([mesh.v0, mesh.v0 + mesh.e1, mesh.v0 + mesh.e2])
centres = mesh.v0 + (mesh.e1 + mesh.e2) / 3
assert np.allclose(np.linalg.norm(corners, axis=1), 2.0)
assert 0 < 2.0 - np.linalg.norm(centres, axis=1).min() < 2e-3
assert abs(mesh.face_area[0] - 4 * math.pi * 4 / 8) < 0.01 and mesh.entity_ids == [7] and mesh.curved == [True]
coarse = scene_of([sphere_patch(20.0)], [MIRROR], [point((40, 30, 35))], tol=0.5)
assert coarse.mesh.triangles < 200
rng = np.random.default_rng(4)
worst = []
for exact in (True, False):
    coarse.exact = exact
    off = []
    for _ in range(300):
        target = rng.normal(size=3)
        target = np.abs(target) / np.linalg.norm(target) * 20
        start_ = np.array([40.0, 30.0, 35.0])
        aim = (target - start_) / np.linalg.norm(target - start_)
        t, tri, b1, b2 = coarse.hit(start_, aim)
        if tri < 0:
            continue
        shade = coarse.mesh.shade_exact if exact else (lambda tr, c1, c2, o, d, tt: coarse.mesh.shade(tr, c1, c2))
        n, u, v = shade(tri, b1, b2, start_, aim, t)
        along = start_ @ aim
        meets = start_ + (-along - math.sqrt(along ** 2 - (start_ @ start_ - 400))) * aim
        off.append(abs(abs(np.dot(n, meets / 20)) - 1))
    worst.append(np.median(off))
assert worst[0] < 1e-12 < 1e-6 < worst[1], worst

# ---- trimming: a disc with a square hole on a 100 mm square
angle = np.linspace(0, 2 * math.pi, 257)
disc = {"type": "trimmed_surface", "surface": rect((0, 0, 0), (100, 0, 0), (0, 100, 0)),
        "outer": [polyline(np.column_stack([0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)]))],
        "holes": [[polyline([(0.45, 0.45), (0.55, 0.45), (0.55, 0.55), (0.45, 0.55), (0.45, 0.45)])]]}
mesh = pt.build_mesh([{"id": 1, "geometry": disc}], 1e-3)
assert abs(mesh.face_area[0] / (math.pi * 40 ** 2 - 100) - 1) < 2e-3 and mesh.mask_res == [512]
texel = mesh.texel_areas(0, 20, 20)
assert abs(texel.sum() - mesh.face_area[0]) < 1e-6 and texel[0, 0] == 0 and texel[10, 10] == 0
assert abs(texel[5, 10] - 25) < 1e-9 and 0 < texel[10, 1] < 25
scene = scene_of([disc], [BLACK], [point((50, 50, 40))])
sensor = pt.TexelSensor(scene, [0], 20)
tri, b1, b2, chosen = sensor.points(20_000, SEED)
assert (tri >= 0).mean() > 0.999
ok = tri >= 0
p = scene.mesh.v0[tri[ok]] + b1[ok, None] * scene.mesh.e1[tri[ok]] + b2[ok, None] * scene.mesh.e2[tri[ok]]
r = np.hypot(p[:, 0] - 50, p[:, 1] - 50)
assert r.max() < 40.3 and not ((abs(p[:, 0] - 50) < 4.8) & (abs(p[:, 1] - 50) < 4.8)).any()
assert abs(r.mean() / 27.13 - 1) < 0.01, r.mean()
counts = np.bincount(chosen[ok], minlength=sensor.count)
asked = 20_000 * sensor.texel_area / sensor.area
assert np.abs(counts - asked).max() < 3 and np.abs(counts - asked).mean() < 0.4

# ---- a point source 50 mm above a black plane: the flux per texel from the solid angles
plane = rect((-100, -100, 0), (200, 0, 0), (0, 200, 0))
scene = scene_of([plane], [BLACK], [point((0, 0, 50))])
sensor = pt.TexelSensor(scene, [0], 10)
assert np.allclose(sensor.texel_area, 400)
for method in METHODS:
    flux = values(scene, sensor, method, 200_000)
    assert abs(flux.sum() / (solid_angle(100, 100, 50) / (4 * math.pi)) - 1) < 0.01, (method, flux.sum())
    centre = flux.reshape(10, 10)[4:6, 4:6].sum() / (solid_angle(20, 20, 50) / (4 * math.pi))
    assert abs(centre - 1) < 0.04, (method, centre)

# ---- a closed diffuse box with a point source inside: 1 + rho + … + rho^11
scene = scene_of([box((0, 0, 0), (100, 100, 100))], [WHITE], [point((50, 50, 50))], (0,))
assert scene.flipped == []
sensor = pt.TexelSensor(scene, list(range(6)), 2)
for method in METHODS:
    total = values(scene, sensor, method, 100_000, 12).sum()
    assert abs(total / ((1 - 0.6 ** 12) / 0.4) - 1) < 0.015, (method, total)

# ---- closed bodies are oriented
turned = box((0, 0, 0), (10, 20, 30))
for k in (1, 4):
    surface = turned["faces"][k]["surface"]
    surface["points"] = [float(c) for c in np.asarray(surface["points"]).reshape(2, 2, 3).transpose(1, 0, 2).ravel()]
scene = scene_of([turned], [WHITE], [point((5, 5, 5))], (0,))
assert scene.flipped == [1, 4]
outward = scene.mesh.v0 + (scene.mesh.e1 + scene.mesh.e2) / 3 - [5, 10, 15]
assert (np.einsum("ij,ij->i", outward, scene.mesh.ng) > 0).all()

# ---- a glass plate in a narrow beam: T = (1 − R) / (1 + R); and absorbing
beam = {"name": "beam", "flux": 1.0, "half_angle": 0.5, "side": "front", "entities": [1]}
plate = [box((-50, -50, 0), (50, 50, 10)), rect((-1, -1, -20), (2, 0, 0), (0, 2, 0)),
         rect((-40, -40, 30), (80, 0, 0), (0, 80, 0))]
scene = scene_of(plate, [GLASS, BLACK, BLACK], [beam], (0,))
sensor = pt.TexelSensor(scene, [7], 4)
for method in ("light", "bidir"):
    total = values(scene, sensor, method, 100_000, 30).sum()
    assert abs(total - 0.96 / 1.04) < 0.005, (method, total)
scene = scene_of(plate, [TINTED, BLACK, BLACK], [beam], (0,))
inner = math.exp(-0.5)
total = values(scene, pt.TexelSensor(scene, [7], 4), "light", 100_000, 30).sum()
assert abs(total - 0.96 ** 2 * inner / (1 - 0.04 ** 2 * inner ** 2)) < 0.005, total

# ---- detectors: the light passes them
floor = rect((-100, -100, 0), (200, 0, 0), (0, 200, 0))
upper, lower = rect((-25, -25, 25), (50, 0, 0), (0, 50, 0)), rect((-10, -10, 10), (20, 0, 0), (0, 20, 0))
source = [point((0, 0, 50))]
scene = scene_of([floor, upper, lower], [BLACK, DETECTOR, DETECTOR], source)
assert scene.has_detectors and scene.hit((0, 0, 50), (0, 0, -1))[0] == 50
sensor = pt.TexelSensor(scene, [0, 1, 2], 2)
assert sensor.crossed
expected = np.array([solid_angle(100, 100, 50), solid_angle(25, 25, 25), solid_angle(10, 10, 40)])
for method in METHODS:
    flux = values(scene, sensor, method, 400_000).reshape(3, 4).sum(axis=1) * 4 * math.pi
    assert (np.abs(flux / expected - 1) < [0.02, 0.02, 0.06]).all(), (method, flux / expected)
scene = scene_of([floor, upper], [WHITE, DETECTOR], source)          # the floor reflects: more passes
sensor = pt.TexelSensor(scene, [1], 2)
reference = values(scene, sensor, "light", 800_000).sum()
assert reference * 4 * math.pi > 1.1 * expected[1]
for method in ("path", "bidir"):
    got = values(scene, sensor, method, 400_000).sum()
    assert abs(got / reference - 1) < 0.02, (method, got, reference)
only = scene_of([upper], [DETECTOR], source)
flux = values(only, pt.TexelSensor(only, [0], 2), "light", 200_000).sum() * 4 * math.pi
assert abs(flux / expected[1] - 1) < 0.02
for bad, why in ((lambda: scene_of([floor, upper], [BLACK, DETECTOR], [{"name": "a", "entities": [1]}]), "detector cannot be a source"),
                 (lambda: scene_of([floor], [BLACK], []), "no source"),
                 (lambda: scene_of([floor], [BLACK], [{"name": "m", "points": [(0, 0, 1)], "curves": [[(0, 0, 0), (0, 0, 1)]]}]),
                  "different kinds"),
                 (lambda: scene_of([floor], [0], source, rows=[{"name": "x", "kind": "wood"}]), "unknown kind")):
    try:
        bad()
        raise AssertionError("not refused")
    except ValueError as e:
        assert why in str(e), e

# ---- a diffusing sheet between a lamp and the floor of a closed box; glossy plate and mirror wall
lamp = {"name": "a", "flux": 1.0, "half_angle": 90, "side": "front", "entities": [2]}
scene = scene_of([box((0, 0, 0), (100, 100, 100)), rect((0.5, 0.5, 50), (99, 0, 0), (0, 99, 0)),
                  rect((40, 40, 80), (0, 20, 0), (20, 0, 0))], [WHITE, SHEET, BLACK], [lamp], (0,))
sensor = pt.TexelSensor(scene, [0, 6], 2)
reference = values(scene, sensor, "light", 400_000, 6)
assert 0.2 < reference[:4].sum() < reference[4:].sum()
for method in ("path", "bidir"):
    got = values(scene, sensor, method, 200_000, 6)
    assert abs(got[:4].sum() / reference[:4].sum() - 1) < 0.04 and abs(got[4:].sum() / reference[4:].sum() - 1) < 0.04, method
scene = scene_of([rect((-100, -100, 0), (200, 0, 0), (0, 200, 0)), rect((-100, -100, 0), (0, 200, 0), (0, 0, 100)),
                  rect((100, -100, 0), (0, 0, 100), (0, 200, 0)), rect((-20, -20, 90), (40, 0, 0), (0, 40, 0))],
                 [WHITE, MIRROR, GLOSSY, BLACK], [{"name": "a", "flux": 1.0, "half_angle": 60, "side": "back", "entities": [3]}])
sensor = pt.TexelSensor(scene, [0, 2], 3)
reference = values(scene, sensor, "light", 1_000_000)
for method in ("path", "bidir"):
    got = values(scene, sensor, method, 400_000)
    assert abs(got.sum() / reference.sum() - 1) < 0.01 and np.allclose(got / reference, 1, atol=0.1), (method, got / reference)

# ---- cameras: a Lambertian emitter (its radiance), a lit wall, left and right, a lens
emitter = scene_of([rect((-100, -100, 0), (200, 0, 0), (0, 200, 0))], [BLACK],
                   [{"name": "a", "flux": 1.0, "entities": [0]}])
for lens, focus in ((0.0, 0.0), (5.0, 0.0), (5.0, 30.0)):
    camera = pt.Camera((0, 0, 100), (0, 0, 0), (0, 1, 0), 20.0, 4, 4, lens, focus)
    assert camera.pinhole == (lens == 0)
    for method in METHODS + ("photon",):
        image = values(emitter, camera, method, 200_000, 3)
        assert abs(image.mean() * math.pi * 40_000 - 1) < (0.05 if method == "light" else 0.01), (lens, method, image.mean())
assert values(emitter, pt.Camera((0, 0, -100), (0, 0, 0), (0, 1, 0), 5.0, 4, 4), "path", 2_000, 3).max() == 0
wall = rect((-100, -100, 0), (200, 0, 0), (0, 200, 0))
lit_wall = scene_of([wall], [WHITE], [point((0, 0, 50))])
camera = pt.Camera((0, 0, 200), (0, 0, 0), (0, 1, 0), 2.0, 4, 4)
exact = 0.6 / math.pi / (4 * math.pi * 50 ** 2)
for method in ("path", "bidir"):
    assert abs(values(lit_wall, camera, method, 100_000, 2).mean() / exact - 1) < 0.01, method
out = pt.render(lit_wall, camera, "photon", 400_000, 2, seed=SEED)
assert abs(out["values"].mean() / exact - 1) < 0.12 and out["passes"] == 16 and 0.5 < out["radius"] < 20, out["values"].mean() / exact
wide = pt.Camera((0, 0, 200), (0, 0, 0), (0, 1, 0), 40.0, 4, 4)
reference = values(lit_wall, wide, "path", 100_000, 2)
for method in ("light", "bidir"):
    assert abs(values(lit_wall, wide, method, 400_000, 2).sum() / reference.sum() - 1) < 0.03, method
side_lit = scene_of([wall], [WHITE], [point((60, 0, 20))])              # light at +x
image = values(side_lit, pt.Camera((0, 0, 300), (0, 0, 0), (0, 1, 0), 30.0, 2, 2), "path", 100_000, 2).reshape(2, 2)
assert image[:, 1].sum() > 3 * image[:, 0].sum() and abs(image[0].sum() / image[1].sum() - 1) < 0.1
star = scene_of([rect((-100, -100, -50), (200, 0, 0), (0, 200, 0))], [BLACK], [point((0.5, 0.5, 0))])
shares = []
for focus in (100.0, 50.0):
    lens = pt.Camera((0, 0, 100), (0, 0, 0), (0, 1, 0), 10.0, 17, 17, 4.0, focus)
    image = values(star, lens, "light", 200_000, 1)
    assert abs(image.sum() * lens.film / lens.count * 4 * math.pi * 1e4 - 1) < 0.02, image.sum()   # E at the lens: I / d²
    assert abs(values(star, lens, "bidir", 100_000, 1).sum() / image.sum() - 1) < 0.02
    assert values(star, lens, "path", 2_000, 1).max() == 0
    shares.append(image.max() / image.sum())
assert shares[0] > 0.9 and shares[1] < 0.1, shares
assert values(star, pt.Camera((0, 0, 100), (0, 0, 0), (0, 1, 0), 10.0, 17, 17), "light", 2_000, 1).max() == 0

# ---- photon mapping: a glass plate between the source and the floor, the camera looks through it
scene = scene_of([wall, box((-200, -200, 20), (200, 200, 30))], [WHITE, GLASS], [point((0, 0, 50))], (1,))
camera = pt.Camera((0, 0, 45), (0, 0, 0), (0, 1, 0), 6.0, 4, 4)
through = 0.96 / 1.04
expected = 0.6 / math.pi * through / (4 * math.pi * (50 - 10 / 3) ** 2) * through
image = values(scene, camera, "photon", 400_000, 8)
assert abs(image.mean() / expected - 1) < 0.08, image.mean() / expected
for method in METHODS:
    assert values(scene, camera, method, 4_000, 8).max() == 0, method
scene = scene_of([wall, rect((-100, 100, 0), (200, 0, 0), (0, 0, 150))], [WHITE, MIRROR], [point((0, 0, 50))])
camera = pt.Camera((0, -50, 80), (0, 100, 60), (0, 0, 1), 20.0, 4, 4)
reference = values(scene, camera, "path", 200_000, 6).mean()
assert abs(values(scene, camera, "photon", 400_000, 6).mean() / reference - 1) < 0.05
sensor = pt.TexelSensor(lit_wall, [0], 4)
assert np.array_equal(values(lit_wall, sensor, "photon", 5_000), values(lit_wall, sensor, "light", 5_000))

# ---- a curve as a source: a thin glowing wire
wire = {"name": "wire", "flux": 1.0, "curves": [[(0, 0, -5), (0, 0, 0), (0, 0, 5)]]}
side, top = rect((50, -10, -10), (0, 20, 0), (0, 0, 20)), rect((-10, -10, 50), (20, 0, 0), (0, 20, 0))
scene = scene_of([box((-100, -100, -100), (100, 100, 100)), side, top], [BLACK, DETECTOR, DETECTOR], [wire], (0,))
assert abs(values(scene, pt.TexelSensor(scene, list(range(6)), 1), "light", 100_000, 2).sum() - 1) < 0.01
grid = (np.arange(20) + 0.5) - 10
along = (np.arange(40) + 0.5) / 4 - 5
expected = []
for at, normal in (((50.0, grid[:, None, None], grid[None, :, None]), (1, 0, 0)),
                   ((grid[:, None, None], grid[None, :, None], 50.0), (0, 0, 1))):
    dx, dy, dz = at[0] - 0.0, at[1] - 0.0, at[2] - along[None, None, :]
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    expected.append(float((np.sqrt(dx ** 2 + dy ** 2) / dist * np.abs(dx * normal[0] + dz * normal[2]) / dist
                           / dist ** 2 / math.pi ** 2).mean() * 400))
sensor = pt.TexelSensor(scene, [6, 7], 1)
for method, samples in (("light", 2_000_000), ("path", 100_000), ("bidir", 100_000)):
    flux = values(scene, sensor, method, samples, 2)
    assert abs(flux[0] / expected[0] - 1) < 0.03 and abs(flux[1] / expected[1] - 1) < 0.08, (method, flux, expected)

# ---- wavelengths: dispersion (Snell per line), two lamps of different colour, coloured materials
angle = math.radians(60)
way = np.array([math.sin(angle), 0.0, -math.cos(angle)])
start = np.array([-60.0, 0.0, 30.0])
across = np.array([-way[2], 0.0, way[0]])
strong = [{"name": "flint", "kind": "glass", "ior": 1.5, "abbe": 10.0}, {"name": "black", "kind": "absorber"},
          {"name": "detector", "kind": "detector"}]
scene = scene_of([box((-300, -300, 0), (300, 300, 20)), rect(start - np.array([0.0, 1.0, 0.0]) - across, (0, 2, 0), 2 * across),
                  rect((-40, -10, -10), (80, 0, 0), (0, 20, 0))], [0, 1, 2],
                 [{"name": "beam", "flux": 1.0, "half_angle": 0.5, "entities": [1], "spectrum": [(450, 1), (650, 3)]}],
                 (0,), rows=strong)
assert list(scene.wavelengths) == [450, 650] and np.allclose(scene.channel_prob, [0.25, 0.75])
index = [scene.ior(0, 0), scene.ior(0, 1)]
assert index[0] > 1.5 > index[1] and abs((index[0] - index[1]) - 0.5 / 10 * (1 / 0.45 ** 2 - 1 / 0.65 ** 2)
                                         / (1 / 0.48613 ** 2 - 1 / 0.65627 ** 2)) < 1e-12
sensor = pt.TexelSensor(scene, [7], 160)
out = pt.render(scene, sensor, "light", 200_000, 3, seed=SEED)
x = -40 + (np.arange(160) + 0.5) * 0.5
straight = start[0] + (start[2] + 10) * math.tan(angle)
for k in range(2):
    per_x = out["channels"][k].reshape(40, 160).sum(axis=0)
    expected = straight - 20 * (math.tan(angle) - math.sin(angle) / math.sqrt(index[k] ** 2 - math.sin(angle) ** 2))
    assert abs((per_x * x).sum() / per_x.sum() - expected) < 0.05, (k, (per_x * x).sum() / per_x.sum(), expected)
assert abs(out["channels"][1].sum() / out["channels"][0].sum() - 3) < 0.25 and np.allclose(out["channels"].sum(axis=0), out["values"])
lamps = [rect((-60, -10, 80), (0, 20, 0), (20, 0, 0)), rect((40, -10, 80), (0, 20, 0), (20, 0, 0))]
scene = scene_of([wall] + lamps, [WHITE, BLACK, BLACK],
                 [{"name": "blue", "flux": 1.0, "entities": [1], "spectrum": [(450, 1)]},
                  {"name": "red", "flux": 3.0, "entities": [2], "spectrum": [(650, 2), (450, 1)]}])
sensor = pt.TexelSensor(scene, [0], 2)
light = pt.render(scene, sensor, "light", 800_000, 4, seed=SEED)["channels"]
assert light[1].reshape(2, 2)[:, 1].sum() > 1.5 * light[1].reshape(2, 2)[:, 0].sum()
assert abs(light[0].sum() / light[1].sum() - 1) < 0.02
for method in ("path", "bidir"):
    got = pt.render(scene, sensor, method, 400_000, 4, seed=SEED)["channels"]
    assert np.allclose(got.reshape(2, 2, 2).sum(axis=1) / light.reshape(2, 2, 2).sum(axis=1), 1, atol=0.06), method
camera = pt.Camera((0, 0, 20), (0, 0, 80), (0, 1, 0), 120.0, 2, 1)
for method in ("path", "bidir"):
    image = pt.render(scene, camera, method, 200_000, 2, seed=SEED)["channels"]
    assert image[1, 1] == 0 and image[1, 0] > 0
    assert abs(image[1, 0] / image[0, 0] - 2) < 0.3 and abs(image[0, 1] / image[0, 0] - 1) < 0.3
paints = [{"name": "paint", "kind": "diffuse", "spectrum": [(650, 0.6), (450, 0.2)]},
          {"name": "tinted", "kind": "glass", "ior": 1.5, "spectrum": [(450, 0.05), (650, 0.0)]},
          {"name": "black", "kind": "absorber"}]
lines = point((50, 50, 50), spectrum=[(450, 1), (650, 1)])
scene = scene_of([box((0, 0, 0), (100, 100, 100))], [0], [lines], (0,), rows=paints)
assert abs(scene.reflectance(0, 0) - 0.2) < 1e-12 and abs(scene.reflectance(0, 1) - 0.6) < 1e-12
sensor = pt.TexelSensor(scene, list(range(6)), 2)
series = [0.5 * (1 - rho ** 12) / (1 - rho) for rho in (0.2, 0.6)]
for method in METHODS:
    got = pt.render(scene, sensor, method, 200_000, 12, seed=SEED)["channels"].sum(axis=1)
    assert np.allclose(got / series, 1, atol=0.015), (method, got / series)
scene = scene_of(plate, [1, 2, 2], [{**beam, "spectrum": [(450, 1), (650, 1)]}], (0,), rows=paints)
got = pt.render(scene, pt.TexelSensor(scene, [7], 2), "light", 200_000, 30, seed=SEED)["channels"].sum(axis=1)
assert np.allclose(got * 2, [0.96 ** 2 * inner / (1 - 0.04 ** 2 * inner ** 2), 0.96 / 1.04], atol=0.008), got * 2

# ---- microfacets: what sample draws is what eval says, also seen from the other end
probe = scene_of([wall], [WHITE], [point((0, 0, 50))])
up_ = (0.0, 0.0, 1.0)
for kind in (METAL, FROSTED):
    for towards in ((0.3, 0.0, 0.954), (0.8, 0.0, 0.6), (0.6, 0.0, -0.8)):
        out_ = np.asarray(towards) / np.linalg.norm(towards)
        for radiance in (True, False):
            wi, weight, dens = probe.sample_material(kind, up_, out_, 20_000, SEED, radiance)
            ok = weight > 0
            f, forward, backward = probe.eval_material(kind, up_, out_, wi[ok], radiance)
            assert ok.mean() > 0.9 and np.allclose(forward, dens[ok], rtol=1e-9)
            assert np.allclose(f * np.abs(wi[ok, 2]) / dens[ok], weight[ok], rtol=1e-9)
            for i in np.nonzero(ok)[0][:200]:                        # the same pair from the other end
                f2, forward2, _ = probe.eval_material(kind, up_, wi[i], out_[None, :], not radiance)
                f1, _, backward1 = probe.eval_material(kind, up_, out_, wi[i][None, :], radiance)
                assert np.allclose(f2, f1, rtol=1e-8) and np.allclose(forward2, backward1, rtol=1e-8)
            if not radiance:
                assert 0.7 < weight.mean() <= 1.0 + 1e-9, (kind, towards, weight.mean())
lamp_face = rect((-10, -10, 80), (0, 20, 0), (20, 0, 0))
frost = scene_of([wall, lamp_face, box((-60, -60, 30), (60, 60, 36)), rect((100, -100, 0), (0, 0, 100), (0, 200, 0))],
                 [WHITE, BLACK, FROSTED, METAL], [{"name": "a", "flux": 1.0, "entities": [1]}], (2,))
sensor = pt.TexelSensor(frost, [0], 2)
reference = values(frost, sensor, "light", 1_000_000, 6)
for method in ("path", "bidir"):
    got = values(frost, sensor, method, 400_000, 6)
    loose = 0.12 if method == "path" else 0.03                       # (path tracing is noisy through frosted glass)
    assert abs(got.sum() / reference.sum() - 1) < loose / 2 and np.allclose(got / reference, 1, atol=loose), (method, got / reference)
fine = scene_of(plate, [FINE, BLACK, BLACK], [beam], (0,))
total = values(fine, pt.TexelSensor(fine, [7], 2), "light", 200_000, 30).sum()
assert abs(total - 0.96 / 1.04) < 0.008, total

# ---- glass at glass: 1 / T − 1 = sum of R / (1 − R) over the surfaces
def through_plates(bodies, kinds, contact=0.02, method="light", samples=200_000):
    parts = [box((-w, -w, z0), (w, w, z1)) for z0, z1, w in bodies]
    parts += [rect((-1, -1, -20), (2, 0, 0), (0, 2, 0)), rect((-40, -40, 80), (80, 0, 0), (0, 80, 0))]
    scene_ = scene_of(parts, list(kinds) + [BLACK, BLACK],
                      [{"name": "beam", "flux": 1.0, "half_angle": 0.5, "entities": [len(bodies)]}], range(len(bodies)))
    scene_.contact = contact
    return values(scene_, pt.TexelSensor(scene_, [6 * len(bodies) + 1], 2), method, samples, 40).sum()


def all_orders(*reflectances):
    return 1 / (1 + sum(r / (1 - r) for r in reflectances))


def fresnel0(n1, n2):
    return ((n1 - n2) / (n1 + n2)) ** 2


one = all_orders(fresnel0(1, 1.5), fresnel0(1.5, 1))
assert abs(through_plates([(0, 10, 50), (10, 20, 50)], [GLASS, GLASS]) - one) < 0.005
assert abs(through_plates([(0, 10, 50), (10, 20, 50)], [GLASS, GLASS], method="bidir", samples=50_000) - one) < 0.01
doublet = all_orders(fresnel0(1, 1.5), fresnel0(1.5, 1.7), fresnel0(1.7, 1))
for second in (10.0, 9.996, 10.004):
    assert abs(through_plates([(0, 10, 50), (second, 20, 50)], [GLASS, FLINT]) - doublet) < 0.005, second
    assert abs(through_plates([(second, 20, 50), (0, 10, 50)], [FLINT, GLASS]) - doublet) < 0.005, second
apart = all_orders(fresnel0(1, 1.5), fresnel0(1.5, 1), fresnel0(1, 1.7), fresnel0(1.7, 1))
assert abs(through_plates([(0, 10, 50), (11, 21, 50)], [GLASS, FLINT]) - apart) < 0.005 and doublet - apart > 0.03
immersed = all_orders(fresnel0(1, 1.33), fresnel0(1.33, 1.5), fresnel0(1.5, 1.33), fresnel0(1.33, 1))
assert abs(through_plates([(0, 30, 50), (10, 20, 30)], [WATER, GLASS]) - immersed) < 0.005
assert abs(through_plates([(10, 20, 30), (0, 30, 50)], [GLASS, WATER]) - immersed) < 0.005
dyed = through_plates([(0, 30, 50), (10, 20, 30)], [DYED, GLASS])
assert abs(dyed / (immersed * math.exp(-0.2)) - 1) < 0.01, dyed

# ---- measured distributions: a Lambertian table on a point (closed form) and on a surface (the cone)
gammas = np.arange(0, 91, 5.0)
lambert = np.cos(np.radians(gammas))
around = pt.Distribution(gammas, [0.0], lambert[None, :])
assert around.symmetric and abs(around.eval(1.0, 0.3) * math.pi - 1) < 0.01 and around.eval(-0.5, 0.0) == 0
quarter = pt.Distribution(gammas, [0, 90, 180, 270], np.array([lambert, 0.2 * lambert, lambert, 0.2 * lambert]))
shape = np.array([quarter.eval(math.cos(math.radians(20)), math.radians(a)) for a in (1, 45, 91, 135, 181, 271)])
assert np.allclose(shape / shape[0], [1, 0.6, 0.2, 0.6, 1, 0.2], atol=0.03), shape / shape[0]
scene = scene_of([wall], [BLACK], [point((0, 0, 50), distribution=around, axis=(0, 0, -2))])
sensor = pt.TexelSensor(scene, [0], 4)
gx, gy = np.meshgrid((np.arange(40) + 0.5) * 5 - 100, (np.arange(40) + 0.5) * 5 - 100)
dist2 = gx ** 2 + gy ** 2 + 2500
by_form = (2500 / dist2 / (math.pi * dist2) * 25).reshape(4, 10, 4, 10).sum(axis=(1, 3))
for method in METHODS:
    got = values(scene, sensor, method, 400_000, 2).reshape(4, 4)
    assert abs(got.sum() / by_form.sum() - 1) < 0.01 and np.allclose(got / by_form, 1, atol=0.04), (method, got / by_form)
tilted = scene_of([wall], [BLACK], [point((0, 0, 50), distribution=around, axis=(1, 0, -1))])
got = values(tilted, pt.TexelSensor(tilted, [0], 2), "light", 100_000, 2).reshape(2, 2)
assert got[:, 1].sum() > 2 * got[:, 0].sum()
planes = scene_of([wall], [BLACK], [point((0, 0, 50), distribution=quarter)])
got = values(planes, pt.TexelSensor(planes, [0], 9), "light", 400_000, 2).reshape(9, 9)
assert got[4, 7] + got[4, 1] > 2.5 * (got[7, 4] + got[1, 4])
by_cone = scene_of([wall, lamp_face], [WHITE, BLACK], [{"name": "a", "flux": 1.0, "entities": [1]}])
by_file = scene_of([wall, lamp_face], [WHITE, BLACK], [{"name": "a", "flux": 1.0, "entities": [1], "distribution": around}])
reference = values(by_cone, pt.TexelSensor(by_cone, [0], 3), "light", 800_000, 3)
for method in METHODS:
    got = values(by_file, pt.TexelSensor(by_file, [0], 3), method, 400_000, 3)
    assert abs(got.sum() / reference.sum() - 1) < 0.01 and np.allclose(got / reference, 1, atol=0.04), (method, got / reference)
assert abs(values(by_file, pt.Camera((0, 0, 20), (0, 0, 80), (0, 1, 0), 10.0, 2, 2), "path", 40_000, 2).mean() * math.pi * 400 - 1) < 0.01
assert pt.v_lambda(555) == 1 and pt.v_lambda(900) == 0 and abs(pt.v_lambda(505) - 0.413) < 1e-9

# ---- luminaire files: IES with one and with two planes, and the same as EULUMDAT in every symmetry
def ies(planes, rows):
    head = ["IESNA:LM-63-2002", "[TEST] made for this test", "TILT=NONE",
            f"1 1000 1 {len(gammas)} {len(planes)} 1 2 0 0 0", "1 1 100"]
    return "\n".join(head + [" ".join(f"{v:g}" for v in gammas), " ".join(f"{v:g}" for v in planes)]
                     + [" ".join(f"{v:.6g}" for v in row) for row in rows])


def ldt(symmetry, planes, rows):
    step = 360 // planes
    lines_ = ["maker", "1", str(symmetry), str(planes), str(step), str(len(gammas)), "5", "report", "name", "number", "file",
              "date", "100", "100", "50", "100", "100", "0", "0", "0", "0", "100", "100", "1", "0", "1",
              "1", "lamp", "1000", "3000", "80", "10"] + ["1"] * 10
    lines_ += [str(k * step) for k in range(planes)] + [f"{v:g}" for v in gammas]
    return "\r\n".join(lines_ + [f"{v:.6g}".replace(".", ",") for row in rows for v in row])    # decimal commas


from_ies = pt.Distribution(*pt.table_ies(ies([0], [1000 * lambert])))
assert from_ies.symmetric and np.allclose(from_ies.density, around.density)
theta_, phi_, cd_ = pt.table_ies(ies([0, 90], [1000 * lambert, 200 * lambert]))
assert list(phi_) == [0, 90, 180, 270] and np.allclose(cd_[2], cd_[0]) and np.allclose(cd_[3], cd_[1])
two = pt.Distribution(theta_, phi_, cd_)
assert np.allclose(two.density, quarter.density)
for symmetry, planes_, rows in ((1, 24, [1000 * lambert]), (4, 4, [1000 * lambert, 200 * lambert]),
                                (2, 4, [1000 * lambert, 200 * lambert, 1000 * lambert]),
                                (0, 4, [1000 * lambert, 200 * lambert, 1000 * lambert, 200 * lambert]),
                                (3, 4, [200 * lambert, 1000 * lambert, 200 * lambert])):
    same_shape = pt.Distribution(*pt.table_ldt(ldt(symmetry, planes_, rows)))
    assert np.allclose(same_shape.density, (around if symmetry == 1 else quarter).density, atol=1e-9), symmetry
for broken in ("no numbers here", ies([0, 45], [lambert, lambert]), ies([0], [lambert]).replace("TILT=NONE", "TILT=lamp.tlt")):
    try:
        pt.table_ies(broken)
        raise AssertionError("not refused")
    except ValueError:
        pass
import tempfile  # noqa: E402

file_ = Path(tempfile.mkdtemp()) / "lamp.ies"
file_.write_text(ies([0], [1000 * lambert]))
assert np.allclose(pt.load_distribution(str(file_)).density, around.density) and len(pt.load_table(str(file_))[0]) == len(gammas)
try:
    pt.load_table("/no/such/file.ies")
    raise AssertionError("not refused")
except ValueError as e:
    assert "cannot be read" in str(e), e

# ---- the error from the spread of the batches; shares of the samples; adaptive sampling; threads
scene = scene_of([wall], [BLACK], [point((0, 0, 50))])
sensor = pt.TexelSensor(scene, [0], 4)
runs = [pt.render(scene, sensor, "light", 40_000, 2, seed=SEED + k, batch=5_000) for k in range(8)]
assert pt.render(scene, sensor, "light", 4_000, 2, seed=SEED, batch=4_000)["error"] is None
scatter = np.std([r["values"] for r in runs], axis=0, ddof=1)
stated = np.mean([r["error"] for r in runs], axis=0)
assert np.all(stated > 0) and abs(np.median(stated / scatter) - 1) < 0.4, stated / scatter
more = pt.render(scene, sensor, "light", 160_000, 2, seed=SEED, batch=10_000)["error"]
assert abs(np.median(stated / more) - 2) < 0.5
smooth = pt.render(scene, sensor, "path", 40_000, 2, seed=SEED, batch=5_000)            # evenly spread points
assert np.median(smooth["error"] / smooth["values"]) < 0.012        # (their number alone would scatter by 2 %)
shares = np.random.default_rng(5).uniform(0.01, 1.0, 16)
for method in ("path", "bidir"):
    even = values(scene, sensor, method, 400_000, 2)
    skew = values(scene, sensor, method, 400_000, 2, weights=shares)
    assert abs(skew.sum() / even.sum() - 1) < 0.005 and np.allclose(skew / even, 1, atol=0.03), (method, skew / even)
    assert np.allclose(sensor.prob, shares / shares.sum())
lit = scene_of([wall, rect((-100, 100, 0), (200, 0, 0), (0, 0, 150))], [WHITE, WHITE], [point((0, 0, 50))])
both = pt.TexelSensor(lit, [0, 1], 3)
reference = values(lit, both, "light", 2_000_000, 4)
shares = np.random.default_rng(6).uniform(0.02, 1.0, both.count)
for method in ("path", "bidir"):
    skew = values(lit, both, method, 600_000, 4, weights=shares)
    assert abs(skew.sum() / reference.sum() - 1) < 0.02 and np.allclose(skew / reference, 1, atol=0.1), (method, skew / reference)
    adapted = values(lit, both, method, 600_000, 4, adaptive=True)
    assert abs(adapted.sum() / reference.sum() - 1) < 0.02 and np.allclose(adapted / reference, 1, atol=0.1), method
camera = pt.Camera((0, 0, 200), (0, 0, 0), (0, 1, 0), 40.0, 4, 4)
shares = np.random.default_rng(7).uniform(0.01, 1.0, 16)
even = values(lit, camera, "path", 400_000, 3)
for method in ("path", "bidir", "photon"):
    skew = values(lit, camera, method, 400_000, 3, weights=shares)
    assert np.allclose(skew / even, 1, atol=0.06), (method, skew / even)
half = pt.Camera((0, 0, 300), (0, 0, 0), (0, 1, 0), 110.0, 16, 8)
plain = pt.render(lit, half, "path", 400_000, 3, seed=SEED, batch=20_000)
steered = pt.render(lit, half, "path", 400_000, 3, seed=SEED, batch=20_000, adaptive=True)
seen = plain["values"] > 0.1 * plain["values"].max()
assert 15 <= seen.sum() <= 40 and half.prob[seen].sum() > 3 * seen.mean()
assert np.all(np.abs(steered["values"] - plain["values"])[seen]
              < (6 * np.hypot(steered["error"], plain["error"]) + 0.02 * plain["values"])[seen])
assert (steered["error"][seen] ** 2).sum() / (plain["error"][seen] ** 2).sum() < 0.8
one_thread = pt.render(lit, both, "bidir", 100_000, 4, seed=SEED, threads=1)
many = pt.render(lit, both, "bidir", 100_000, 4, seed=SEED, threads=4)
assert one_thread["threads"] == 1 and many["threads"] == 4
assert np.allclose(one_thread["values"], many["values"], rtol=1e-9) and one_thread["rays"] == many["rays"]
# ---- the light that leaves the scene: a point source over a mirror square
square = rect((-10, -10, 0), (20, 0, 0), (0, 20, 0))                      # normal +z, towards the source
scene = scene_of([square], [MIRROR], [point((0, 0, 5), flux=2.0)])
share = solid_angle(10, 10, 5) / (4 * math.pi)
gone = [pt.escapes(scene, 200_000, 3, seed=SEED, index=k) for k in range(2)]
assert all(g["d"].shape == (len(g["flux"]), 3) for g in gone)
direct = sum(g["flux"][g["bounces"] == 0].sum() for g in gone) / 2
mirrored = sum(g["flux"][g["bounces"] == 1].sum() for g in gone) / 2
assert abs(direct / (2 * (1 - share)) - 1) < 0.005 and abs(mirrored / (2 * 0.9 * share) - 1) < 0.005, (direct, mirrored)
for g in gone:
    once = g["bounces"] == 1
    assert once.any() and (g["bounces"] <= 1).all() and (g["first"][once] == 0).all() and (g["last"][once] == 0).all()
    assert (g["first"][~once] == -1).all() and (g["d"][once, 2] > 0).all()
    down = ~once & (g["d"][:, 2] < 0)                                    # straight down: beside the square
    reach = 5 / -g["d"][down, 2]
    assert (np.maximum(np.abs(g["d"][down, 0]), np.abs(g["d"][down, 1])) * reach > 10 - 1e-6).all()
assert not np.array_equal(gone[0]["flux"], gone[1]["flux"])                # the batches differ
short = pt.escapes(scene, 50_000, 1, seed=SEED)
assert (short["bounces"] == 0).all() and abs(short["flux"].sum() / (2 * (1 - share)) - 1) < 0.01
alone, crowd = (pt.escapes(scene, 20_000, 3, seed=SEED, threads=t) for t in (1, 4))
assert alone["threads"] == 1 and crowd["threads"] == 4 and alone["rays"] == crowd["rays"]
assert all(np.array_equal(alone[key], crowd[key]) for key in ("d", "flux", "channel", "bounces", "first", "last"))
past = pt.escapes(scene, 20_000, 3, seed=SEED, skip=[0])                  # the square passed over: no first surface
assert (past["first"] == -1).all() and np.array_equal(past["last"], alone["last"])
assert np.array_equal(pt.escapes(scene, 20_000, 3, seed=SEED, skip=[5])["first"], alone["first"])
for bad, word in (({"depth": 0}, "segment"), ({"skip": [-1]}, "skip")):
    try:
        pt.escapes(scene, 10, **{"depth": 3, **bad})
        raise AssertionError("not refused")
    except ValueError as e:
        assert word in str(e), e

fractions, calls = [], []
stopped = pt.render(lit, both, "bidir", 5_000_000, 6, seed=SEED, batch=20_000, progress=fractions.append,
                    cancelled=lambda: calls.append(1) or len(calls) > 2)
assert stopped["samples"] == 60_000 and np.allclose(fractions, [0.004, 0.008, 0.012])
points, starts = pt.light_paths(lit, 50, 4, SEED)
assert len(starts) == 51 and starts[-1] == len(points) and points.shape[1] == 3 and (np.diff(starts) >= 2).all()

print(f"ok ({time.time() - started:.1f} s)")
