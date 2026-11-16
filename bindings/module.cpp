// Python module pathtracer._core: the kernel with NumPy arrays and the geometry dicts of the
// .nfcad format (as Raytatouille Fusion hands them to plugins).
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pathtracer/pathtracer.hpp"

namespace py = pybind11;
using namespace pt;

namespace {

std::vector<Vec3> points_from(const py::object& flat) {
  const auto v = flat.cast<std::vector<double>>();
  if (v.size() % 3) throw std::invalid_argument("points need three coordinates each");
  std::vector<Vec3> out(v.size() / 3);
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = {v[3 * i], v[3 * i + 1], v[3 * i + 2]};
  return out;
}

std::vector<double> weights_from(const py::dict& g) {
  if (!g.contains("weights") || g["weights"].is_none()) return {};
  return g["weights"].cast<std::vector<double>>();
}

Curve curve_from(const py::dict& g) {
  Curve c;
  c.degree = g["degree"].cast<int>();
  c.knots = g["knots"].cast<std::vector<double>>();
  c.points = points_from(g["points"]);
  c.weights = weights_from(g);
  if (c.knots.size() != c.points.size() + c.degree + 1 || (!c.weights.empty() && c.weights.size() != c.points.size()))
    throw std::invalid_argument("control points do not match the knot vector");
  return c;
}

Surface surface_from(const py::dict& g) {
  Surface s;
  s.degree_u = g["degree_u"].cast<int>();
  s.degree_v = g["degree_v"].cast<int>();
  s.knots_u = g["knots_u"].cast<std::vector<double>>();
  s.knots_v = g["knots_v"].cast<std::vector<double>>();
  s.num_u = g["num_u"].cast<int>();
  s.num_v = g["num_v"].cast<int>();
  s.points = points_from(g["points"]);
  s.weights = weights_from(g);
  s.check();
  return s;
}

TrimmedFace face_from(const py::dict& g) {
  TrimmedFace f;
  f.surface = surface_from(g["surface"].cast<py::dict>());
  auto loop = [](const py::object& curves) {
    std::vector<Curve> out;
    for (const auto& c : curves.cast<py::list>()) out.push_back(curve_from(c.cast<py::dict>()));
    return out;
  };
  if (g.contains("outer") && !g["outer"].is_none() && py::len(g["outer"]) > 0) f.loops.push_back(loop(g["outer"]));
  if (g.contains("holes") && !g["holes"].is_none())
    for (const auto& h : g["holes"].cast<py::list>()) f.loops.push_back(loop(py::reinterpret_borrow<py::object>(h)));
  return f;
}

/// The faces of an entity's geometry dict (surface, trimmed_surface, shell).
std::vector<TrimmedFace> faces_from(const py::dict& g) {
  const auto kind = g.contains("type") ? g["type"].cast<std::string>() : std::string();
  std::vector<TrimmedFace> out;
  if (kind == "surface") {
    TrimmedFace f;
    f.surface = surface_from(g);
    out.push_back(std::move(f));
  } else if (kind == "trimmed_surface") {
    out.push_back(face_from(g));
  } else if (kind == "shell") {
    for (const auto& f : g["faces"].cast<py::list>()) out.push_back(face_from(f.cast<py::dict>()));
  }
  return out;
}

Vec3 vec_from(const py::object& o) {
  const auto v = o.cast<std::vector<double>>();
  if (v.size() != 3) throw std::invalid_argument("three coordinates are needed");
  return {v[0], v[1], v[2]};
}

template <class T>
py::array_t<T> array_of(const std::vector<T>& v) { return py::array_t<T>(v.size(), v.data()); }

py::array_t<double> array_of(const std::vector<Vec3>& v) {
  py::array_t<double> out({v.size(), std::size_t{3}});
  auto r = out.mutable_unchecked<2>();
  for (std::size_t i = 0; i < v.size(); ++i) {
    r(i, 0) = v[i].x;
    r(i, 1) = v[i].y;
    r(i, 2) = v[i].z;
  }
  return out;
}

py::array_t<bool> flags_of(const std::vector<std::uint8_t>& v) {
  py::array_t<bool> out(v.size());
  auto r = out.mutable_unchecked<1>();
  for (std::size_t i = 0; i < v.size(); ++i) r(i) = v[i] != 0;
  return out;
}

template <class T, class D>
T get(const py::dict& d, const char* key, D fallback) {
  return d.contains(key) && !d[key].is_none() ? d[key].cast<T>() : static_cast<T>(fallback);
}

std::vector<std::pair<double, double>> pairs_from(const py::dict& d, const char* key) {
  std::vector<std::pair<double, double>> out;
  if (!d.contains(key) || d[key].is_none()) return out;
  for (const auto& item : d[key]) {
    const auto v = item.cast<std::vector<double>>();
    if (v.size() != 2) throw std::invalid_argument("a spectrum is made of pairs (wavelength, value)");
    out.emplace_back(v[0], v[1]);
  }
  return out;
}

Method method_from(const std::string& name) {
  if (name == "light") return kLight;
  if (name == "path") return kPath;
  if (name == "bidir") return kBidir;
  if (name == "photon") return kPhoton;
  throw std::invalid_argument("unknown method '" + name + "'");
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "PathTracer kernel (C++): light transport on trimmed NURBS geometry";
  m.attr("KINDS") = std::vector<std::string>(kKindNames, kKindNames + 8);
  m.attr("METHODS") = std::vector<std::string>{"light", "path", "bidir", "photon"};

  py::class_<Mesh>(m, "Mesh")
      .def_property_readonly("triangles", &Mesh::triangles)
      .def_property_readonly("faces", &Mesh::faces)
      .def_property_readonly("entity_ids", [](const Mesh& s) { return s.entity_ids; })
      .def_property_readonly("face_entity", [](const Mesh& s) { return array_of(s.face_entity); })
      .def_property_readonly("face_index", [](const Mesh& s) { return array_of(s.face_index); })
      .def_property_readonly("face_area", [](const Mesh& s) { return array_of(s.face_area); })
      .def_property_readonly("face_first", [](const Mesh& s) { return array_of(s.face_first); })
      .def_property_readonly("face", [](const Mesh& s) { return array_of(s.face); })
      .def_property_readonly("v0", [](const Mesh& s) { return array_of(s.v0); })
      .def_property_readonly("e1", [](const Mesh& s) { return array_of(s.e1); })
      .def_property_readonly("e2", [](const Mesh& s) { return array_of(s.e2); })
      .def_property_readonly("ng", [](const Mesh& s) { return array_of(s.ng); })
      .def_property_readonly("curved", [](const Mesh& s) {
        std::vector<bool> out;
        for (const auto& g : s.grids) out.push_back(g.curved);
        return out;
      })
      .def_property_readonly("mask_res", [](const Mesh& s) {
        std::vector<int> out;
        for (const auto& g : s.grids) out.push_back(g.mask_res);
        return out;
      })
      .def("texel_areas", [](const Mesh& s, int face, int w, int h) {
        return array_of(s.texel_areas(face, w, h)).reshape({h, w});
      }, "Area / mm² of the face in every texel: [row v, column u]")
      .def("shade", [](const Mesh& s, long long tri, double b1, double b2) {
        Vec3 n;
        double u, v;
        s.shade(tri, b1, b2, n, u, v);
        return py::make_tuple(py::make_tuple(n.x, n.y, n.z), u, v);
      })
      .def("shade_exact", [](const Mesh& s, long long tri, double b1, double b2, const py::object& o,
                             const py::object& d, double t) {
        Vec3 n;
        double u, v;
        s.shade_exact(tri, b1, b2, vec_from(o), vec_from(d), t, n, u, v);
        return py::make_tuple(py::make_tuple(n.x, n.y, n.z), u, v);
      });

  m.def("build_mesh", [](const py::list& entities, double tol) {
    std::vector<Body> bodies;
    long long number = 0;
    for (const auto& item : entities) {
      const auto e = item.cast<py::dict>();
      Body b;
      b.id = e.contains("id") ? e["id"].cast<long long>() : number;
      if (e.contains("geometry") && !e["geometry"].is_none()) b.faces = faces_from(e["geometry"].cast<py::dict>());
      bodies.push_back(std::move(b));
      ++number;
    }
    py::gil_scoped_release release;
    return build_mesh(bodies, tol);
  }, py::arg("entities"), py::arg("tol"),
        "The mesh of the surfaces, trimmed surfaces and shells among the entities (dicts with id and geometry)");

  py::class_<Distribution, std::shared_ptr<Distribution>>(m, "Distribution")
      .def(py::init([](const std::vector<double>& theta, const std::vector<double>& phi, const py::array_t<double>& cd) {
        const auto flat = py::array_t<double, py::array::c_style | py::array::forcecast>(cd);
        return std::make_shared<Distribution>(theta, phi, std::vector<double>(flat.data(), flat.data() + flat.size()));
      }), py::arg("theta"), py::arg("phi"), py::arg("intensity"))
      .def("eval", &Distribution::eval)
      .def_property_readonly("symmetric", &Distribution::symmetric)
      .def_property_readonly("density", [](const Distribution& d) { return array_of(d.density()); });
  m.def("v_lambda", &v_lambda, "The eye's photopic sensitivity at a wavelength in nm");
  auto table = [](const LuminaireTable& t) {
    return py::make_tuple(array_of(t.theta), array_of(t.phi), array_of(t.intensity).reshape({t.phi.size(), t.theta.size()}));
  };
  m.def("table_ies", [table](const std::string& text) { return table(table_ies(text)); },
        "theta, phi (degrees) and intensity[phi, theta] from the text of an IES LM-63 file (type C)");
  m.def("table_ldt", [table](const std::string& text) { return table(table_ldt(text)); },
        "theta, phi (degrees) and intensity[phi, theta] from the text of an EULUMDAT file");
  m.def("load_table", [table](const std::string& path) { return table(load_luminaire(path)); },
        "theta, phi and intensity[phi, theta] from an .ies or .ldt file");
  m.def("load_distribution", [](const std::string& path) {
    return std::make_shared<Distribution>(load_luminaire(path).distribution());
  }, "The distribution of an .ies or .ldt file");

  py::class_<Scene>(m, "Scene")
      .def(py::init([](const Mesh& mesh, const py::list& materials, const std::vector<int>& entity_material,
                       const py::list& sources, const std::vector<int>& closed) {
        std::vector<Material> rows;
        for (const auto& item : materials) {
          const auto d = item.cast<py::dict>();
          Material r;
          r.name = get<std::string>(d, "name", "");
          const auto kind = get<std::string>(d, "kind", "diffuse");
          r.kind = -1;
          for (int k = 0; k < 8; ++k)
            if (kind == kKindNames[k]) r.kind = k;
          if (r.kind < 0) throw std::invalid_argument("material '" + r.name + "': unknown kind '" + kind + "'");
          r.reflectance = get<double>(d, "reflectance", 0.0);
          r.transmittance = get<double>(d, "transmittance", 0.0);
          r.roughness = get<double>(d, "roughness", 0.2);
          r.ior = get<double>(d, "ior", 1.5);
          r.absorption = get<double>(d, "absorption", 0.0);
          r.abbe = get<double>(d, "abbe", 0.0);
          r.spectrum = pairs_from(d, "spectrum");
          rows.push_back(std::move(r));
        }
        std::vector<SourceSpec> specs;
        for (const auto& item : sources) {
          const auto d = item.cast<py::dict>();
          SourceSpec s;
          s.name = get<std::string>(d, "name", "");
          s.flux = get<double>(d, "flux", 1.0);
          s.half_angle = get<double>(d, "half_angle", 90.0);
          const auto side = get<std::string>(d, "side", "front");
          if (side != "front" && side != "back" && side != "both")
            throw std::invalid_argument("source '" + s.name + "': side must be front, back or both");
          s.side = side == "front" ? 1 : side == "back" ? -1 : 0;
          if (d.contains("entities") && !d["entities"].is_none()) s.entities = d["entities"].cast<std::vector<int>>();
          if (d.contains("points") && !d["points"].is_none())
            for (const auto& p : d["points"]) s.points.push_back(vec_from(py::reinterpret_borrow<py::object>(p)));
          if (d.contains("curves") && !d["curves"].is_none())
            for (const auto& c : d["curves"]) {
              std::vector<Vec3> line;
              for (const auto& p : c) line.push_back(vec_from(py::reinterpret_borrow<py::object>(p)));
              s.curves.push_back(std::move(line));
            }
          if (d.contains("spectrum") && !d["spectrum"].is_none()) s.spectrum = pairs_from(d, "spectrum");
          if (d.contains("distribution") && !d["distribution"].is_none())
            s.distribution = d["distribution"].cast<std::shared_ptr<Distribution>>();
          if (d.contains("axis") && !d["axis"].is_none()) s.axis = vec_from(d["axis"]);
          specs.push_back(std::move(s));
        }
        return std::make_unique<Scene>(mesh, Materials(std::move(rows)), entity_material, specs, closed);
      }), py::arg("mesh"), py::arg("materials"), py::arg("entity_material"), py::arg("sources"),
           py::arg("closed") = std::vector<int>{})
      .def_property_readonly("mesh", [](const Scene& s) -> const Mesh& { return s.mesh; },
                             py::return_value_policy::reference_internal)
      .def_readwrite("exact", &Scene::exact)
      .def_readwrite("contact", &Scene::contact)
      .def_readonly("size", &Scene::size)
      .def_readonly("flux", &Scene::flux)
      .def_readonly("flipped", &Scene::flipped)
      .def_readonly("has_detectors", &Scene::has_detectors)
      .def_property_readonly("wavelengths", [](const Scene& s) { return array_of(s.wavelengths); })
      .def_property_readonly("channel_prob", [](const Scene& s) { return array_of(s.channel_prob); })
      .def_property_readonly("src_flux", [](const Scene& s) { return array_of(s.src_flux); })
      .def_property_readonly("src_prob", [](const Scene& s) { return array_of(s.src_prob); })
      .def_property_readonly("src_is_wire", [](const Scene& s) { return flags_of(s.src_is_wire); })
      .def_property_readonly("src_is_point", [](const Scene& s) { return flags_of(s.src_is_point); })
      .def_property_readonly("src_is_measured", [](const Scene& s) {
        std::vector<bool> out;
        for (const auto& d : s.src_measured) out.push_back(d != nullptr);
        return out;
      })
      .def_property_readonly("face_virtual", [](const Scene& s) { return flags_of(s.face_virtual); })
      .def_property_readonly("face_material", [](const Scene& s) { return array_of(s.face_material); })
      .def_property_readonly("material_names", [](const Scene& s) {
        std::vector<std::string> out;
        for (std::size_t k = 0; k < s.materials.size(); ++k) out.push_back(s.materials.row(static_cast<int>(k)).name);
        return out;
      })
      .def("ior", [](const Scene& s, int material, int channel) { return s.materials.ior(material, channel); })
      .def("reflectance", [](const Scene& s, int material, int channel) { return s.materials.rho(material, channel); })
      .def("sample_material", [](const Scene& s, int material, const py::object& n, const py::object& wo, long long count,
                                 std::uint64_t seed, bool radiance, int channel, double outside) {
        // `count` scattered directions of one material for one incoming direction: wi, weight, pdf
        const Vec3 nn = vec_from(n), w = vec_from(wo);
        std::vector<Vec3> wi(count);
        std::vector<double> weight(count), pdf(count);
        for (long long i = 0; i < count; ++i) {
          Rng rng(seed, 0, static_cast<std::uint64_t>(i));
          s.materials.sample(material, nn, w, rng, radiance, channel, outside, false, wi[i], weight[i], pdf[i]);
        }
        return py::make_tuple(array_of(wi), array_of(weight), array_of(pdf));
      }, py::arg("material"), py::arg("n"), py::arg("wo"), py::arg("count"), py::arg("seed") = 0,
           py::arg("radiance") = true, py::arg("channel") = 0, py::arg("outside") = 1.0)
      .def("eval_material", [](const Scene& s, int material, const py::object& n, const py::object& wo,
                               const py::array_t<double>& wi, bool radiance, int channel, double outside) {
        // f and the densities of sampling wi from wo and wo from wi, for directions wi (count, 3)
        const Vec3 nn = vec_from(n), w = vec_from(wo);
        const auto r = wi.unchecked<2>();
        std::vector<double> f(r.shape(0)), fwd(r.shape(0)), rev(r.shape(0));
        for (py::ssize_t i = 0; i < r.shape(0); ++i)
          s.materials.eval(material, nn, w, {r(i, 0), r(i, 1), r(i, 2)}, channel, radiance, outside, f[i], fwd[i], rev[i]);
        return py::make_tuple(array_of(f), array_of(fwd), array_of(rev));
      }, py::arg("material"), py::arg("n"), py::arg("wo"), py::arg("wi"), py::arg("radiance") = true,
           py::arg("channel") = 0, py::arg("outside") = 1.0)
      .def("hit", [](const Scene& s, const py::object& o, const py::object& d) {
        const Hit h = s.hit(vec_from(o), vec_from(d));
        return py::make_tuple(h.t, h.tri, h.b1, h.b2);
      }, "Nearest hit of one ray: t, triangle (-1: none), b1, b2");

  py::class_<Sensor>(m, "Sensor")
      .def_readonly("count", &Sensor::count)
      .def_readonly("camera", &Sensor::camera)
      .def_readonly("pinhole", &Sensor::pinhole)
      .def_readonly("crossed", &Sensor::crossed)
      .def_property_readonly("prob", [](const Sensor& s) { return array_of(s.prob); })
      .def("reset", &Sensor::reset)
      .def("set_weights", &Sensor::set_weights)
      .def("adapt", [](Sensor& s, const std::vector<double>& error, double floor) { s.adapt(error, floor); },
           py::arg("error"), py::arg("floor") = 0.3);

  py::class_<TexelSensor, Sensor>(m, "TexelSensor")
      .def(py::init<const Scene&, const std::vector<int>&, int>(), py::arg("scene"), py::arg("faces"),
           py::arg("long_side"), py::keep_alive<1, 2>())
      .def_property_readonly("scene", [](const TexelSensor& s) -> const Scene& { return s.scene; },
                             py::return_value_policy::reference)
      .def_property_readonly("faces", [](const TexelSensor& s) { return array_of(s.faces); })
      .def_property_readonly("w", [](const TexelSensor& s) { return array_of(s.w); })
      .def_property_readonly("h", [](const TexelSensor& s) { return array_of(s.h); })
      .def_property_readonly("first", [](const TexelSensor& s) { return array_of(s.first); })
      .def_property_readonly("texel_area", [](const TexelSensor& s) { return array_of(s.texel_area); })
      .def_readonly("area", &TexelSensor::area)
      .def("points", [](const TexelSensor& s, long long n, std::uint64_t seed) {
        // n points as the sensor draws them: triangle (-1: none found), b1, b2, texel
        std::vector<long long> tri(n), tex(n);
        std::vector<double> b1(n), b2(n);
        Rng order(seed, 0, 0xffffffffULL);
        std::vector<double> pick(n);
        for (long long i = 0; i < n; ++i) pick[i] = (static_cast<double>(i) + order.next()) / static_cast<double>(n);
        for (long long i = 0; i < n; ++i) {
          Rng rng(seed, 0, static_cast<std::uint64_t>(i));
          if (!s.point(pick[i], rng, tri[i], b1[i], b2[i], tex[i])) tri[i] = -1;
        }
        return py::make_tuple(array_of(tri), array_of(b1), array_of(b2), array_of(tex));
      });

  py::class_<Camera, Sensor>(m, "Camera")
      .def(py::init([](const py::object& position, const py::object& target, const py::object& up, double fov,
                       int width, int height, double lens_radius, double focus) {
        return std::make_unique<Camera>(vec_from(position), vec_from(target), vec_from(up), fov, width, height,
                                        lens_radius, focus);
      }), py::arg("position"), py::arg("target"), py::arg("up"), py::arg("fov"), py::arg("width"), py::arg("height"),
           py::arg("lens_radius") = 0.0, py::arg("focus") = 0.0)
      .def_readonly("w", &Camera::width)
      .def_readonly("h", &Camera::height)
      .def_readonly("film", &Camera::film)
      .def_readonly("focus", &Camera::focus)
      .def_readonly("lens_radius", &Camera::lens_radius);

  m.def("render", [](const Scene& scene, Sensor& sensor, const std::string& method, long long samples, int depth,
                     std::uint64_t seed, int threads, double radius, bool adaptive, const py::object& weights,
                     long long batch, const py::object& progress, const py::object& cancelled) {
    RenderOptions opt;
    opt.method = method_from(method);
    opt.samples = samples;
    opt.depth = depth;
    opt.seed = seed;
    opt.threads = threads;
    opt.radius = radius;
    opt.adaptive = adaptive;
    opt.batch = batch;
    if (!weights.is_none()) opt.weights = weights.cast<std::vector<double>>();
    if (!progress.is_none())
      opt.progress = [progress](double f) {
        py::gil_scoped_acquire gil;
        progress(f);
      };
    if (!cancelled.is_none())
      opt.cancelled = [cancelled]() {
        py::gil_scoped_acquire gil;
        return cancelled().cast<bool>();
      };
    Result r;
    {
      py::gil_scoped_release release;
      r = render(scene, sensor, opt);
    }
    py::dict out;
    out["values"] = array_of(r.values);
    out["channels"] = array_of(r.channels).reshape({scene.wavelengths.size(), sensor.count});
    out["error"] = r.error.empty() ? py::object(py::none()) : py::object(array_of(r.error));
    out["samples"] = r.samples;
    out["rays"] = r.rays;
    out["threads"] = r.threads;
    if (opt.method == kPhoton && sensor.camera) {
      out["passes"] = r.passes;
      out["radius"] = r.radius;
    }
    return out;
  }, py::arg("scene"), py::arg("sensor"), py::arg("method"), py::arg("samples"), py::arg("depth"),
        py::arg("seed") = 0, py::arg("threads") = 0, py::arg("radius") = 0.0, py::arg("adaptive") = false,
        py::arg("weights") = py::none(), py::arg("batch") = 0, py::arg("progress") = py::none(),
        py::arg("cancelled") = py::none(),
        "Estimates the sensor's values: {values, channels, error, samples, rays, threads[, passes, radius]}");

  m.def("light_paths", [](const Scene& scene, int n, int depth, std::uint64_t seed) {
    std::vector<Vec3> points;
    std::vector<long long> starts;
    light_paths(scene, n, depth, seed, points, starts);
    return py::make_tuple(array_of(points), array_of(starts));
  }, py::arg("scene"), py::arg("n"), py::arg("depth"), py::arg("seed") = 0,
        "Light paths as polylines for drawing: points (m, 3) and the start of each polyline (plus the total)");

  m.def("escapes", [](const Scene& scene, long long n, int depth, std::uint64_t seed, std::uint64_t index,
                      int threads, const std::vector<int>& skip) {
    Escapes e;
    {
      py::gil_scoped_release release;
      e = escapes(scene, n, depth, seed, index, threads, skip);
    }
    py::dict out;
    out["d"] = array_of(e.d);
    out["flux"] = array_of(e.flux);
    out["channel"] = array_of(e.channel);
    out["bounces"] = array_of(e.bounces);
    out["first"] = array_of(e.first);
    out["last"] = array_of(e.last);
    out["rays"] = e.rays;
    out["threads"] = e.threads;
    return out;
  }, py::arg("scene"), py::arg("n"), py::arg("depth"), py::arg("seed") = 0, py::arg("index") = 0,
        py::arg("threads") = 0, py::arg("skip") = std::vector<int>{},
        "The light that leaves the scene, from n light subpaths of up to depth segments: {d (m, 3), flux, channel, "
        "bounces, first, last, rays, threads} – the fluxes of one call add up to the escaping flux; bounces: surface "
        "events before it left; first / last: entities of the first surface not in skip and of the last surface "
        "(-1: none)");
}
