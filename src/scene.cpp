// The scene of a computation: mesh, materials, sources (see the Python kernel: pt_scene.py).
#include "pathtracer/pathtracer.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>

namespace pt {
namespace {

/// Two unit vectors across the axis: the first is the world's X axis projected across it (Y where
/// the axis is along X) – the direction phi = 0 of a measured distribution.
void across(const Vec3& axis, Vec3& a, Vec3& b) {
  const Vec3 ref = std::abs(axis.x) < 0.999 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
  a = normalized(ref - axis * dot(ref, axis));
  b = cross(axis, a);
}

std::size_t pick_from(const std::vector<double>& cum, double u) {
  return std::min<std::size_t>(std::upper_bound(cum.begin(), cum.end(), u * cum.back()) - cum.begin(), cum.size() - 1);
}

}  // namespace

Scene::Scene(Mesh mesh_in, Materials materials_in, std::vector<int> entity_material_in,
             const std::vector<SourceSpec>& sources, const std::vector<int>& closed)
    : mesh(std::move(mesh_in)), materials(std::move(materials_in)), entity_material(std::move(entity_material_in)) {
  const std::size_t tris = mesh.triangles();
  if (entity_material.size() != mesh.entity_ids.size())
    throw std::invalid_argument("one material per body is needed");
  for (int m : entity_material)
    if (m < 0 || m >= static_cast<int>(materials.size())) throw std::invalid_argument("unknown material number");
  Vec3 lo{kInf, kInf, kInf}, hi{-kInf, -kInf, -kInf};
  for (std::size_t t = 0; t < tris; ++t)
    for (const Vec3& p : {mesh.v0[t], mesh.v0[t] + mesh.e1[t], mesh.v0[t] + mesh.e2[t]}) {
      lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
      hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
  size = norm(hi - lo);
  eps = 1e-7 * std::max(size, 1e-12);
  contact = 1e-5 * size;
  tri_entity.resize(tris);
  for (std::size_t t = 0; t < tris; ++t) tri_entity[t] = mesh.face_entity[mesh.face[t]];
  if (!closed.empty()) orient(closed);
  face_material.resize(mesh.faces());
  face_virtual.resize(mesh.faces());
  for (std::size_t f = 0; f < mesh.faces(); ++f) {
    face_material[f] = entity_material[mesh.face_entity[f]];
    face_virtual[f] = materials.kind(face_material[f]) == kDetector;
  }
  tri_material.resize(tris);
  tri_glassy.resize(tris);
  std::vector<long long> real, virt;
  for (std::size_t t = 0; t < tris; ++t) {
    tri_material[t] = face_material[mesh.face[t]];
    tri_glassy[t] = materials.glassy(tri_material[t]);
    (face_virtual[mesh.face[t]] ? virt : real).push_back(static_cast<long long>(t));
  }
  // detectors are no surfaces for the light: they get a BVH of their own
  has_detectors = !virt.empty();
  bvh_ = Bvh(mesh, real);
  detectors_ = Bvh(mesh, virt);
  make_sources(sources);
}

Hit Scene::hit(const Vec3& o, const Vec3& d) const {
  return bvh_.intersect(o, d, kInf, false, [&](long long t, double b1, double b2) { return mesh.accept(t, b1, b2); });
}

bool Scene::occluded(const Vec3& o, const Vec3& target) const {
  return bvh_.intersect(o, target - o, 1.0, true, [&](long long t, double b1, double b2) {
           return mesh.accept(t, b1, b2);
         }).tri >= 0;
}

Hit Scene::crossing(const Vec3& o, const Vec3& d, double tmax) const {
  return detectors_.intersect(o, d, tmax, false,
                              [&](long long t, double b1, double b2) { return mesh.accept(t, b1, b2); });
}

Vec3 Scene::offset(const Vec3& p, const Vec3& ng, const Vec3& towards) const {
  const double s = dot(ng, towards);
  return p + ng * ((s > 0 ? 1.0 : s < 0 ? -1.0 : 0.0) * eps);
}

int Scene::touching(const Vec3& p, const Vec3& d, int body, bool ends) const {
  if (bvh_.empty() || !(contact > 0)) return -1;
  const Hit h = bvh_.intersect(p - d * contact, d, 2 * contact, false, [&](long long t, double b1, double b2) {
    const double facing = dot(mesh.ng[t], d);
    const bool ok = ends ? (tri_entity[t] == body && facing > 0)
                         : (tri_glassy[t] && tri_entity[t] != body && facing < 0);
    return ok && mesh.accept(t, b1, b2);
  });
  return h.tri >= 0 ? tri_entity[h.tri] : -1;
}

// ---------------------------------------------------------------------------- orientation
void Scene::orient(const std::vector<int>& closed) {
  // A ray from a face along its normal leaves a closed body through an even number of its faces,
  // the opposite ray through an odd number. Faces without a clear answer (open shells) stay.
  std::vector<long long> all(mesh.triangles());
  std::iota(all.begin(), all.end(), 0);
  const Bvh full(mesh, all);
  Rng rng(1);
  const int per = 5;
  std::vector<int> flip;
  for (std::size_t f = 0; f < mesh.faces(); ++f) {
    if (!(mesh.face_area[f] > 0) || std::find(closed.begin(), closed.end(), mesh.face_entity[f]) == closed.end())
      continue;
    const int entity = mesh.face_entity[f];
    int keep = 0, turn = 0;
    for (int k = 0; k < per; ++k) {
      long long tri;
      double b1, b2;
      mesh.sample_face(static_cast<int>(f), rng, rng.next(), tri, b1, b2);
      const Vec3 p = mesh.point(tri, b1, b2), n = mesh.ng[tri];
      int counts[2];
      for (int s = 0; s < 2; ++s) {
        const Vec3 d = n * (s == 0 ? 1.0 : -1.0);
        Vec3 o = p + d * eps;
        int count = 0;
        for (int step = 0; step < 256; ++step) {
          const Hit h = full.intersect(o, d, kInf, false, [&](long long t, double c1, double c2) {
            return tri_entity[t] == entity && mesh.accept(t, c1, c2);
          });
          if (h.tri < 0) break;
          ++count;
          o = o + d * (h.t + eps);
        }
        counts[s] = count;
      }
      if (counts[0] % 2 == 0 && counts[1] % 2 == 1) ++keep;  // normal points outwards
      if (counts[0] % 2 == 1 && counts[1] % 2 == 0) ++turn;  // normal points inwards
    }
    if (turn > std::max(keep, per - turn - keep)) flip.push_back(static_cast<int>(f));
  }
  if (!flip.empty()) mesh.flip_faces(flip);
  flipped = flip;
}

// ---------------------------------------------------------------------------- sources
void Scene::make_sources(const std::vector<SourceSpec>& specs) {
  face_source.assign(mesh.faces(), -1);
  std::vector<std::map<double, double>> spectra;
  auto add = [&](const SourceSpec& spec, double flux_, const std::map<double, double>& lines) {
    src_flux.push_back(flux_);
    src_sin.push_back(1.0);
    src_sides.push_back(0);
    src_area.push_back(1.0);
    src_point.push_back({});
    src_axis.push_back({0, 0, -1});
    src_is_point.push_back(1);
    src_is_wire.push_back(0);
    src_measured.push_back(nullptr);
    src_names.push_back(spec.name);
    src_faces_.emplace_back();
    src_face_cum_.emplace_back();
    wire_a_.emplace_back();
    wire_b_.emplace_back();
    wire_cum_.emplace_back();
    spectra.push_back(lines);
    return src_flux.size() - 1;
  };
  for (const SourceSpec& spec : specs) {
    const std::string who = "source '" + spec.name + "'";
    if (spec.flux < 0) throw std::invalid_argument(who + ": negative flux");
    const int kinds = (spec.entities.empty() ? 0 : 1) + (spec.points.empty() ? 0 : 1) + (spec.curves.empty() ? 0 : 1);
    if (kinds > 1)
      throw std::invalid_argument(who + " is on objects of different kinds (surfaces, points, curves): use one "
                                        "source for each kind");
    if (spec.flux == 0 || kinds == 0) continue;
    std::map<double, double> lines;
    double shares = 0;
    for (const auto& [nm, share] : spec.spectrum) {
      if (!(nm > 0 && share >= 0)) throw std::invalid_argument(who + ": wavelengths must be positive, shares not negative");
      lines[nm] += share;
      shares += share;
    }
    if (!(shares > 0)) throw std::invalid_argument(who + ": its spectrum is empty");
    if (!spec.entities.empty()) {
      if (!(spec.half_angle > 0 && spec.half_angle <= 90)) throw std::invalid_argument(who + ": half-angle must lie in 0 … 90°");
      std::vector<int> fs;
      for (std::size_t f = 0; f < mesh.faces(); ++f)
        if (mesh.face_area[f] > 0 &&
            std::find(spec.entities.begin(), spec.entities.end(), mesh.face_entity[f]) != spec.entities.end())
          fs.push_back(static_cast<int>(f));
      if (fs.empty()) throw std::invalid_argument(who + ": its surfaces have no area");
      for (int f : fs) {
        if (face_virtual[f])
          throw std::invalid_argument(who + ": a detector cannot be a source (light passes it as if it were not there)");
        if (face_source[f] >= 0) throw std::invalid_argument(who + ": a surface already belongs to another source");
      }
      const std::size_t k = add(spec, spec.flux, lines);
      src_is_point[k] = 0;
      src_sin[k] = std::sin(spec.half_angle * kPi / 180.0);
      src_sides[k] = spec.side;
      src_measured[k] = spec.distribution;
      double total = 0;
      for (int f : fs) {
        face_source[f] = static_cast<int>(k);
        src_face_cum_[k].push_back(total += mesh.face_area[f]);
      }
      src_faces_[k] = fs;
      src_area[k] = total;
    } else if (!spec.curves.empty()) {
      const std::size_t k = add(spec, spec.flux, lines);
      double total = 0;
      for (const auto& curve : spec.curves)
        for (std::size_t i = 0; i + 1 < curve.size(); ++i) {
          const double len = norm(curve[i + 1] - curve[i]);
          if (!(len > 0)) continue;
          wire_a_[k].push_back(curve[i]);
          wire_b_[k].push_back(curve[i + 1]);
          wire_cum_[k].push_back(total += len);
        }
      if (wire_cum_[k].empty()) throw std::invalid_argument(who + ": its curves have no length");
      src_is_wire[k] = 1;
    } else {
      for (const Vec3& p : spec.points) {
        const std::size_t k = add(spec, spec.flux / static_cast<double>(spec.points.size()), lines);
        src_point[k] = p;
        if (spec.distribution) {
          if (!(norm(spec.axis) > 0)) throw std::invalid_argument(who + ": its axis has no direction");
          src_measured[k] = spec.distribution;
          src_axis[k] = normalized(spec.axis);
        }
      }
    }
  }
  if (src_flux.empty()) throw std::invalid_argument("no source: give at least one object a source with a flux above 0");
  const std::size_t n = src_flux.size();
  flux = std::accumulate(src_flux.begin(), src_flux.end(), 0.0);
  src_prob.resize(n);
  src_cos.resize(n);
  src_radiance.resize(n);
  src_dir_norm.resize(n);
  double run = 0;
  for (std::size_t k = 0; k < n; ++k) {
    src_prob[k] = src_flux[k] / flux;
    src_cum_.push_back(run += src_prob[k]);
    src_cos[k] = std::sqrt(std::max(0.0, 1 - src_sin[k] * src_sin[k]));
    const double nsides = src_sides[k] == 0 ? 2.0 : 1.0;
    if (src_is_wire[k]) {  // intensity across the wire
      src_radiance[k] = src_flux[k] / (kPi * kPi);
      src_dir_norm[k] = 1 / (kPi * kPi);
    } else if (src_is_point[k]) {
      src_radiance[k] = src_flux[k] / (4 * kPi);
      src_dir_norm[k] = 1 / (4 * kPi);
    } else {
      src_radiance[k] = src_flux[k] / (nsides * src_area[k] * kPi * src_sin[k] * src_sin[k]);
      src_dir_norm[k] = 1 / (nsides * kPi * src_sin[k] * src_sin[k]);
    }
  }
  // wavelengths: the lines of all sources; per source the share of its flux in each
  for (const auto& lines : spectra)
    for (const auto& [nm, share] : lines)
      if (share > 0) wavelengths.push_back(nm);
  std::sort(wavelengths.begin(), wavelengths.end());
  wavelengths.erase(std::unique(wavelengths.begin(), wavelengths.end()), wavelengths.end());
  const std::size_t c = wavelengths.size();
  src_spectrum_.assign(n * c, 0.0);
  spectrum_cum_.assign(n * c, 0.0);
  channel_prob.assign(c, 0.0);
  for (std::size_t k = 0; k < n; ++k) {
    double total = 0;
    for (const auto& kv : spectra[k]) total += kv.second;
    double cum = 0;
    for (std::size_t j = 0; j < c; ++j) {
      const auto it = spectra[k].find(wavelengths[j]);
      const double share = it == spectra[k].end() ? 0.0 : it->second / total;
      src_spectrum_[k * c + j] = share;
      spectrum_cum_[k * c + j] = (cum += share);
      channel_prob[j] += src_prob[k] * share;
    }
  }
  materials.set_wavelengths(wavelengths);
}

double Scene::spectral_weight(int src, int channel) const {
  return src_spectrum_[static_cast<std::size_t>(src) * wavelengths.size() + channel] / channel_prob[channel];
}

double Scene::emit_pdf_pos(int src) const { return src_prob[src] / (src_is_point[src] ? 1.0 : src_area[src]); }

Vec3 Scene::measured_axis(int src, const Vec3& n) const {
  if (src_is_point[src]) return src_axis[src];
  return src_sides[src] < 0 ? -n : n;
}

void Scene::emitted(int src, const Vec3& n, const Vec3& w, double& le, double& pdf) const {
  if (src_measured[src]) {
    const Vec3 axis = measured_axis(src, n);
    Vec3 a, b;
    across(axis, a, b);
    const double c = dot(axis, w);
    const double density = src_measured[src]->eval(c, std::atan2(dot(b, w), dot(a, w)));
    pdf = density;
    // intensity; a surface: radiance = it / (area |cos|)
    le = src_is_point[src] ? src_flux[src] * density
                           : src_flux[src] * density / (src_area[src] * std::max(std::abs(c), 1e-9));
    return;
  }
  const double cos = dot(n, w);
  if (src_is_wire[src]) {
    const double sin = std::sqrt(std::max(0.0, 1 - cos * cos));  // to the wire
    le = src_radiance[src] * sin;
    pdf = src_dir_norm[src] * sin;
  } else if (src_is_point[src]) {
    le = src_radiance[src];
    pdf = src_dir_norm[src];
  } else {
    const int sides = src_sides[src];
    const bool ok = std::abs(cos) >= src_cos[src] && std::abs(cos) > 0 && (sides == 0 || (cos > 0 ? 1 : -1) == sides);
    le = ok ? src_radiance[src] : 0.0;
    pdf = ok ? src_dir_norm[src] * std::abs(cos) : 0.0;
  }
}

Emission Scene::emit(double pick, Rng& rng) const {
  Emission e;
  const std::size_t k = pick_from(src_cum_, pick);
  const std::size_t c = wavelengths.size();
  e.src = static_cast<int>(k);
  const double uc = rng.next();
  for (std::size_t j = 0; j < c; ++j)
    if (uc < spectrum_cum_[k * c + j] || j + 1 == c) {
      e.channel = static_cast<int>(j);
      break;
    }
  e.pdf_pos = emit_pdf_pos(e.src);
  e.point = src_is_point[k] != 0;
  auto surface_point = [&]() {
    const int f = src_faces_[k][pick_from(src_face_cum_[k], rng.next())];
    long long tri;
    double b1, b2, u, v;
    mesh.sample_face(f, rng, rng.next(), tri, b1, b2);
    e.p = mesh.point(tri, b1, b2);
    mesh.shade(tri, b1, b2, e.ns, u, v);
    e.ng = mesh.ng[tri];
  };
  if (src_measured[k]) {  // a direction by the measured distribution
    Vec3 axis;
    if (e.point) {
      e.p = src_point[k];
      axis = src_axis[k];
    } else {
      surface_point();
      axis = measured_axis(e.src, e.ns);
    }
    double cos, phi, density;
    src_measured[k]->sample(rng, cos, phi, density);
    Vec3 a, b;
    across(axis, a, b);
    const double sin = std::sqrt(std::max(0.0, 1 - cos * cos));
    e.d = a * (sin * std::cos(phi)) + b * (sin * std::sin(phi)) + axis * cos;
    if (e.point) e.ns = axis;  // (kept for emitted(); a point has no surface)
    e.pdf_dir = density;
    e.ok = e.point || dot(e.ng, e.d) * dot(e.ns, e.d) > 0;
  } else if (src_is_wire[k]) {  // a point on the wire, a direction as from a thin Lambertian cylinder
    const std::size_t seg = pick_from(wire_cum_[k], rng.next());
    const Vec3 &a = wire_a_[k][seg], &b = wire_b_[k][seg];
    e.p = a + (b - a) * rng.next();
    const Vec3 along = normalized(b - a);
    const Vec3 outward = around(along, 0.0, 2 * kPi * rng.next());
    const double cos = std::sqrt(1 - rng.next());
    e.d = around(outward, cos, 2 * kPi * rng.next());
    e.ns = along;
    const double t = dot(along, e.d);
    e.pdf_dir = src_dir_norm[k] * std::sqrt(std::max(0.0, 1 - t * t));
  } else if (e.point) {
    e.p = src_point[k];
    const double z = 1 - 2 * rng.next(), r = std::sqrt(std::max(0.0, 1 - z * z)), phi = 2 * kPi * rng.next();
    e.d = {r * std::cos(phi), r * std::sin(phi), z};
    e.pdf_dir = src_dir_norm[k];
  } else {
    surface_point();
    const double side = src_sides[k] != 0 ? static_cast<double>(src_sides[k]) : (rng.next() < 0.5 ? 1.0 : -1.0);
    const double cos = std::sqrt(1 - src_sin[k] * src_sin[k] * rng.next());
    e.d = around(e.ns * side, cos, 2 * kPi * rng.next());
    e.pdf_dir = src_dir_norm[k] * cos;
    e.ok = cos > 0 && dot(e.ng, e.d) * dot(e.ns, e.d) > 0;
  }
  return e;
}

}  // namespace pt
