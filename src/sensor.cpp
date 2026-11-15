// Sensors: irradiance on faces (a texture per face) and cameras (see pt_scene.py).
#include "pathtracer/pathtracer.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace pt {
namespace {
std::size_t pick_from(const std::vector<double>& cum, double u) {
  return std::min<std::size_t>(std::upper_bound(cum.begin(), cum.end(), u * cum.back()) - cum.begin(), cum.size() - 1);
}
}  // namespace

void Sensor::set_weights(const std::vector<double>& weights) {
  if (weights.size() != count) throw std::invalid_argument("one weight per texel / pixel is needed");
  std::vector<double> w = weights;
  normalise(w);
  const double total = std::accumulate(w.begin(), w.end(), 0.0);
  if (!(total > 0)) throw std::invalid_argument("the weights of the texels / pixels are all zero");
  prob.resize(count);
  cum_prob_.resize(count);
  double run = 0;
  for (std::size_t k = 0; k < count; ++k) {
    prob[k] = w[k] / total;
    cum_prob_[k] = (run += prob[k]);
  }
}

// ---------------------------------------------------------------------------- TexelSensor
TexelSensor::TexelSensor(const Scene& scene_in, const std::vector<int>& faces_in, int long_side) : scene(scene_in) {
  const Mesh& mesh = scene.mesh;
  for (int f : faces_in) {
    if (f < 0 || f >= static_cast<int>(mesh.faces())) throw std::invalid_argument("unknown face");
    if (mesh.face_area[f] > 0) faces.push_back(f);
  }
  if (faces.empty()) throw std::invalid_argument("no face to compute the irradiance on");
  if (long_side < 1) throw std::invalid_argument("textures need at least one texel");
  slot_.assign(mesh.faces(), -1);
  first.push_back(0);
  for (std::size_t k = 0; k < faces.size(); ++k) {
    const int f = faces[k];
    slot_[f] = static_cast<int>(k);
    const auto [tw, th] = mesh.texels(f, long_side);
    w.push_back(tw);
    h.push_back(th);
    first.push_back(first.back() + static_cast<long long>(tw) * th);
    const auto areas = mesh.texel_areas(f, tw, th);
    texel_area.insert(texel_area.end(), areas.begin(), areas.end());
    area += mesh.face_area[f];
    crossed = crossed || scene.face_virtual[f];
    // the largest area density in every texel (points are accepted against it)
    const FaceGrid& g = mesh.grids[f];
    const auto cells = mesh.cell_density(f);
    const long gu = static_cast<long>(g.ug.size()) - 1, gv = static_cast<long>(g.vg.size()) - 1;
    auto range = [](const std::vector<double>& lines, int n, int k, long& a, long& b) {
      const long last = static_cast<long>(lines.size()) - 2;
      a = std::clamp<long>(std::upper_bound(lines.begin(), lines.end(), static_cast<double>(k) / n) - lines.begin() - 1, 0, last);
      b = std::clamp<long>(std::lower_bound(lines.begin(), lines.end(), static_cast<double>(k + 1) / n) - lines.begin() - 1, 0, last);
      b = std::max(a, b);
    };
    (void)gu;
    for (int j = 0; j < th; ++j)
      for (int i = 0; i < tw; ++i) {
        long u0, u1, v0, v1;
        range(g.ug, tw, i, u0, u1);
        range(g.vg, th, j, v0, v1);
        double best = 0;
        for (long a = u0; a <= u1; ++a)
          for (long b = v0; b <= v1; ++b) best = std::max(best, cells[a * gv + b]);
        texel_density.push_back(best);
      }
  }
  count = static_cast<std::size_t>(first.back());
  reset();
}

void TexelSensor::normalise(std::vector<double>& weights) const {
  for (std::size_t k = 0; k < count; ++k)
    if (!(texel_area[k] > 0)) weights[k] = 0;
}

void TexelSensor::reset() { set_weights(texel_area); }

void TexelSensor::adapt(const std::vector<double>& error, double floor) {
  std::vector<double> positive;
  for (double a : texel_area)
    if (a > 0) positive.push_back(a);
  std::nth_element(positive.begin(), positive.begin() + positive.size() / 2, positive.end());
  const double median = positive[positive.size() / 2];
  std::vector<double> want(count, 0.0);
  double sum = 0, total_area = 0;
  for (std::size_t k = 0; k < count; ++k) {
    total_area += texel_area[k];
    if (texel_area[k] > 0.1 * median) want[k] = error[k] / texel_area[k] * std::sqrt(prob[k]);
    if (!std::isfinite(want[k])) return;
    sum += want[k];
  }
  if (!(sum > 0)) return;
  for (std::size_t k = 0; k < count; ++k) want[k] = floor * texel_area[k] / total_area + (1 - floor) * want[k] / sum;
  set_weights(want);
}

long long TexelSensor::texel(long long tri, double u, double v) const {
  const int s = slot_[scene.mesh.face[tri]];
  if (s < 0) return -1;
  const int i = std::clamp(static_cast<int>(u * w[s]), 0, w[s] - 1), j = std::clamp(static_cast<int>(v * h[s]), 0, h[s] - 1);
  return first[s] + static_cast<long long>(j) * w[s] + i;
}

bool TexelSensor::point(double pick, Rng& rng, long long& tri, double& b1, double& b2, long long& tex) const {
  const Mesh& mesh = scene.mesh;
  tex = static_cast<long long>(pick_from(cum_prob_, pick));
  const int s = static_cast<int>(std::upper_bound(first.begin(), first.end(), tex) - first.begin()) - 1;
  const int f = faces[s];
  const long long local = tex - first[s];
  const int i = static_cast<int>(local % w[s]), j = static_cast<int>(local / w[s]);
  for (int attempt = 0; attempt < 256; ++attempt) {  // evenly in the texel's (u, v), kept where the face is,
    const double u = (i + rng.next()) / w[s], v = (j + rng.next()) / h[s];  // by the area there
    const double keep = rng.next();
    double density;
    tri = mesh.locate(f, u, v, b1, b2, density);
    if (tri >= 0 && keep * texel_density[tex] < density && mesh.grids[f].inside(u, v)) return true;
  }
  return false;  // (texels with hardly any area)
}

SensorSample TexelSensor::sample(double pick, Rng& rng) const {
  const Mesh& mesh = scene.mesh;
  SensorSample out;
  long long tri, tex;
  double b1, b2;
  out.exists = point(pick, rng, tri, b1, b2, tex);
  if (!out.exists) return out;
  out.texel = tex;
  out.pdf_pos = pdf_pos_at(tex);
  out.p = mesh.point(tri, b1, b2);
  double u, v;
  mesh.shade(tri, b1, b2, out.ns, u, v);
  out.ng = mesh.ng[tri];
  const double side = rng.next() < 0.5 ? 1.0 : -1.0;
  const double cos = std::sqrt(1 - rng.next());
  out.d = around(out.ns * side, cos, 2 * kPi * rng.next());
  out.ok = cos > 0 && dot(out.ng, out.d) * dot(out.ns, out.d) > 0;
  out.pdf_dir = cos / (2 * kPi);
  out.beta = 2 * kPi / out.pdf_pos;
  return out;
}

void TexelSensor::response(const Vec3& ns, const Vec3& w_, const Vec3&, double& importance, double& pdf,
                           long long& pixel) const {
  importance = 1.0;
  pdf = std::abs(dot(ns, w_)) / (2 * kPi);
  pixel = -2;  // the point's own texel
}

// ---------------------------------------------------------------------------- Camera
Camera::Camera(const Vec3& position_, const Vec3& target, const Vec3& up_, double fov, int width_, int height_,
               double lens_radius_, double focus_)
    : position(position_), width(width_), height(height_), lens_radius(lens_radius_) {
  Vec3 f = target - position;
  if (norm(f) < 1e-12) throw std::invalid_argument("camera: position and target are the same point");
  if (lens_radius < 0 || focus_ < 0) throw std::invalid_argument("camera: the lens radius and the focus distance must not be negative");
  if (width < 1 || height < 1) throw std::invalid_argument("camera: the image needs pixels");
  focus = focus_ > 0 ? focus_ : norm(f);
  forward = normalized(f);
  const Vec3 r = cross(forward, up_);
  if (norm(r) < 1e-9) throw std::invalid_argument("camera: the view direction is parallel to the up direction");
  if (!(fov > 0 && fov < 180)) throw std::invalid_argument("camera: field of view must lie in 0 … 180°");
  right = normalized(r);
  up = cross(right, forward);
  tan_x = std::tan(fov * kPi / 360.0);
  tan_y = tan_x * height / width;
  film = 4 * tan_x * tan_y;
  lens_area = kPi * lens_radius * lens_radius;
  camera = true;
  pinhole = lens_radius == 0;
  count = static_cast<std::size_t>(width) * height;
  reset();
}

void Camera::reset() { set_weights(std::vector<double>(count, 1.0)); }

void Camera::adapt(const std::vector<double>& error, double floor) {
  std::vector<double> want(count);
  double sum = 0;
  for (std::size_t k = 0; k < count; ++k) {
    want[k] = error[k] * std::sqrt(prob[k]);
    if (!std::isfinite(want[k])) return;
    sum += want[k];
  }
  if (!(sum > 0)) return;
  for (std::size_t k = 0; k < count; ++k) want[k] = floor / static_cast<double>(count) + (1 - floor) * want[k] / sum;
  set_weights(want);
}

long long Camera::pixel_of(double x, double y) const {
  const double i = std::floor((x / tan_x + 1) * 0.5 * width), j = std::floor((y / tan_y + 1) * 0.5 * height);
  if (i < 0 || i >= width || j < 0 || j >= height) return -1;
  return static_cast<long long>(j) * width + static_cast<long long>(i);
}

SensorSample Camera::sample(double pick, Rng& rng) const {
  SensorSample out;
  const long long pixel = static_cast<long long>(pick_from(cum_prob_, pick));
  const double x = (2 * (pixel % width + rng.next()) / width - 1) * tan_x;
  const double y = (2 * (pixel / width + rng.next()) / height - 1) * tan_y;
  Vec3 d = forward + right * x + up * y;
  const double length = norm(d);
  d = d / length;
  double cos = 1.0 / length;
  const double share = prob[pixel] * static_cast<double>(count);  // of the rays, against an even spread
  out.texel = pixel;
  out.ok = out.exists = true;
  out.beta = static_cast<double>(count) / share;
  if (pinhole) {
    out.p = position;
    out.d = d;
    out.point = true;
    out.pdf_pos = 1.0;
  } else {  // a point of the lens, and from it through the point in focus that the pinhole's ray meets
    const Vec3 sharp = position + d * (focus * length);
    const double r = lens_radius * std::sqrt(rng.next()), phi = 2 * kPi * rng.next();
    out.p = position + right * (r * std::cos(phi)) + up * (r * std::sin(phi));
    out.d = normalized(sharp - out.p);
    cos = dot(out.d, forward);
    out.ns = out.ng = forward;
    out.pdf_pos = 1.0 / lens_area;
  }
  out.pdf_dir = share / (film * cos * cos * cos);
  return out;
}

void Camera::response(const Vec3&, const Vec3& w_, const Vec3& p, double& importance, double& pdf,
                      long long& pixel) const {
  importance = pdf = 0;
  pixel = -1;
  const double c = dot(w_, forward);
  if (!(c > 1e-9)) return;
  double x = dot(w_, right) / c, y = dot(w_, up) / c;
  if (!pinhole) {  // the pixel: where the ray meets the plane in focus
    const Vec3 off = p - position;
    x += dot(off, right) / focus;
    y += dot(off, up) / focus;
  }
  pixel = pixel_of(x, y);
  if (pixel < 0) return;
  const double even = 1.0 / (film * c * c * c);
  importance = pinhole ? even * static_cast<double>(count) : even * static_cast<double>(count) / (lens_area * c);
  pdf = even * prob[pixel] * static_cast<double>(count);
}

}  // namespace pt
