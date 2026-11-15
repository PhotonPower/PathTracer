// Photon mapping for camera images (progressive, after Knaus and Zwicker 2011; see pt_photon.py).
// Light paths leave their flux on every surface that is not specular; camera paths are followed
// through mirrors and glass to the first other surface, and the light it sends on is estimated
// from the photons within a radius that shrinks from pass to pass.
#include <algorithm>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

#include "integrator.hpp"

namespace pt {
namespace {

using namespace detail;

constexpr double kAlpha = 0.7;  // radius reduction: r²(i+1) = r²(i) (i + alpha) / (i + 1)
constexpr double kPhotons = 30;  // photons expected within the first radius when it is chosen

struct Photon {
  Vec3 p, wo, ns;
  double beta;
  int k, channel;
};

struct Seen {  // the point a camera path sees: its last vertex, if it is not specular
  Vec3 p, ns, wo;
  int m, t, channel;
  double beta, outside;
  long long slot;
};

/// Photons in a grid of cells as wide as the search diameter.
class Grid {
 public:
  Grid(const std::vector<Photon>& photons, double radius) : photons_(photons), radius_(radius), cell_(2 * radius) {
    lo_ = {kInf, kInf, kInf};
    for (const Photon& ph : photons) lo_ = {std::min(lo_.x, ph.p.x), std::min(lo_.y, ph.p.y), std::min(lo_.z, ph.p.z)};
    lo_ = lo_ - Vec3{2 * cell_, 2 * cell_, 2 * cell_};
    order_.resize(photons.size());
    std::iota(order_.begin(), order_.end(), 0);
    std::vector<std::uint64_t> key(photons.size());
    for (std::size_t i = 0; i < photons.size(); ++i) key[i] = key_of(cells(photons[i].p));
    std::sort(order_.begin(), order_.end(), [&](int a, int b) { return key[a] < key[b]; });
    keys_.resize(photons.size());
    for (std::size_t i = 0; i < photons.size(); ++i) keys_[i] = key[order_[i]];
  }
  /// Calls f(photon index) for every photon closer than the radius to x.
  template <class F>
  void near(const Vec3& x, F&& f) const {
    const auto base = cells(x - Vec3{radius_, radius_, radius_});
    for (int dx = 0; dx < 2; ++dx)
      for (int dy = 0; dy < 2; ++dy)
        for (int dz = 0; dz < 2; ++dz) {
          const std::uint64_t key = key_of({base[0] + dx, base[1] + dy, base[2] + dz});
          auto [a, b] = std::equal_range(keys_.begin(), keys_.end(), key);
          for (auto it = a; it != b; ++it) {
            const int i = order_[it - keys_.begin()];
            const Vec3 gap = photons_[i].p - x;
            if (dot(gap, gap) < radius_ * radius_) f(i);
          }
        }
  }

 private:
  std::array<std::uint64_t, 3> cells(const Vec3& p) const {
    auto one = [&](double v, double lo) {
      return static_cast<std::uint64_t>(std::clamp(std::floor((v - lo) / cell_), 0.0, 1048575.0));
    };
    return {one(p.x, lo_.x), one(p.y, lo_.y), one(p.z, lo_.z)};
  }
  static std::uint64_t key_of(const std::array<std::uint64_t, 3>& c) { return c[0] + (c[1] << 21) + (c[2] << 42); }
  const std::vector<Photon>& photons_;
  double radius_, cell_;
  Vec3 lo_;
  std::vector<int> order_;
  std::vector<std::uint64_t> keys_;
};

/// Squared radius in which about kPhotons photons lie around the points the camera sees.
double first_radius2(const Scene& sc, const std::vector<Photon>& photons, const std::vector<Seen>& seen) {
  double area = 0;
  for (std::size_t f = 0; f < sc.mesh.faces(); ++f)
    if (!sc.face_virtual[f] && !sc.materials.specular(sc.face_material[f])) area += sc.mesh.face_area[f];
  double r2 = kPhotons * std::max(area, 1e-300) / (kPi * static_cast<double>(std::max<std::size_t>(photons.size(), 1)));
  const std::size_t step = std::max<std::size_t>(1, seen.size() / 1000);
  for (int it = 0; it < 6; ++it) {
    const Grid grid(photons, std::sqrt(r2));
    double found = 0, some = 0;
    for (std::size_t i = 0; i < seen.size(); i += step) {
      grid.near(seen[i].p, [&](int) { found += 1; });
      some += 1;
    }
    found /= std::max(some, 1.0);
    if (0.6 * kPhotons <= found && found <= 1.6 * kPhotons) break;
    r2 *= std::clamp(kPhotons / std::max(found, 0.25), 1.0 / 64, 64.0);  // the number grows with r²
  }
  return r2;
}

}  // namespace

Result render_photon(const Scene& scene, Sensor& camera, const RenderOptions& opt) {
  if (!camera.camera) throw std::invalid_argument("photon mapping is for camera images");
  const Materials& mats = scene.materials;
  const int depth = opt.depth;
  const int threads = opt.threads > 0 ? opt.threads : std::max(1u, std::thread::hardware_concurrency());
  const long long pass = opt.batch > 0 ? opt.batch : std::clamp<long long>(opt.samples / 16, 5000, 100000);
  const std::size_t channels = scene.wavelengths.size(), size = channels * camera.count;
  std::vector<double> accum(size, 0.0);
  Spread spread(camera.count, channels);
  Result out;
  out.threads = threads;
  double r2 = opt.radius > 0 ? opt.radius * opt.radius : -1.0, last_r2 = 0;
  long long done = 0;
  std::vector<Scratch> scratch(threads);
  for (auto& s : scratch) s.accum.assign(size, 0.0);
  std::mutex merge;
  for (std::uint64_t index = 0; done < opt.samples; ++index) {
    const long long count = std::min(pass, opt.samples - done);
    Rng order(opt.seed, index, 0xffffffffULL);
    const auto source_pick = stratified(count, order), sensor_pick = stratified(count, order);
    std::vector<Photon> photons;
    std::vector<Seen> seen;
    std::vector<int> path_channel(static_cast<std::size_t>(count));
    parallel_for(count, threads, [&](long long a, long long b, int id) {
      Scratch& sx = scratch[id];
      std::vector<Photon> mine;
      std::vector<Seen> mine_seen;
      for (long long i = a; i < b; ++i) {
        Rng rng(opt.seed, index, static_cast<std::uint64_t>(i));
        Path& L = sx.light;
        light_subpath(scene, &camera, L, depth, source_pick[i], rng, false);
        path_channel[i] = L.channel;
        for (int k = 1; k < L.len; ++k)  // photons: the vertices on surfaces that are not specular
          if (!L.v[k].spec) mine.push_back({L.v[k].p, L.v[k].wo, L.v[k].ns, L.v[k].beta, k, L.channel});
        Path& E = sx.eye;
        eye_subpath(scene, camera, E, depth + 1, sensor_pick[i], rng, L.channel, true);
        sx.rays += L.rays + E.rays;
        if (E.len < 1 || E.v[0].tex < 0) continue;
        const long long slot = static_cast<long long>(L.channel) * static_cast<long long>(camera.count) + E.v[0].tex;
        // sources seen directly, in mirrors or through glass (every vertex before the last is specular)
        for (int k = 1; k < E.len; ++k)
          if (E.v[k].src >= 0) {
            double le, dens;
            scene.emitted(E.v[k].src, E.v[k].ns, E.v[k].wo, le, dens);
            sx.accum[slot] += E.v[k].beta * le * scene.spectral_weight(E.v[k].src, L.channel);
          }
        const int t = E.len - 1;
        if (t >= 1 && !E.v[t].spec)
          mine_seen.push_back({E.v[t].p, E.v[t].ns, E.v[t].wo, E.v[t].m, t, L.channel, E.v[t].beta, E.v[t].outside, slot});
      }
      const std::lock_guard<std::mutex> lock(merge);
      photons.insert(photons.end(), mine.begin(), mine.end());
      seen.insert(seen.end(), mine_seen.begin(), mine_seen.end());
    });
    if (!seen.empty() && !photons.empty()) {
      if (r2 < 0) r2 = first_radius2(scene, photons, seen);
      const Grid grid(photons, std::sqrt(r2));
      const double scale = 1.0 / (kPi * r2 * static_cast<double>(count));
      parallel_for(static_cast<long long>(seen.size()), threads, [&](long long a, long long b, int id) {
        Scratch& sx = scratch[id];
        for (long long i = a; i < b; ++i) {
          const Seen& x = seen[i];
          double total = 0;
          grid.near(x.p, [&](int j) {  // only photons of the camera path's wavelength
            const Photon& ph = photons[j];
            if (ph.k + x.t > depth || ph.channel != x.channel || !(std::abs(dot(ph.ns, x.ns)) > 0.5)) return;
            double f, p_fwd, p_rev;
            mats.eval(x.m, x.ns, x.wo, ph.wo, x.channel, true, x.outside, f, p_fwd, p_rev);
            total += f * ph.beta;
          });
          // the path stands for all wavelengths
          sx.accum[x.slot] += x.beta * total / scene.channel_prob[x.channel] * scale;
        }
      });
    }
    for (auto& s : scratch) {
      for (std::size_t k = 0; k < size; ++k) accum[k] += s.accum[k];
      std::fill(s.accum.begin(), s.accum.end(), 0.0);
      out.rays += s.rays;
      s.rays = 0;
    }
    done += count;
    spread.add(accum, count);
    ++out.passes;
    if (r2 > 0) {
      last_r2 = r2;
      r2 = r2 * (out.passes + kAlpha) / (out.passes + 1);
    }
    if (opt.adaptive && spread.batches >= 2 && done < opt.samples) camera.adapt(spread.error(done));
    if (opt.progress) opt.progress(static_cast<double>(done) / static_cast<double>(opt.samples));
    if (opt.cancelled && opt.cancelled()) break;
  }
  out.samples = done;
  out.radius = std::sqrt(last_r2);
  out.channels.resize(size);
  out.values.assign(camera.count, 0.0);
  for (std::size_t c = 0; c < channels; ++c)
    for (std::size_t k = 0; k < camera.count; ++k) {
      out.channels[c * camera.count + k] = accum[c * camera.count + k] / static_cast<double>(std::max<long long>(done, 1));
      out.values[k] += out.channels[c * camera.count + k];
    }
  out.error = spread.error(done);
  return out;
}

}  // namespace pt
