// Internals shared by the integrator and photon mapping.
#pragma once

#include "pathtracer/pathtracer.hpp"

namespace pt {
namespace detail {

constexpr int kMedia = 4;  // glass bodies a path can be inside of at once

/// A vertex of a subpath. beta: weight of the subpath on arrival; pf: density (per area) with
/// which the subpath made the vertex; pr: density with which the opposite direction would make
/// it (known once the next vertex has scattered); wo: unit direction to the previous vertex.
struct Vertex {
  Vec3 p, ns, ng, wo;
  int m = 0;
  long long tex = -1;
  int src = -1;
  double beta = 0, pf = 0, pr = 0, outside = 1;
  bool spec = false;
};

struct Crossing {
  int k = 0;  // the vertex the subpath would end with here
  Vertex v;
};

struct Path {
  std::vector<Vertex> v;
  int len = 0;          // valid vertices
  bool point = false;   // vertex 0 is a point (no surface)
  int channel = 0;      // number of the path's wavelength
  int escape = -1;      // vertex whose ray left the scene
  Vec3 escape_dir;
  std::vector<Crossing> cross;
  long long rays = 0;
  void reset(int k) {
    if (static_cast<int>(v.size()) < k) v.resize(k);
    len = 0;
    point = false;
    escape = -1;
    cross.clear();
    rays = 0;
  }
  /// |cos| between the normal of vertex k and w; 1 for points.
  double cos(int k, const Vec3& w) const { return k == 0 && point ? 1.0 : std::abs(dot(v[k].ns, w)); }
};

/// Per thread: paths, the sums of a batch, work space of the weights.
struct Scratch {
  Path light, eye;
  std::vector<double> accum;
  std::vector<double> pf, pr;
  std::vector<std::uint8_t> spec;
  long long rays = 0;
};

struct Spread {  // the standard error of the estimate from the spread of its batches
  std::size_t count, channels;
  std::vector<double> before, squares;
  int batches = 0;
  Spread(std::size_t n, std::size_t c) : count(n), channels(c), before(n, 0.0), squares(n, 0.0) {}
  void add(const std::vector<double>& accum, long long n);
  std::vector<double> error(long long done) const;
};

void leave(int* media, int& held, int body);
void walk(const Scene& sc, const Sensor* sensor, Path& path, Vec3 d, double beta, double pdf, bool alive, Rng& rng,
          bool radiance, bool stop_rough, int kmax, bool record_cross);
void light_subpath(const Scene& sc, const Sensor* sensor, Path& path, int k, double pick, Rng& rng, bool record_cross);
void eye_subpath(const Scene& sc, const Sensor& sensor, Path& path, int k, double pick, Rng& rng, int channel,
                 bool stop_rough);
void contribute(const Scene& sc, const Sensor& sensor, const Path& L, const Path* E, int s, int t,
                const std::uint8_t* ways, int n_ways, Scratch& scratch);
void detector_deposits(const Scene& sc, const Sensor& sensor, Path& L, const Path* E, Method method, int depth,
                       const std::vector<std::uint8_t>& ways, const std::vector<int>& n_ways, Scratch& scratch);
void parallel_for(long long count, int threads, const std::function<void(long long, long long, int)>& body);
std::vector<double> stratified(long long n, Rng& rng);

}  // namespace detail

Result render_photon(const Scene& scene, Sensor& camera, const RenderOptions& options);

}  // namespace pt
