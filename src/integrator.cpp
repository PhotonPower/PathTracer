// Light transport: light tracing, path tracing and bidirectional path tracing as one estimator
// (see the Python kernel: pt_integrator.py). A path x0 … x(n-1) from a source to the sensor is
// made from s vertices from the source and t = n - s from the sensor; the methods differ in the
// (s, t) they use, and ways that make the same path are combined with the balance heuristic.
#include "pathtracer/pathtracer.hpp"

#include <algorithm>
#include <atomic>
#include <numeric>
#include <stdexcept>
#include <thread>

#include "integrator.hpp"

namespace pt {
namespace detail {

void leave(int* media, int& held, int body) {
  for (int k = 0; k < held; ++k)
    if (media[k] == body) {
      for (int c = k; c + 1 < held; ++c) media[c] = media[c + 1];
      media[--held] = -1;
      return;
    }
}

/// Where the ray towards vertex k crosses detector faces of the sensor before it arrives.
static void crossings(const Scene& sc, const Sensor& sensor, Path& path, int k, const Vec3& o, const Vec3& d,
                      double tmax, double pdf, double beta) {
  const Mesh& mesh = sc.mesh;
  double done = 0;
  for (int step = 0; step < 64; ++step) {
    const Hit h = sc.crossing(o + d * done, d, tmax - done);
    ++path.rays;
    if (h.tri < 0) break;
    const double t = h.t + done;
    done = t + sc.eps;
    Vec3 ns;
    double u, v;
    mesh.shade(h.tri, h.b1, h.b2, ns, u, v);
    const double cs = dot(ns, d);
    const long long tex = sensor.texel(h.tri, u, v);
    if (tex < 0 || !(cs * dot(mesh.ng[h.tri], d) > 0)) continue;
    Vertex c;
    c.p = o + d * t;
    c.ns = ns;
    c.ng = mesh.ng[h.tri];
    c.wo = -d;
    c.tex = tex;
    c.beta = beta;
    const Vec3 gap = c.p - path.v[k - 1].p;
    c.pf = pdf * std::abs(cs) / dot(gap, gap);
    path.cross.push_back({k, c});
  }
}

void walk(const Scene& sc, const Sensor* sensor, Path& path, Vec3 d, double beta, double pdf, bool alive, Rng& rng,
          bool radiance, bool stop_rough, int kmax, bool record_cross) {
  const Mesh& mesh = sc.mesh;
  const Materials& mats = sc.materials;
  const int ch = path.channel;
  int media[kMedia] = {-1, -1, -1, -1};  // the glass bodies the path is inside, as entered
  int held = 0, ghost = -1;
  for (int k = 1; k < kmax && alive; ++k) {
    Vertex& prev = path.v[k - 1];
    const Vec3 o = sc.offset(prev.p, prev.ng, d);
    const Hit h = sc.hit(o, d);
    ++path.rays;
    if (record_cross) crossings(sc, *sensor, path, k, o, d, h.t, pdf, beta);
    if (h.tri < 0) {
      path.escape = k - 1;
      path.escape_dir = d;
      break;
    }
    Vec3 ns;
    double u, v;
    if (sc.exact)
      mesh.shade_exact(h.tri, h.b1, h.b2, o, d, h.t, ns, u, v);
    else
      mesh.shade(h.tri, h.b1, h.b2, ns, u, v);
    const Vec3 ng = mesh.ng[h.tri];
    const double cs = -dot(ns, d);
    if (!(cs * -dot(ng, d) > 0)) break;  // both normals must see the ray on the same side
    const Vec3 p = o + d * h.t;
    const double dist2 = dot(p - prev.p, p - prev.p);
    const int m = sc.tri_material[h.tri];
    const bool glassy = sc.tri_glassy[h.tri] != 0;
    const int top = held > 0 ? media[held - 1] : -1;
    // the segment ran through the body entered last (a path that began inside one: the body it leaves)
    if (top >= 0 || (glassy && cs < 0)) {
      const double a = mats.absorbs(top >= 0 ? sc.entity_material[top] : m, ch);
      if (a > 0) beta *= std::exp(-a * std::sqrt(dist2));
    }
    double outside = 1.0;
    bool straight = false;
    int contact = -1, ending = -1;
    const int body = sc.tri_entity[h.tri];
    const int passed = ghost;
    ghost = -1;
    if (glassy) {  // which medium lies in front of this surface?
      bool inside_own = false;
      for (int c = 0; c < held; ++c) inside_own = inside_own || media[c] == body;
      const bool on_top = top == body, enters = cs > 0;
      const int below = held >= 2 ? media[held - 2] : -1;
      straight = enters ? inside_own : ((inside_own && !on_top) || passed == body);
      int beyond = enters ? top : (on_top ? below : top);
      if (!enters && !straight && beyond < 0) {  // out into air – or into a body that touches?
        beyond = sc.touching(p, d, body, false);
        contact = beyond;
      }
      if (enters && !straight && top >= 0) ending = sc.touching(p, d, top, true);  // does the other end here?
      if (beyond >= 0) outside = mats.ior(sc.entity_material[beyond], ch);
      if (!enters && straight) leave(media, held, body);  // left a body that was not the innermost
    }
    Vertex& vx = path.v[k];
    vx = Vertex{};
    vx.p = p;
    vx.ns = ns;
    vx.ng = ng;
    vx.wo = -d;
    vx.m = m;
    vx.tex = sensor && !sensor->camera ? sensor->texel(h.tri, u, v) : -1;
    vx.src = sc.face_source[mesh.face[h.tri]];
    vx.beta = beta;
    vx.pf = pdf * std::abs(cs) / dist2;
    vx.outside = outside;
    vx.spec = mats.specular(m) || straight;
    path.len = k + 1;
    if (k == kmax - 1) break;
    Vec3 wi;
    double weight, pd;
    mats.sample(m, ns, -d, rng, radiance, ch, outside, straight, wi, weight, pd);
    double f, p_fwd, p_rev;
    mats.eval(m, ns, -d, wi, ch, true, outside, f, p_fwd, p_rev);
    prev.pr = p_rev * path.cos(k - 1, d) / dist2;
    bool ok = weight > 0 && dot(ns, wi) * dot(ng, wi) > 0;
    if (glassy && !straight && dot(ns, wi) * cs < 0) {  // the path went through the surface
      if (cs < 0) {
        leave(media, held, body);
        if (contact >= 0 && held < kMedia) media[held++] = contact;
      } else {
        if (ending >= 0) {  // the body it was in ended at this surface
          leave(media, held, ending);
          ghost = ending;
        }
        if (held < kMedia) media[held++] = body;
      }
    }
    if (stop_rough && !vx.spec) ok = false;
    if (k >= 3) {  // Russian roulette on dim paths
      const double q = vx.spec ? 1.0 : std::clamp(weight, 0.1, 1.0);
      if (!(rng.next() < q)) ok = false;
      weight /= q;
    }
    beta *= weight;
    d = wi;
    pdf = pd;
    alive = ok;
  }
}

void light_subpath(const Scene& sc, const Sensor* sensor, Path& path, int k, double pick, Rng& rng,
                   bool record_cross) {
  path.reset(std::max(k, 1));
  const Emission e = sc.emit(pick, rng);
  Vertex& v0 = path.v[0];
  v0 = Vertex{};
  v0.p = e.p;
  v0.ns = e.ns;
  v0.ng = e.ng;
  v0.src = e.src;
  v0.beta = 1.0 / e.pdf_pos;
  v0.pf = e.pdf_pos * sc.spectral_weight(e.src, e.channel);  // point and wavelength
  path.point = e.point;
  path.channel = e.channel;
  path.len = 1;
  walk(sc, sensor, path, e.d, sc.flux, e.pdf_dir, e.ok, rng, false, false, k, record_cross);
}

void eye_subpath(const Scene& sc, const Sensor& sensor, Path& path, int k, double pick, Rng& rng, int channel,
                 bool stop_rough) {
  path.reset(std::max(k, 1));
  path.channel = channel;
  const SensorSample e = sensor.sample(pick, rng);
  if (!e.exists) return;
  Vertex& v0 = path.v[0];
  v0 = Vertex{};
  v0.p = e.p;
  v0.ns = e.ns;
  v0.ng = e.ng;
  v0.tex = e.texel;
  v0.beta = 1.0 / e.pdf_pos;
  v0.pf = e.pdf_pos;
  path.point = e.point;
  path.len = 1;
  walk(sc, &sensor, path, e.d, e.beta, e.pdf_dir, e.ok, rng, true, stop_rough, k, false);
}

namespace {

double gap2(const Path& path, int k) {
  const Vec3 g = path.v[k].p - path.v[k - 1].p;
  return dot(g, g);
}

struct Connection {
  double dist2 = 0, dens_y = 0, dens_z = 0, dens_y_rev = 0, dens_z_rev = 0, cy = 0, cz = 0;
};

/// Balance-heuristic weight of the way (s, t) among the ways `ways[sp]` of making the path.
double mis_weight(const Scene& sc, const Sensor& sensor, const Path& L, const Path* E, int s, int t,
                  const Connection& con, const std::uint8_t* ways, std::vector<double>& pf, std::vector<double>& pr,
                  std::vector<std::uint8_t>& spec) {
  const int n = s + t;
  const int ch = L.channel;
  pf.assign(n, 0.0);
  pr.assign(n, 0.0);
  spec.assign(n, 0);
  for (int i = 0; i < n; ++i) {
    if (i < s)
      pf[i] = L.v[i].pf;
    else
      pr[i] = E->v[n - 1 - i].pf;
  }
  for (int i = 0; i < s - 2; ++i) pr[i] = L.v[i].pr;
  for (int i = s + 2; i < n; ++i) pf[i] = E->v[n - 1 - i].pr;
  if (s >= 1 && t >= 1) {
    pf[s] = con.dens_y * con.cz / con.dist2;
    pr[s - 1] = con.dens_z * con.cy / con.dist2;
    if (s >= 2) pr[s - 2] = con.dens_y_rev * L.cos(s - 2, L.v[s - 1].wo) / gap2(L, s - 1);
    if (t >= 2) pf[s + 1] = con.dens_z_rev * E->cos(t - 2, E->v[t - 1].wo) / gap2(*E, t - 1);
  } else if (s == 0) {
    const Vertex& z = E->v[t - 1];
    pf[0] = sc.emit_pdf_pos(z.src) * sc.spectral_weight(z.src, ch);
    double le, dens;
    sc.emitted(z.src, z.ns, z.wo, le, dens);
    pf[1] = dens * E->cos(t - 2, z.wo) / gap2(*E, t - 1);
  } else {
    const Vertex& y = L.v[s - 1];
    pr[n - 1] = sensor.pdf_pos_at(y.tex);
    double importance, dens;
    long long pixel;
    sensor.response(y.ns, y.wo, y.p, importance, dens, pixel);
    pr[n - 2] = dens * L.cos(s - 2, y.wo) / gap2(L, s - 1);
  }
  // specular vertices: their directions are certain – the densities cancel, and no connection ends on them
  for (int i = 1; i < n - 1; ++i) {
    spec[i] = i < s ? L.v[i].spec : E->v[n - 1 - i].spec;
    if (spec[i]) {
      pf[i + 1] = 1.0;
      pr[i - 1] = 1.0;
    }
  }
  const bool light_point = s >= 1 && L.point;
  const bool no_image = sensor.camera && n == 2 && sensor.pinhole && light_point;
  auto possible = [&](int sp) {
    if (sp == 0) return !light_point;
    if (sp == n) return true;
    return !spec[sp - 1] && !spec[sp] && !no_image;
  };
  auto step = [](double r, double num, double den) {
    const double v = r * num / den;
    return std::isnan(v) ? 0.0 : std::min(v, 1e300);
  };
  double total = 1.0, r = 1.0;
  for (int sp = s + 1; sp <= n; ++sp) {
    r = step(r, pf[sp - 1], pr[sp - 1]);
    if (ways[sp] && possible(sp)) total += r;
  }
  r = 1.0;
  for (int sp = s - 1; sp >= 0; --sp) {
    r = step(r, pr[sp], pf[sp]);
    if (ways[sp] && possible(sp)) total += r;
  }
  return 1.0 / total;
}

}  // namespace

void contribute(const Scene& sc, const Sensor& sensor, const Path& L, const Path* E, int s, int t,
                const std::uint8_t* ways, int n_ways, Scratch& scratch) {
  const int n = s + t;
  if (s > L.len || t > (E ? E->len : 0)) return;
  const Materials& mats = sc.materials;
  const int ch = L.channel;
  double c = 0;
  long long tex = -1;
  Connection con;
  if (s == 0) {  // the eye subpath hit a source
    const Vertex& z = E->v[t - 1];
    if (z.src < 0) return;
    double le, dens;
    sc.emitted(z.src, z.ns, z.wo, le, dens);
    c = z.beta * le * sc.spectral_weight(z.src, ch);
    tex = E->v[0].tex;
  } else if (t == 0) {  // the light subpath hit (or crossed) a sensor face
    const Vertex& y = L.v[s - 1];
    if (y.tex < 0) return;
    c = y.beta;
    tex = y.tex;
  } else {  // connection y -> z
    const Vertex &y = L.v[s - 1], &z = E->v[t - 1];
    Vec3 w = z.p - y.p;
    con.dist2 = dot(w, w);
    if (!(con.dist2 > 1e-24)) return;
    w = w / std::sqrt(con.dist2);
    double fy, fz;
    if (s == 1)
      sc.emitted(y.src, y.ns, w, fy, con.dens_y);
    else
      mats.eval(y.m, y.ns, y.wo, w, ch, false, y.outside, fy, con.dens_y, con.dens_y_rev);
    if (t == 1) {
      long long pixel;
      sensor.response(z.ns, -w, z.p, fz, con.dens_z, pixel);
      tex = pixel == -2 ? z.tex : pixel;
    } else {
      mats.eval(z.m, z.ns, z.wo, -w, ch, true, z.outside, fz, con.dens_z, con.dens_z_rev);
      tex = E->v[0].tex;
    }
    if (tex < 0) return;
    con.cy = L.cos(s - 1, w);
    con.cz = E->cos(t - 1, w);
    c = y.beta * fy * con.cy * con.cz / con.dist2 * fz * z.beta;
    // both normals of a surface vertex must see the connection on the same side
    const bool side_y = (s == 1 && L.point) || dot(y.ns, w) * dot(y.ng, w) > 0;
    const bool side_z = (t == 1 && E->point) || dot(z.ns, w) * dot(z.ng, w) > 0;
    if (!side_y || !side_z || !(c > 0)) return;
    if (sensor.camera && n == 2 && sensor.pinhole && L.point) return;  // a point source has no image in a pinhole
    ++scratch.rays;
    if (sc.occluded(sc.offset(y.p, y.ng, w), sc.offset(z.p, z.ng, -w))) return;
  }
  if (!(c > 0) || tex < 0 || !std::isfinite(c)) return;
  if (n_ways > 1) c *= mis_weight(sc, sensor, L, E, s, t, con, ways, scratch.pf, scratch.pr, scratch.spec);
  scratch.accum[static_cast<std::size_t>(ch) * sensor.count + tex] += c;
}

void detector_deposits(const Scene& sc, const Sensor& sensor, Path& L, const Path* E, Method method, int depth,
                       const std::vector<std::uint8_t>& ways, const std::vector<int>& n_ways, Scratch& scratch) {
  // every crossing is the end of a path made with t = 0, like a hit of a sensor face – only that
  // the subpath goes on behind it
  const int len = L.len;
  for (const Crossing& c : L.cross) {
    const int s = c.k + 1;
    if (s > depth + 1 || !enabled(method, false, s, 0)) continue;
    const Vertex saved = L.v[c.k];
    L.v[c.k] = c.v;
    L.len = s;
    contribute(sc, sensor, L, E, s, 0, &ways[static_cast<std::size_t>(s) * (depth + 3)], n_ways[s], scratch);
    L.v[c.k] = saved;
    L.len = len;
  }
}

void parallel_for(long long count, int threads, const std::function<void(long long, long long, int)>& body) {
  const long long chunk = 512;
  std::atomic<long long> next{0};
  auto work = [&](int id) {
    for (;;) {
      const long long a = next.fetch_add(chunk);
      if (a >= count) return;
      body(a, std::min(count, a + chunk), id);
    }
  };
  if (threads <= 1) {
    work(0);
    return;
  }
  std::vector<std::thread> pool;
  for (int id = 1; id < threads; ++id) pool.emplace_back(work, id);
  work(0);
  for (auto& th : pool) th.join();
}

std::vector<double> stratified(long long n, Rng& rng) {
  std::vector<double> out(static_cast<std::size_t>(n));
  for (long long i = 0; i < n; ++i) out[i] = (static_cast<double>(i) + rng.next()) / static_cast<double>(n);
  for (long long i = n - 1; i > 0; --i) std::swap(out[i], out[static_cast<long long>(rng.next() * static_cast<double>(i + 1))]);
  return out;
}

void Spread::add(const std::vector<double>& accum, long long n) {
  for (std::size_t k = 0; k < count; ++k) {
    double total = 0;
    for (std::size_t c = 0; c < channels; ++c) total += accum[c * count + k];
    const double diff = total - before[k];
    squares[k] += diff * diff / static_cast<double>(n);
    before[k] = total;
  }
  ++batches;
}

std::vector<double> Spread::error(long long done) const {
  std::vector<double> out;
  if (batches < 2 || done < 1) return out;
  out.resize(count);
  for (std::size_t k = 0; k < count; ++k) {
    const double mean = before[k] / static_cast<double>(done);
    out[k] = std::sqrt(std::max(0.0, (squares[k] / static_cast<double>(done) - mean * mean) / (batches - 1)));
  }
  return out;
}

}  // namespace detail

using namespace detail;

bool enabled(Method method, bool camera, int s, int t) {
  if (s + t < 2 || (camera && t == 0)) return false;
  if (method == kLight || method == kPhoton) return t == (camera ? 1 : 0);
  if (method == kPath) return s <= 1 && t >= (camera ? 2 : 1);
  return true;
}

Result render(const Scene& scene, Sensor& sensor, const RenderOptions& opt) {
  if (opt.depth < 1) throw std::invalid_argument("paths need at least one segment");
  if (opt.depth > 120) throw std::invalid_argument("paths of more than 120 segments are not supported");
  if (opt.samples < 1) throw std::invalid_argument("at least one sample is needed");
  sensor.reset();
  if (!opt.weights.empty()) sensor.set_weights(opt.weights);
  Method method = opt.method;
  const bool camera = sensor.camera;
  if (method == kPhoton) {
    if (camera) return render_photon(scene, sensor, opt);
    method = kLight;  // on faces the texels already collect the photons
  }
  const int depth = opt.depth;
  const int k_light = method == kPath ? 1 : (camera ? depth : depth + 1);
  const int k_eye = method == kLight ? (camera ? 1 : 0) : depth + 1;
  const long long batch = opt.batch > 0 ? opt.batch : std::clamp<long long>(opt.samples / 32, 10000, 250000);
  const int threads = opt.threads > 0 ? opt.threads : std::max(1u, std::thread::hardware_concurrency());
  const bool adapts = method == kPath || method == kBidir;
  const bool record_cross = !camera && sensor.crossed && method != kPath;
  // ways[n][sp]: whether the method makes paths of n vertices with sp light vertices
  const int stride = depth + 3;
  std::vector<std::uint8_t> ways(static_cast<std::size_t>(depth + 2) * stride, 0);
  std::vector<int> n_ways(depth + 2, 0);
  for (int n = 2; n <= depth + 1; ++n)
    for (int sp = 0; sp <= n; ++sp)
      if (enabled(method, camera, sp, n - sp)) {
        ways[static_cast<std::size_t>(n) * stride + sp] = 1;
        ++n_ways[n];
      }
  const std::size_t channels = scene.wavelengths.size(), size = channels * sensor.count;
  std::vector<double> accum(size, 0.0);
  std::vector<Scratch> scratch(threads);
  for (auto& s : scratch) s.accum.assign(size, 0.0);
  Spread spread(sensor.count, channels);
  Result out;
  out.threads = threads;
  long long done = 0;
  for (std::uint64_t index = 0; done < opt.samples; ++index) {
    const long long n = std::min(batch, opt.samples - done);
    Rng order(opt.seed, index, 0xffffffffULL);
    const auto source_pick = stratified(n, order), sensor_pick = stratified(n, order);
    parallel_for(n, threads, [&](long long a, long long b, int id) {
      Scratch& sx = scratch[id];
      Path& L = sx.light;
      Path& E = sx.eye;
      for (long long i = a; i < b; ++i) {
        Rng rng(opt.seed, index, static_cast<std::uint64_t>(i));
        light_subpath(scene, &sensor, L, k_light, source_pick[i], rng, record_cross);
        if (k_eye) eye_subpath(scene, sensor, E, k_eye, sensor_pick[i], rng, L.channel, false);
        sx.rays += L.rays + (k_eye ? E.rays : 0);
        for (int s = 0; s <= std::min(k_light, L.len); ++s)
          for (int t = 0; t <= (k_eye ? std::min(k_eye, E.len) : 0); ++t) {
            const int nn = s + t;
            if (nn < 2 || nn > depth + 1 || !ways[static_cast<std::size_t>(nn) * stride + s]) continue;
            contribute(scene, sensor, L, k_eye ? &E : nullptr, s, t, &ways[static_cast<std::size_t>(nn) * stride],
                       n_ways[nn], sx);
          }
        if (!L.cross.empty()) detector_deposits(scene, sensor, L, k_eye ? &E : nullptr, method, depth, ways, n_ways, sx);
      }
    });
    for (auto& s : scratch) {
      for (std::size_t k = 0; k < size; ++k) accum[k] += s.accum[k];
      std::fill(s.accum.begin(), s.accum.end(), 0.0);
      out.rays += s.rays;
      s.rays = 0;
    }
    done += n;
    spread.add(accum, n);
    if (opt.adaptive && adapts && spread.batches >= 2 && done < opt.samples) sensor.adapt(spread.error(done));
    if (opt.progress) opt.progress(static_cast<double>(done) / static_cast<double>(opt.samples));
    if (opt.cancelled && opt.cancelled()) break;
  }
  out.samples = done;
  out.channels.resize(size);
  out.values.assign(sensor.count, 0.0);
  for (std::size_t c = 0; c < channels; ++c)
    for (std::size_t k = 0; k < sensor.count; ++k) {
      out.channels[c * sensor.count + k] = accum[c * sensor.count + k] / static_cast<double>(std::max<long long>(done, 1));
      out.values[k] += out.channels[c * sensor.count + k];
    }
  out.error = spread.error(done);
  return out;
}

void light_paths(const Scene& scene, int n, int depth, std::uint64_t seed, std::vector<Vec3>& points,
                 std::vector<long long>& starts) {
  points.clear();
  starts.assign(1, 0);
  Rng order(seed, 0, 0xffffffffULL);
  const auto pick = stratified(n, order);
  Path path;
  for (int i = 0; i < n; ++i) {
    Rng rng(seed, 0, static_cast<std::uint64_t>(i));
    light_subpath(scene, nullptr, path, depth + 1, pick[i], rng, false);
    const std::size_t before = points.size();
    for (int k = 0; k < path.len; ++k) points.push_back(path.v[k].p);
    if (path.escape == path.len - 1) points.push_back(points.back() + path.escape_dir * (0.1 * scene.size));
    if (points.size() - before >= 2)
      starts.push_back(static_cast<long long>(points.size()));
    else
      points.resize(before);
  }
}

}  // namespace pt
