// PathTracer – light transport on trimmed NURBS geometry: light tracing, path tracing,
// bidirectional path tracing and photon mapping as estimators of the same quantities
// (irradiance on faces, radiance seen by a camera). Lengths in mm, power in W.
//
// The C++ twin of the NumPy kernel of the Path Tracer plugin of Raytatouille Fusion
// (python/plugins/pathtracer/pt_*.py): the same model and the same estimators, one path at a time
// instead of arrays of paths.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pt {

constexpr double kPi = 3.14159265358979323846;
constexpr double kInf = std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------------------- vectors, random numbers
struct Vec3 {
  double x = 0, y = 0, z = 0;
  Vec3() = default;
  Vec3(double a, double b, double c) : x(a), y(b), z(c) {}
  double operator[](int i) const { return i == 0 ? x : i == 1 ? y : z; }
  Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator-() const { return {-x, -y, -z}; }
  Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
  Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
  Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
};
inline Vec3 operator*(double s, const Vec3& v) { return v * s; }
inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalized(const Vec3& a) {
  const double n = norm(a);
  return n > 0 ? a / n : a;
}
/// Two unit vectors perpendicular to the unit vector `axis` and to each other.
void frame(const Vec3& axis, Vec3& a, Vec3& b);
/// The direction with the cosine `cos` to `axis` and the angle `phi` around it.
Vec3 around(const Vec3& axis, double cos, double phi);

/// xoshiro256++ – one generator per path, seeded from (seed, batch, path).
struct Rng {
  std::uint64_t s[4];
  explicit Rng(std::uint64_t seed = 0, std::uint64_t a = 0, std::uint64_t b = 0);
  std::uint64_t bits();
  double next() { return static_cast<double>(bits() >> 11) * (1.0 / 9007199254740992.0); }
};

// ---------------------------------------------------------------------------- NURBS
/// Curve of the .nfcad format; trim loops are curves in parameter space (z = 0).
struct Curve {
  int degree = 1;
  std::vector<double> knots;
  std::vector<Vec3> points;
  std::vector<double> weights;  // empty: all 1
  Vec3 at(double u) const;
  std::vector<double> spans() const;  // distinct knots of the domain
};

/// Surface of the .nfcad format; control point index = i_u * num_v + i_v.
struct Surface {
  int degree_u = 1, degree_v = 1;
  std::vector<double> knots_u, knots_v;
  int num_u = 0, num_v = 0;
  std::vector<Vec3> points;
  std::vector<double> weights;  // empty: all 1
  /// Point and first derivatives at (u, v).
  void at(double u, double v, Vec3& s, Vec3& su, Vec3& sv) const;
  std::array<double, 4> domain() const;  // u0, u1, v0, v1
  std::vector<double> spans_u() const, spans_v() const;
  void check() const;  // throws std::invalid_argument
};

/// The degree + 1 basis functions that are not zero at u and their derivatives (The NURBS Book,
/// A2.2); returns the knot span. N and dN need room for degree + 1 values.
int basis_functions(int degree, const std::vector<double>& knots, double u, double* N, double* dN);

// ---------------------------------------------------------------------------- geometry given
struct TrimmedFace {
  Surface surface;
  std::vector<std::vector<Curve>> loops;  // outer loop and holes (even-odd rule); empty: untrimmed
};
struct Body {
  long long id = 0;
  std::vector<TrimmedFace> faces;
};

// ---------------------------------------------------------------------------- mesh
/// What the area integrals of one face need: grid lines (normalised), the area density of every
/// grid cell (mm² per unit of normalised uv) and the trim mask.
struct FaceGrid {
  std::vector<double> ug, vg;
  std::vector<double> density;     // [(len(ug)-1) x (len(vg)-1)], index iu * (nv-1) + iv
  std::vector<std::uint8_t> mask;  // [res x res], index iv * res + iu; empty: untrimmed
  int mask_res = 0;
  double length_u = 0, length_v = 0;  // mean arc length along u and v
  bool curved = false;
  std::vector<long long> quad_tri;  // per grid cell its two triangles within the face (-1: left out)
  double max_density = 0;
  /// Area of the face inside every cell of the grid lines_u x lines_v: [iu * (len(lines_v)-1) + iv].
  std::vector<double> areas(const std::vector<double>& lines_u, const std::vector<double>& lines_v) const;
  bool inside(double u, double v) const;
};

struct Hit {
  double t = kInf;
  long long tri = -1;
  double b1 = 0, b2 = 0;
};

class Mesh {
 public:
  std::vector<Vec3> v0, e1, e2, ng;           // per triangle
  std::vector<std::array<Vec3, 3>> vn;        // vertex normals
  std::vector<std::array<double, 6>> vt;      // vertex (u, v): u0 v0 u1 v1 u2 v2
  std::vector<double> area;
  std::vector<int> face;                      // face of a triangle
  std::vector<int> face_entity, face_index;   // per face
  std::vector<double> face_area;
  std::vector<long long> face_first;          // triangles of face f: [face_first[f], face_first[f+1])
  std::vector<FaceGrid> grids;
  std::vector<Surface> surfaces;
  std::vector<long long> entity_ids;
  std::vector<double> cum_area;               // running sum of the triangle areas

  std::size_t triangles() const { return v0.size(); }
  std::size_t faces() const { return face_area.size(); }
  bool accept(long long tri, double b1, double b2) const;
  void shade(long long tri, double b1, double b2, Vec3& n, double& u, double& v) const;
  /// As shade, for a ray o + t d that hit the triangle: normal and (u, v) where the ray meets the
  /// surface itself (Newton's method); the triangle's values where that fails or the face is flat.
  void shade_exact(long long tri, double b1, double b2, const Vec3& o, const Vec3& d, double t, Vec3& n,
                   double& u, double& v) const;
  void flip_faces(const std::vector<int>& faces);
  /// A point uniform by area on one face (inside its trimming).
  void sample_face(int face, Rng& rng, double pick, long long& tri, double& b1, double& b2) const;
  std::pair<int, int> texels(int face, int long_side) const;  // (w, h)
  std::vector<double> texel_areas(int face, int w, int h) const;  // [j * w + i]
  /// The triangle of a face at (u, v): -1 if none; barycentric coordinates and its area density.
  long long locate(int face, double u, double v, double& b1, double& b2, double& density) const;
  std::vector<double> cell_density(int face) const;
  Vec3 point(long long tri, double b1, double b2) const { return v0[tri] + e1[tri] * b1 + e2[tri] * b2; }
};

Mesh build_mesh(const std::vector<Body>& bodies, double tol);

// ---------------------------------------------------------------------------- BVH
class Bvh {
 public:
  Bvh() = default;
  /// Over the triangles `tris` of the mesh (all if empty and `all`).
  Bvh(const Mesh& mesh, std::vector<long long> tris);
  bool empty() const { return tri_.empty(); }
  /// Nearest hit with 0 < t < tmax that `accept(tri, b1, b2)` lets through; any: the first found.
  template <class Accept>
  Hit intersect(const Vec3& o, const Vec3& d, double tmax, bool any, Accept&& accept) const;
  std::size_t nodes() const { return node_.size(); }

 private:
  struct Node {
    Vec3 lo, hi;
    int left = -1, right = -1;  // children, or -1 for a leaf
    int start = 0, count = 0;   // leaf: slots
  };
  int build(std::vector<int>& order, int start, int end, const std::vector<Vec3>& lo, const std::vector<Vec3>& hi,
            const std::vector<Vec3>& centre, int depth);
  std::vector<Node> node_;
  std::vector<long long> tri_;         // mesh triangle of a slot
  std::vector<Vec3> v0_, e1_, e2_;     // per slot
  int depth_ = 0;
};

// ---------------------------------------------------------------------------- materials
enum Kind { kAbsorber, kDiffuse, kMirror, kGlossy, kGlass, kDetector, kMetal, kFrosted };
extern const char* const kKindNames[8];

struct Material {
  std::string name;
  int kind = kDiffuse;
  double reflectance = 0, transmittance = 0, roughness = 0.2, ior = 1.5, absorption = 0, abbe = 0;
  std::vector<std::pair<double, double>> spectrum;  // (nm, value): reflectance or (glass) absorption
};

class Materials {
 public:
  explicit Materials(std::vector<Material> rows);  // throws std::invalid_argument
  void set_wavelengths(const std::vector<double>& nm);
  std::size_t size() const { return rows_.size(); }
  const Material& row(int m) const { return rows_[m]; }
  int kind(int m) const { return rows_[m].kind; }
  bool specular(int m) const { return rows_[m].kind == kMirror || rows_[m].kind == kGlass; }
  bool glassy(int m) const { return rows_[m].kind == kGlass || rows_[m].kind == kFrosted; }
  double ior(int m, int channel) const { return ior_[m * channels_ + channel]; }
  double rho(int m, int channel) const { return rho_[m * channels_ + channel]; }
  double absorbs(int m, int channel) const { return absorb_[m * channels_ + channel]; }
  /// f(wo, wi) and the densities of sampling wi from wo and wo from wi (0 for specular materials).
  /// radiance: the light arrives from wi; outside: the index in front of the surface.
  void eval(int m, const Vec3& n, const Vec3& wo, const Vec3& wi, int channel, bool radiance, double outside,
            double& f, double& p_fwd, double& p_rev) const;
  /// A scattered direction: weight = f |cos| / pdf (0: absorbed), pdf 0 for specular events.
  void sample(int m, const Vec3& n, const Vec3& wo, Rng& rng, bool radiance, int channel, double outside,
              bool straight, Vec3& wi, double& weight, double& pdf) const;

 private:
  std::vector<Material> rows_;
  std::vector<double> exponent_, alpha_;
  std::vector<double> ior_, rho_, absorb_;  // [material x channel]
  int channels_ = 1;
};

// ---------------------------------------------------------------------------- photometry
/// Intensity over the directions as a table of cells, scaled to integral 1 over the sphere.
class Distribution {
 public:
  /// theta, phi in degrees (phi a full turn, or one value: the same all around); intensity[phi][theta].
  Distribution(const std::vector<double>& theta, const std::vector<double>& phi, const std::vector<double>& intensity);
  double eval(double cos, double phi) const;
  void sample(Rng& rng, double& cos, double& phi, double& density) const;
  bool symmetric() const { return cells_p_ == 1; }
  const std::vector<double>& density() const { return density_; }

 private:
  int cells_t_ = 360, cells_p_ = 1;
  std::vector<double> density_, cdf_, cos_edges_;
};
/// The eye's photopic sensitivity (CIE 1924) at a wavelength in nm.
double v_lambda(double nm);

/// A luminaire file as a table: theta, phi in degrees (phi a full turn, or one value) and
/// intensity[phi][theta].
struct LuminaireTable {
  std::vector<double> theta, phi, intensity;
  Distribution distribution() const { return Distribution(theta, phi, intensity); }
};
LuminaireTable table_ies(const std::string& text);   // IES LM-63, photometry of type C
LuminaireTable table_ldt(const std::string& text);   // EULUMDAT
LuminaireTable load_luminaire(const std::string& path);  // by the ending, else by the content

// ---------------------------------------------------------------------------- scene
struct SourceSpec {
  std::string name;
  double flux = 1;                       // W
  double half_angle = 90;                // degrees, surfaces
  int side = 1;                          // +1 front, -1 back, 0 both
  std::vector<int> entities;             // bodies (indices into the mesh's bodies)
  std::vector<Vec3> points;
  std::vector<std::vector<Vec3>> curves;  // polylines of wire sources
  std::vector<std::pair<double, double>> spectrum{{550.0, 1.0}};
  std::shared_ptr<const Distribution> distribution;
  Vec3 axis{0, 0, -1};
};

struct Emission {
  Vec3 p, ns, ng, d;
  int src = 0, channel = 0;
  double pdf_pos = 1, pdf_dir = 1;
  bool ok = true, point = false;
};

class Scene {
 public:
  Scene(Mesh mesh, Materials materials, std::vector<int> entity_material, const std::vector<SourceSpec>& sources,
        const std::vector<int>& closed);

  Mesh mesh;
  Materials materials;
  bool exact = true;       // normals of the surfaces themselves where rays hit curved faces
  double contact = 0;      // glass bodies whose surfaces lie this close touch
  double size = 0, eps = 0, flux = 0;
  std::vector<int> flipped;  // faces turned outwards
  std::vector<int> entity_material, face_material, tri_material, tri_entity, face_source;
  std::vector<std::uint8_t> tri_glassy, face_virtual;
  std::vector<double> wavelengths, channel_prob;
  bool has_detectors = false;

  // sources (one entry per surface source, wire source and point)
  std::vector<double> src_flux, src_prob, src_area, src_sin, src_cos, src_radiance, src_dir_norm;
  std::vector<int> src_sides;
  std::vector<std::uint8_t> src_is_point, src_is_wire;
  std::vector<Vec3> src_point, src_axis;
  std::vector<std::shared_ptr<const Distribution>> src_measured;
  std::vector<std::string> src_names;
  std::size_t sources() const { return src_flux.size(); }

  Hit hit(const Vec3& o, const Vec3& d) const;                    // detectors are not hit
  bool occluded(const Vec3& o, const Vec3& target) const;
  Hit crossing(const Vec3& o, const Vec3& d, double tmax) const;  // nearest detector crossing
  Vec3 offset(const Vec3& p, const Vec3& ng, const Vec3& towards) const;
  /// Glass surfaces within `contact` of p along d (see the Python kernel: Scene.touching).
  int touching(const Vec3& p, const Vec3& d, int body, bool ends) const;
  /// Radiance (points, wires: intensity) of a source into w and the density of emission picking w.
  void emitted(int src, const Vec3& n, const Vec3& w, double& le, double& pdf) const;
  double emit_pdf_pos(int src) const;
  double spectral_weight(int src, int channel) const;
  /// An emitted ray; `pick` in [0, 1) chooses the source (stratified by the caller).
  Emission emit(double pick, Rng& rng) const;

 private:
  void orient(const std::vector<int>& closed);
  void make_sources(const std::vector<SourceSpec>& specs);
  Vec3 measured_axis(int src, const Vec3& n) const;
  Bvh bvh_, detectors_;
  std::vector<std::vector<int>> src_faces_;
  std::vector<std::vector<double>> src_face_cum_;  // cumulative face areas per source
  std::vector<std::vector<Vec3>> wire_a_, wire_b_;
  std::vector<std::vector<double>> wire_cum_;
  std::vector<double> src_spectrum_;  // [source x channel], rows sum to 1
  std::vector<double> src_cum_, spectrum_cum_;
};

// ---------------------------------------------------------------------------- sensors
struct SensorSample {
  Vec3 p, ns, ng, d;
  long long texel = -1;
  bool ok = false, exists = false, point = false;
  double pdf_pos = 1, pdf_dir = 1, beta = 0;
};

class Sensor {
 public:
  virtual ~Sensor() = default;
  bool camera = false;    // no light path can land on it
  bool pinhole = false;
  bool crossed = false;   // light passes some of its faces (detectors)
  std::size_t count = 0;  // texels or pixels
  std::vector<double> prob;  // share of the samples per texel / pixel
  /// A ray from the sensor; `pick` in [0, 1) chooses the texel / pixel (stratified by the caller).
  virtual SensorSample sample(double pick, Rng& rng) const = 0;
  /// Importance into w from a sensor point, the density of picking w, and the pixel (-2: the point's own).
  virtual void response(const Vec3& ns, const Vec3& w, const Vec3& p, double& importance, double& pdf,
                        long long& pixel) const = 0;
  virtual double pdf_pos_at(long long tex) const = 0;
  virtual long long texel(long long /*tri*/, double /*u*/, double /*v*/) const { return -1; }
  virtual void reset() = 0;
  void set_weights(const std::vector<double>& weights);
  virtual void adapt(const std::vector<double>& error, double floor = 0.3) = 0;

 protected:
  std::vector<double> cum_prob_;
  virtual void normalise(std::vector<double>& /*weights*/) const {}
};

/// Irradiance on faces: every face gets a texture over its normalised (u, v).
class TexelSensor : public Sensor {
 public:
  TexelSensor(const Scene& scene, const std::vector<int>& faces, int long_side);
  const Scene& scene;
  std::vector<int> faces;                 // sensor faces
  std::vector<int> w, h;                  // texture size per face
  std::vector<long long> first;           // texels of sensor face k: [first[k], first[k+1])
  std::vector<double> texel_area, texel_density;
  double area = 0;
  SensorSample sample(double pick, Rng& rng) const override;
  void response(const Vec3& ns, const Vec3& w, const Vec3& p, double& importance, double& pdf,
                long long& pixel) const override;
  double pdf_pos_at(long long tex) const override { return prob[tex] / texel_area[tex]; }
  long long texel(long long tri, double u, double v) const override;
  void reset() override;
  void adapt(const std::vector<double>& error, double floor = 0.3) override;
  /// A point on the sensor: its texel by `pick`, uniform by area inside; false if none was found.
  bool point(double pick, Rng& rng, long long& tri, double& b1, double& b2, long long& tex) const;

 private:
  std::vector<int> slot_;  // face -> sensor face or -1
  void normalise(std::vector<double>& weights) const override;
};

/// A camera: a pinhole (lens_radius 0) or a thin lens that is sharp at `focus`.
class Camera : public Sensor {
 public:
  Camera(const Vec3& position, const Vec3& target, const Vec3& up, double fov, int width, int height,
         double lens_radius = 0, double focus = 0);
  Vec3 position, forward, right, up;
  int width, height;
  double tan_x, tan_y, film, lens_radius, focus, lens_area;
  SensorSample sample(double pick, Rng& rng) const override;
  void response(const Vec3& ns, const Vec3& w, const Vec3& p, double& importance, double& pdf,
                long long& pixel) const override;
  double pdf_pos_at(long long) const override { return 1.0; }
  void reset() override;
  void adapt(const std::vector<double>& error, double floor = 0.3) override;

 private:
  long long pixel_of(double x, double y) const;
};

// ---------------------------------------------------------------------------- rendering
enum Method { kLight, kPath, kBidir, kPhoton };

struct RenderOptions {
  Method method = kLight;
  long long samples = 100000;
  int depth = 8;            // longest path, in segments
  std::uint64_t seed = 0;
  int threads = 0;          // 0: all cores
  long long batch = 0;      // 0: chosen (eight batches where the samples allow, for the error)
  double radius = 0;        // photon mapping: first search radius, 0: chosen
  bool adaptive = false;
  std::vector<double> weights;  // shares of the texels / pixels to start with
  std::function<void(double)> progress;  // called between batches with the fraction done
  std::function<bool()> cancelled;
};

struct Result {
  std::vector<double> values;    // per texel the flux / W, per pixel the radiance / W/(mm² sr)
  std::vector<double> channels;  // [wavelength x texel]
  std::vector<double> error;     // standard error of the values; empty with a single batch
  long long samples = 0, rays = 0;
  int passes = 0;                // photon mapping
  double radius = 0;
  int threads = 1;
};

/// Whether a method uses the way with s light and t eye vertices.
bool enabled(Method method, bool camera, int s, int t);
Result render(const Scene& scene, Sensor& sensor, const RenderOptions& options);
/// Light paths as polylines for drawing: points and the start of each polyline (plus the total).
void light_paths(const Scene& scene, int n, int depth, std::uint64_t seed, std::vector<Vec3>& points,
                 std::vector<long long>& starts);

/// The light that leaves the scene: per escaped light subpath its direction, its flux (its weight
/// over n, so that the fluxes of one call add up to the escaping flux), its wavelength's number,
/// its number of surface events before it left (0: straight from the source) and the entities of
/// its first and its last surface (-1 when there was none).
struct Escapes {
  std::vector<Vec3> d;
  std::vector<double> flux;
  std::vector<int> channel, bounces, first, last;
  long long rays = 0;
  int threads = 1;
};
/// n light subpaths of up to `depth` segments, batch `index` of the seed (batches of one seed are
/// independent); threads 0: all cores. The result does not depend on the number of threads.
Escapes escapes(const Scene& scene, long long n, int depth, std::uint64_t seed, std::uint64_t index = 0,
                int threads = 0);

// ---------------------------------------------------------------------------- inline: BVH traversal
template <class Accept>
Hit Bvh::intersect(const Vec3& o, const Vec3& d, double tmax, bool any, Accept&& accept) const {
  Hit best;
  best.t = tmax;
  if (node_.empty()) {
    best.t = kInf;
    return best;
  }
  const double inv[3] = {std::abs(d.x) > 1e-300 ? 1.0 / d.x : 1e300, std::abs(d.y) > 1e-300 ? 1.0 / d.y : 1e300,
                         std::abs(d.z) > 1e-300 ? 1.0 / d.z : 1e300};
  auto entry = [&](const Node& n, double limit) {  // distance at which the ray enters the box, or inf
    double near = 0.0, far = limit;
    for (int a = 0; a < 3; ++a) {
      const double t0 = (n.lo[a] - o[a]) * inv[a], t1 = (n.hi[a] - o[a]) * inv[a];
      near = std::max(near, std::min(t0, t1));
      far = std::min(far, std::max(t0, t1));
    }
    return far >= near ? near : kInf;
  };
  int stack[128];
  double at[128];
  int top = 0;
  stack[top] = 0;
  at[top++] = entry(node_[0], best.t);
  while (top > 0) {
    --top;
    if (at[top] >= best.t) continue;  // also: the box is missed (inf)
    const Node& n = node_[stack[top]];
    if (n.left < 0) {
      for (int slot = n.start; slot < n.start + n.count; ++slot) {
        const Vec3 &e1 = e1_[slot], &e2 = e2_[slot];
        const Vec3 p = cross(d, e2);
        const double det = dot(e1, p);
        if (det == 0.0) continue;
        const double idet = 1.0 / det;
        const Vec3 sv = o - v0_[slot];
        const double b1 = dot(sv, p) * idet;
        if (b1 < -1e-12) continue;
        const Vec3 q = cross(sv, e1);
        const double b2 = dot(d, q) * idet;
        if (b2 < -1e-12 || b1 + b2 > 1 + 1e-12) continue;
        const double t = dot(e2, q) * idet;
        if (!(t > 0.0) || !(t < best.t)) continue;
        if (!accept(tri_[slot], b1, b2)) continue;
        best = {t, tri_[slot], b1, b2};
        if (any) return best;
      }
      continue;
    }
    const double tl = entry(node_[n.left], best.t), tr = entry(node_[n.right], best.t);
    const bool left_first = tl <= tr;  // the farther child below, the nearer on top
    const int far_child = left_first ? n.right : n.left, near_child = left_first ? n.left : n.right;
    const double tfar = left_first ? tr : tl, tnear = left_first ? tl : tr;
    if (tfar < best.t && top < 127) { stack[top] = far_child; at[top++] = tfar; }
    if (tnear < best.t && top < 127) { stack[top] = near_child; at[top++] = tnear; }
  }
  if (best.tri < 0) best.t = kInf;
  return best;
}

}  // namespace pt
