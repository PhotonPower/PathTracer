// CAD geometry as a triangle mesh for tracing: every face tessellated on a grid of its surface
// parameters to a chordal tolerance; trimming as a mask over (u, v) that is tested at every hit;
// areas integrated over the same mask (see the Python kernel: pt_mesh.py).
#include "pathtracer/pathtracer.hpp"

#include <algorithm>
#include <stdexcept>

namespace pt {
namespace {

std::vector<double> lines(const std::vector<double>& spans, int per_span) {
  std::vector<double> out;
  for (std::size_t k = 0; k + 1 < spans.size(); ++k)
    for (int i = 0; i < per_span; ++i) out.push_back(spans[k] + (spans[k + 1] - spans[k]) * i / per_span);
  out.push_back(spans.back());
  return out;
}

/// Inside per cell of a res x res raster over the normalised domain: even-odd over all loops.
std::vector<std::uint8_t> trim_mask(const std::vector<std::vector<Curve>>& loops, const std::array<double, 4>& dom,
                                    int res) {
  std::vector<int> toggles(static_cast<std::size_t>(res) * (res + 1), 0);
  for (const auto& loop : loops) {
    std::vector<std::array<double, 2>> p;
    for (const Curve& c : loop) {
      std::vector<double> t = c.spans();
      if (c.degree > 1) {
        std::vector<double> fine;
        for (std::size_t k = 0; k + 1 < t.size(); ++k)
          for (int i = 0; i < 24; ++i) fine.push_back(t[k] + (t[k + 1] - t[k]) * i / 24.0);
        fine.push_back(t.back());
        t = fine;
      }
      for (double u : t) {
        const Vec3 q = c.at(u);
        p.push_back({(q.x - dom[0]) / (dom[1] - dom[0]), (q.y - dom[2]) / (dom[3] - dom[2])});
      }
    }
    if (p.size() < 3) continue;
    for (std::size_t k = 0; k < p.size(); ++k) {
      const auto& a = p[k];
      const auto& b = p[(k + 1) % p.size()];
      const double ylo = std::min(a[1], b[1]), yhi = std::max(a[1], b[1]);
      if (!(yhi > ylo)) continue;
      const int r0 = std::max(0, static_cast<int>(std::ceil(ylo * res - 0.5)));
      const int r1 = std::min(res - 1, static_cast<int>(std::ceil(yhi * res - 0.5)) - 1);
      for (int r = r0; r <= r1; ++r) {
        const double yc = (r + 0.5) / res;
        if (!(ylo <= yc && yc < yhi)) continue;
        const double x = a[0] + (yc - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
        const int col = std::clamp(static_cast<int>(std::ceil(x * res - 0.5)), 0, res);  // first centre right of x
        ++toggles[static_cast<std::size_t>(r) * (res + 1) + col];
      }
    }
  }
  std::vector<std::uint8_t> mask(static_cast<std::size_t>(res) * res);
  for (int r = 0; r < res; ++r) {
    int sum = 0;
    for (int i = 0; i < res; ++i) {
      sum += toggles[static_cast<std::size_t>(r) * (res + 1) + i];
      mask[static_cast<std::size_t>(r) * res + i] = static_cast<std::uint8_t>(sum & 1);
    }
  }
  return mask;
}

/// Overlap of every interval of a with every interval of b: [(len(a)-1) x (len(b)-1)].
std::vector<double> overlap(const std::vector<double>& a, const std::vector<double>& b) {
  const std::size_t na = a.size() - 1, nb = b.size() - 1;
  std::vector<double> out(na * nb, 0.0);
  for (std::size_t i = 0; i < na; ++i)
    for (std::size_t j = 0; j < nb; ++j)
      out[i * nb + j] = std::max(0.0, std::min(a[i + 1], b[j + 1]) - std::max(a[i], b[j]));
  return out;
}

struct Lines {
  std::vector<double> us, vs;
  bool curved = false;
};

/// Grid lines so that the triangles stay within tol of the surface (second differences of a
/// sample with 8 steps per knot span), and whether the surface is curved at all.
Lines subdivisions(const Surface& surface, double tol, int limit) {
  const auto ku = surface.spans_u(), kv = surface.spans_v();
  const int n = 8;
  const auto su = lines(ku, n), sv = lines(kv, n);
  const std::size_t nu = su.size(), nv = sv.size();
  std::vector<Vec3> s(nu * nv);
  Vec3 du, dv, lo{kInf, kInf, kInf}, hi{-kInf, -kInf, -kInf};
  for (std::size_t i = 0; i < nu; ++i)
    for (std::size_t j = 0; j < nv; ++j) {
      Vec3& p = s[i * nv + j];
      surface.at(su[i], sv[j], p, du, dv);
      lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
      hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
  double d2u = 0, d2v = 0, twist = 0;
  for (std::size_t i = 1; i + 1 < nu; ++i)
    if (i % n != 0)
      for (std::size_t j = 0; j < nv; ++j)
        d2u = std::max(d2u, norm(s[(i + 1) * nv + j] - s[i * nv + j] * 2 + s[(i - 1) * nv + j]));
  for (std::size_t j = 1; j + 1 < nv; ++j)
    if (j % n != 0)
      for (std::size_t i = 0; i < nu; ++i)
        d2v = std::max(d2v, norm(s[i * nv + j + 1] - s[i * nv + j] * 2 + s[i * nv + j - 1]));
  for (std::size_t i = 0; i + 1 < nu; ++i)
    for (std::size_t j = 0; j + 1 < nv; ++j)
      twist = std::max(twist, norm(s[(i + 1) * nv + j + 1] - s[(i + 1) * nv + j] - s[i * nv + j + 1] + s[i * nv + j]));
  const double size = norm(hi - lo);
  Lines out;
  auto count = [&](double d2, const std::vector<double>& spans) {
    const double err = std::max(d2 / 8.0, twist / 4.0);
    out.curved = out.curved || err > 1e-12 * std::max(size, 1e-300);
    int m = err > 0 ? static_cast<int>(std::ceil(n * std::sqrt(err / tol))) : 1;
    return std::max(1, std::min(m, std::max(1, limit / static_cast<int>(spans.size() - 1))));
  };
  out.us = lines(ku, count(d2u, ku));
  out.vs = lines(kv, count(d2v, kv));
  return out;
}

struct Tessellation {
  std::vector<Vec3> v0, e1, e2;
  std::vector<std::array<Vec3, 3>> vn;
  std::vector<std::array<double, 6>> vt;
  FaceGrid grid;
};

Tessellation tessellate(const TrimmedFace& face, double tol, int mask_res, int limit = 192) {
  const Surface& surface = face.surface;
  surface.check();
  const auto dom = surface.domain();
  Lines l = subdivisions(surface, tol, limit);
  const bool trimmed = !face.loops.empty();
  if (trimmed) {  // masks need a parametrisation that is not too coarse
    const auto ku = surface.spans_u(), kv = surface.spans_v();
    if (l.us.size() < 9) l.us = lines(ku, static_cast<int>(std::ceil(8.0 / (ku.size() - 1))));
    if (l.vs.size() < 9) l.vs = lines(kv, static_cast<int>(std::ceil(8.0 / (kv.size() - 1))));
  }
  const std::size_t nu = l.us.size(), nv = l.vs.size();
  std::vector<Vec3> s(nu * nv), nrm(nu * nv);
  for (std::size_t i = 0; i < nu; ++i)
    for (std::size_t j = 0; j < nv; ++j) {
      Vec3 su, sv;
      surface.at(l.us[i], l.vs[j], s[i * nv + j], su, sv);
      const Vec3 c = cross(su, sv);
      const double len = norm(c);
      nrm[i * nv + j] = len > 1e-300 ? c / len : Vec3{};  // poles: filled per triangle later
    }
  Tessellation out;
  FaceGrid& g = out.grid;
  g.ug.resize(nu);
  g.vg.resize(nv);
  for (std::size_t i = 0; i < nu; ++i) g.ug[i] = (l.us[i] - dom[0]) / (dom[1] - dom[0]);
  for (std::size_t j = 0; j < nv; ++j) g.vg[j] = (l.vs[j] - dom[2]) / (dom[3] - dom[2]);
  g.curved = l.curved;
  const std::size_t cu = nu - 1, cv = nv - 1;
  std::vector<double> area1(cu * cv), area2(cu * cv), cell(cu * cv);
  g.density.resize(cu * cv);
  for (std::size_t i = 0; i < cu; ++i)
    for (std::size_t j = 0; j < cv; ++j) {
      const Vec3 &a = s[i * nv + j], &b = s[(i + 1) * nv + j], &c = s[(i + 1) * nv + j + 1], &d = s[i * nv + j + 1];
      const std::size_t q = i * cv + j;
      area1[q] = 0.5 * norm(cross(b - a, c - a));
      area2[q] = 0.5 * norm(cross(c - a, d - a));
      cell[q] = (g.ug[i + 1] - g.ug[i]) * (g.vg[j + 1] - g.vg[j]);
      g.density[q] = (area1[q] + area2[q]) / cell[q];
    }
  if (trimmed) {
    g.mask = trim_mask(face.loops, dom, mask_res);
    g.mask_res = mask_res;
  }
  for (std::size_t j = 0; j < nv; ++j)
    for (std::size_t i = 0; i < cu; ++i) g.length_u += norm(s[(i + 1) * nv + j] - s[i * nv + j]);
  g.length_u /= static_cast<double>(nv);
  for (std::size_t i = 0; i < nu; ++i)
    for (std::size_t j = 0; j < cv; ++j) g.length_v += norm(s[i * nv + j + 1] - s[i * nv + j]);
  g.length_v /= static_cast<double>(nu);
  std::vector<double> inside_area;
  if (trimmed) inside_area = g.areas(g.ug, g.vg);
  g.quad_tri.assign(cu * cv * 2, -1);
  long long count = 0;
  for (int which = 0; which < 2; ++which) {
    const auto& area = which == 0 ? area1 : area2;
    for (std::size_t i = 0; i < cu; ++i)
      for (std::size_t j = 0; j < cv; ++j) {
        const std::size_t q = i * cv + j;
        if ((trimmed && !(inside_area[q] > 0)) || !(area[q] > 0)) continue;
        g.quad_tri[q * 2 + which] = count++;
        g.max_density = std::max(g.max_density, 2 * area[q] / cell[q]);
        const std::size_t ia = i * nv + j, ib = (i + 1) * nv + j, ic = (i + 1) * nv + j + 1, id = i * nv + j + 1;
        const std::size_t i1 = which == 0 ? ib : ic, i2 = which == 0 ? ic : id;
        out.v0.push_back(s[ia]);
        out.e1.push_back(s[i1] - s[ia]);
        out.e2.push_back(s[i2] - s[ia]);
        out.vn.push_back({nrm[ia], nrm[i1], nrm[i2]});
        auto uv = [&](std::size_t k, double& u, double& v) { u = g.ug[k / nv]; v = g.vg[k % nv]; };
        std::array<double, 6> t{};
        uv(ia, t[0], t[1]);
        uv(i1, t[2], t[3]);
        uv(i2, t[4], t[5]);
        out.vt.push_back(t);
      }
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------- FaceGrid
bool FaceGrid::inside(double u, double v) const {
  if (mask.empty()) return true;
  const int i = std::clamp(static_cast<int>(u * mask_res), 0, mask_res - 1);
  const int j = std::clamp(static_cast<int>(v * mask_res), 0, mask_res - 1);
  return mask[static_cast<std::size_t>(j) * mask_res + i] != 0;
}

std::vector<double> FaceGrid::areas(const std::vector<double>& lines_u, const std::vector<double>& lines_v) const {
  const std::size_t tu = lines_u.size() - 1, tv = lines_v.size() - 1;
  const std::size_t gu = ug.size() - 1, gv = vg.size() - 1;
  // x[su][sv]: the area density on segments that lie in one grid cell (and one mask cell)
  std::vector<double> su = ug, sv = vg, x;
  if (!mask.empty()) {
    for (int k = 0; k <= mask_res; ++k) {
      su.push_back(static_cast<double>(k) / mask_res);
      sv.push_back(static_cast<double>(k) / mask_res);
    }
    for (auto* line : {&su, &sv}) {
      std::sort(line->begin(), line->end());
      line->erase(std::unique(line->begin(), line->end()), line->end());
    }
  }
  const std::size_t nsu = su.size() - 1, nsv = sv.size() - 1;
  auto cell_of = [](const std::vector<double>& grid, double mid) {
    const long k = static_cast<long>(std::lower_bound(grid.begin(), grid.end(), mid) - grid.begin()) - 1;
    return static_cast<std::size_t>(std::clamp<long>(k, 0, static_cast<long>(grid.size()) - 2));
  };
  std::vector<std::size_t> qu(nsu), qv(nsv), mu(nsu), mv(nsv);
  for (std::size_t a = 0; a < nsu; ++a) {
    const double mid = 0.5 * (su[a] + su[a + 1]);
    qu[a] = cell_of(ug, mid);
    mu[a] = mask.empty() ? 0 : std::min<std::size_t>(static_cast<std::size_t>(mid * mask_res), mask_res - 1);
  }
  for (std::size_t b = 0; b < nsv; ++b) {
    const double mid = 0.5 * (sv[b] + sv[b + 1]);
    qv[b] = cell_of(vg, mid);
    mv[b] = mask.empty() ? 0 : std::min<std::size_t>(static_cast<std::size_t>(mid * mask_res), mask_res - 1);
  }
  (void)gu;
  x.resize(nsu * nsv);
  for (std::size_t a = 0; a < nsu; ++a)
    for (std::size_t b = 0; b < nsv; ++b) {
      const bool in = mask.empty() || mask[mv[b] * mask_res + mu[a]] != 0;
      x[a * nsv + b] = in ? density[qu[a] * gv + qv[b]] : 0.0;
    }
  const auto wu = overlap(lines_u, su), wv = overlap(lines_v, sv);
  std::vector<double> tmp(tu * nsv, 0.0), out(tu * tv, 0.0);
  for (std::size_t i = 0; i < tu; ++i)
    for (std::size_t a = 0; a < nsu; ++a) {
      const double w = wu[i * nsu + a];
      if (w == 0.0) continue;
      for (std::size_t b = 0; b < nsv; ++b) tmp[i * nsv + b] += w * x[a * nsv + b];
    }
  for (std::size_t i = 0; i < tu; ++i)
    for (std::size_t j = 0; j < tv; ++j) {
      double sum = 0;
      for (std::size_t b = 0; b < nsv; ++b) sum += tmp[i * nsv + b] * wv[j * nsv + b];
      out[i * tv + j] = sum;
    }
  return out;
}

// ---------------------------------------------------------------------------- Mesh
bool Mesh::accept(long long tri, double b1, double b2) const {
  const FaceGrid& g = grids[face[tri]];
  if (g.mask.empty()) return true;
  const auto& t = vt[tri];
  const double b0 = 1 - b1 - b2;
  return g.inside(t[0] * b0 + t[2] * b1 + t[4] * b2, t[1] * b0 + t[3] * b1 + t[5] * b2);
}

void Mesh::shade(long long tri, double b1, double b2, Vec3& n, double& u, double& v) const {
  const double b0 = 1 - b1 - b2;
  const auto& t = vt[tri];
  u = t[0] * b0 + t[2] * b1 + t[4] * b2;
  v = t[1] * b0 + t[3] * b1 + t[5] * b2;
  n = vn[tri][0] * b0 + vn[tri][1] * b1 + vn[tri][2] * b2;
  const double len = norm(n);
  if (len < 1e-12 || dot(n, ng[tri]) <= 0)
    n = ng[tri];
  else
    n = n / len;
}

void Mesh::shade_exact(long long tri, double b1, double b2, const Vec3& o, const Vec3& d, double t, Vec3& n,
                       double& u, double& v) const {
  shade(tri, b1, b2, n, u, v);
  const int f = face[tri];
  if (!grids[f].curved) return;
  const Surface& surface = surfaces[f];
  const auto dom = surface.domain();
  double pu_ = dom[0] + u * (dom[1] - dom[0]), pv_ = dom[2] + v * (dom[3] - dom[2]), s = t;
  const double scale = face_area[f] > 0 ? std::sqrt(face_area[f]) : 1.0;
  Vec3 p, su, sv;
  for (int it = 0; it < 6; ++it) {
    surface.at(pu_, pv_, p, su, sv);
    const Vec3 miss = p - o - d * s;
    // solve [su sv -d] step = -miss
    const Vec3 c0 = su, c1 = sv, c2 = -d, r = -miss;
    const double det = dot(c0, cross(c1, c2));
    if (!(std::abs(det) > 1e-14 * scale * scale)) return;
    const double x = dot(r, cross(c1, c2)) / det, y = dot(c0, cross(r, c2)) / det, z = dot(c0, cross(c1, r)) / det;
    pu_ = std::clamp(pu_ + x, dom[0], dom[1]);
    pv_ = std::clamp(pv_ + y, dom[2], dom[3]);
    s += z;
  }
  surface.at(pu_, pv_, p, su, sv);
  Vec3 normal = cross(su, sv);
  const double len = norm(normal);
  if (!(norm(p - o - d * s) < 1e-9 * scale) || !(len > 1e-12 * scale) || !(s > 0)) return;
  normal = normal / len;
  const double side = dot(normal, ng[tri]);
  if (side < 0) normal = -normal;  // the triangle's side
  const double nu_ = (pu_ - dom[0]) / (dom[1] - dom[0]), nv_ = (pv_ - dom[2]) / (dom[3] - dom[2]);
  if (!(std::abs(side) > 0.5) || !(std::max(std::abs(nu_ - u), std::abs(nv_ - v)) < 0.25)) return;  // near the triangle
  n = normal;
  u = nu_;
  v = nv_;
}

void Mesh::flip_faces(const std::vector<int>& faces) {
  for (int f : faces)
    for (long long t = face_first[f]; t < face_first[f + 1]; ++t) {
      std::swap(e1[t], e2[t]);
      ng[t] = -ng[t];
      vn[t] = {-vn[t][0], -vn[t][2], -vn[t][1]};
      vt[t] = {vt[t][0], vt[t][1], vt[t][4], vt[t][5], vt[t][2], vt[t][3]};
    }
}

void Mesh::sample_face(int f, Rng& rng, double pick, long long& tri, double& b1, double& b2) const {
  const long long a = face_first[f], b = face_first[f + 1];
  const double lo = a > 0 ? cum_area[a - 1] : 0.0, hi = cum_area[b - 1];
  for (int attempt = 0; attempt < 1000; ++attempt) {
    const double target = lo + (attempt == 0 ? pick : rng.next()) * (hi - lo);
    tri = std::upper_bound(cum_area.begin() + a, cum_area.begin() + b, target) - cum_area.begin();
    tri = std::clamp(tri, a, b - 1);
    b1 = rng.next();
    b2 = rng.next();
    if (b1 + b2 > 1) {
      b1 = 1 - b1;
      b2 = 1 - b2;
    }
    if (accept(tri, b1, b2)) return;
  }
}

std::pair<int, int> Mesh::texels(int f, int long_side) const {
  const FaceGrid& g = grids[f];
  const double lu = std::max(g.length_u, 1e-300), lv = std::max(g.length_v, 1e-300);
  if (lu >= lv) return {long_side, std::max(1, static_cast<int>(std::lround(long_side * lv / lu)))};
  return {std::max(1, static_cast<int>(std::lround(long_side * lu / lv))), long_side};
}

std::vector<double> Mesh::texel_areas(int f, int w, int h) const {
  std::vector<double> lu(w + 1), lv(h + 1);
  for (int i = 0; i <= w; ++i) lu[i] = static_cast<double>(i) / w;
  for (int j = 0; j <= h; ++j) lv[j] = static_cast<double>(j) / h;
  const auto a = grids[f].areas(lu, lv);  // [i * h + j]
  std::vector<double> out(static_cast<std::size_t>(w) * h);
  for (int i = 0; i < w; ++i)
    for (int j = 0; j < h; ++j) out[static_cast<std::size_t>(j) * w + i] = a[static_cast<std::size_t>(i) * h + j];
  return out;
}

long long Mesh::locate(int f, double u, double v, double& b1, double& b2, double& density) const {
  const FaceGrid& g = grids[f];
  const long nu = static_cast<long>(g.ug.size()), nv = static_cast<long>(g.vg.size());
  const long iu = std::clamp<long>(std::upper_bound(g.ug.begin(), g.ug.end(), u) - g.ug.begin() - 1, 0, nu - 2);
  const long iv = std::clamp<long>(std::upper_bound(g.vg.begin(), g.vg.end(), v) - g.vg.begin() - 1, 0, nv - 2);
  const double du = g.ug[iu + 1] - g.ug[iu], dv = g.vg[iv + 1] - g.vg[iv];
  const int above = (v - g.vg[iv]) / dv > (u - g.ug[iu]) / du ? 1 : 0;  // of the cell's diagonal
  const long long local = g.quad_tri[(iu * (nv - 1) + iv) * 2 + above];
  density = 0;
  b1 = b2 = 0;
  if (local < 0) return -1;
  const long long tri = face_first[f] + local;
  const auto& t = vt[tri];  // barycentric from the vertices' (u, v): right also for faces that were turned
  const double d1u = t[2] - t[0], d1v = t[3] - t[1], d2u = t[4] - t[0], d2v = t[5] - t[1];
  const double ru = u - t[0], rv = v - t[1];
  double det = d1u * d2v - d1v * d2u;
  if (det == 0) det = 1;
  b1 = std::clamp((ru * d2v - rv * d2u) / det, 0.0, 1.0);
  b2 = std::clamp((d1u * rv - d1v * ru) / det, 0.0, 1.0 - b1);
  density = 2 * area[tri] / (du * dv);
  return tri;
}

std::vector<double> Mesh::cell_density(int f) const {
  const FaceGrid& g = grids[f];
  const std::size_t cu = g.ug.size() - 1, cv = g.vg.size() - 1;
  std::vector<double> out(cu * cv, 0.0);
  for (std::size_t i = 0; i < cu; ++i)
    for (std::size_t j = 0; j < cv; ++j) {
      const double cell = (g.ug[i + 1] - g.ug[i]) * (g.vg[j + 1] - g.vg[j]);
      for (int which = 0; which < 2; ++which) {
        const long long local = g.quad_tri[(i * cv + j) * 2 + which];
        if (local >= 0) out[i * cv + j] = std::max(out[i * cv + j], 2 * area[face_first[f] + local] / cell);
      }
    }
  return out;
}

Mesh build_mesh(const std::vector<Body>& bodies, double tol) {
  std::size_t total = 0, trimmed = 0;
  for (const Body& b : bodies)
    for (const TrimmedFace& f : b.faces) {
      ++total;
      if (!f.loops.empty()) ++trimmed;
    }
  if (total == 0) throw std::invalid_argument("no surfaces to trace");
  if (!(tol > 0)) throw std::invalid_argument("the mesh tolerance must be above 0");
  const int res = trimmed <= 64 ? 512 : trimmed <= 1024 ? 256 : 128;
  Mesh m;
  m.face_first.push_back(0);
  for (std::size_t k = 0; k < bodies.size(); ++k) {
    m.entity_ids.push_back(bodies[k].id);
    for (std::size_t i = 0; i < bodies[k].faces.size(); ++i) {
      Tessellation t = tessellate(bodies[k].faces[i], tol, res);
      const int f = static_cast<int>(m.face_entity.size());
      m.face_entity.push_back(static_cast<int>(k));
      m.face_index.push_back(static_cast<int>(i));
      m.face_area.push_back(t.grid.areas({0.0, 1.0}, {0.0, 1.0})[0]);
      for (std::size_t q = 0; q < t.v0.size(); ++q) {
        m.v0.push_back(t.v0[q]);
        m.e1.push_back(t.e1[q]);
        m.e2.push_back(t.e2[q]);
        m.vn.push_back(t.vn[q]);
        m.vt.push_back(t.vt[q]);
        const Vec3 c = cross(t.e1[q], t.e2[q]);
        const double len = norm(c);
        m.ng.push_back(len > 0 ? c / len : c);
        m.area.push_back(0.5 * len);
        m.face.push_back(f);
      }
      m.face_first.push_back(static_cast<long long>(m.v0.size()));
      m.grids.push_back(std::move(t.grid));
      m.surfaces.push_back(bodies[k].faces[i].surface);
    }
  }
  m.cum_area.resize(m.area.size());
  double sum = 0;
  for (std::size_t q = 0; q < m.area.size(); ++q) m.cum_area[q] = (sum += m.area[q]);
  return m;
}

}  // namespace pt
