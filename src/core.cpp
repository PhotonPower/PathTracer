// Vectors, random numbers and NURBS evaluation.
#include "pathtracer/pathtracer.hpp"

#include <algorithm>
#include <stdexcept>

namespace pt {

void frame(const Vec3& axis, Vec3& a, Vec3& b) {
  const Vec3 helper = std::abs(axis.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
  a = normalized(cross(axis, helper));
  b = cross(axis, a);
}

Vec3 around(const Vec3& axis, double cos, double phi) {
  Vec3 a, b;
  frame(axis, a, b);
  const double sin = std::sqrt(std::max(0.0, 1 - cos * cos));
  return a * (sin * std::cos(phi)) + b * (sin * std::sin(phi)) + axis * cos;
}

static std::uint64_t splitmix(std::uint64_t& x) {
  std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

Rng::Rng(std::uint64_t seed, std::uint64_t a, std::uint64_t b) {
  std::uint64_t x = seed * 0x9e3779b97f4a7c15ULL + a * 0xd1b54a32d192ed03ULL + b * 0x8cb92ba72f3d8dd7ULL + 0x2545f4914f6cdd1dULL;
  for (auto& v : s) v = splitmix(x);
}

std::uint64_t Rng::bits() {
  auto rotl = [](std::uint64_t v, int k) { return (v << k) | (v >> (64 - k)); };
  const std::uint64_t result = rotl(s[0] + s[3], 23) + s[0];
  const std::uint64_t t = s[1] << 17;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = rotl(s[3], 45);
  return result;
}

// ---------------------------------------------------------------------------- NURBS
int basis_functions(int degree, const std::vector<double>& k, double u, double* N, double* dN) {
  const int n = static_cast<int>(k.size()) - degree - 1;
  int span = static_cast<int>(std::upper_bound(k.begin(), k.end(), u) - k.begin()) - 1;
  span = std::clamp(span, degree, n - 1);
  constexpr int kMax = 32;
  if (degree >= kMax) throw std::invalid_argument("NURBS degree too high");
  double left[kMax], right[kMax], lower[kMax];
  N[0] = 1.0;
  for (int r = 0; r < degree; ++r) lower[r] = 0.0;
  if (degree == 0) {
    if (dN) dN[0] = 0.0;
    return span;
  }
  for (int j = 1; j <= degree; ++j) {
    if (j == degree)
      for (int r = 0; r < degree; ++r) lower[r] = N[r];  // the functions of degree - 1
    left[j] = u - k[span + 1 - j];
    right[j] = k[span + j] - u;
    double saved = 0.0;
    for (int r = 0; r < j; ++r) {
      const double den = right[r + 1] + left[j - r];
      const double temp = den != 0.0 ? N[r] / den : 0.0;
      N[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    N[j] = saved;
  }
  if (dN) {
    for (int r = 0; r <= degree; ++r) {
      double v = 0.0;
      if (r > 0) {
        const double den = k[span + r] - k[span + r - degree];
        if (den != 0.0) v += degree * lower[r - 1] / den;
      }
      if (r < degree) {
        const double den = k[span + r + 1] - k[span + r - degree + 1];
        if (den != 0.0) v -= degree * lower[r] / den;
      }
      dN[r] = v;
    }
  }
  return span;
}

static std::vector<double> distinct(int degree, const std::vector<double>& k) {
  std::vector<double> out(k.begin() + degree, k.end() - degree);
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<double> Curve::spans() const { return distinct(degree, knots); }

Vec3 Curve::at(double u) const {
  double N[33];
  const int span = basis_functions(degree, knots, u, N, nullptr);
  Vec3 p;
  double w = 0;
  for (int r = 0; r <= degree; ++r) {
    const int i = span - degree + r;
    const double wi = weights.empty() ? 1.0 : weights[i];
    p += points[i] * (N[r] * wi);
    w += N[r] * wi;
  }
  return p / w;
}

void Surface::check() const {
  const auto n = static_cast<std::size_t>(num_u) * static_cast<std::size_t>(num_v);
  if (num_u < 1 || num_v < 1 || points.size() != n || (!weights.empty() && weights.size() != n) ||
      knots_u.size() != static_cast<std::size_t>(num_u + degree_u + 1) ||
      knots_v.size() != static_cast<std::size_t>(num_v + degree_v + 1))
    throw std::invalid_argument("control points do not match the knot vectors");
}

std::array<double, 4> Surface::domain() const {
  return {knots_u[degree_u], knots_u[knots_u.size() - degree_u - 1], knots_v[degree_v],
          knots_v[knots_v.size() - degree_v - 1]};
}

std::vector<double> Surface::spans_u() const { return distinct(degree_u, knots_u); }
std::vector<double> Surface::spans_v() const { return distinct(degree_v, knots_v); }

void Surface::at(double u, double v, Vec3& s, Vec3& su, Vec3& sv) const {
  double nu[33], du[33], nv[33], dv[33];
  const int iu = basis_functions(degree_u, knots_u, u, nu, du) - degree_u;
  const int iv = basis_functions(degree_v, knots_v, v, nv, dv) - degree_v;
  Vec3 a, au, av;
  double w = 0, wu = 0, wv = 0;
  for (int r = 0; r <= degree_u; ++r) {
    for (int c = 0; c <= degree_v; ++c) {
      const std::size_t i = static_cast<std::size_t>(iu + r) * num_v + (iv + c);
      const double wi = weights.empty() ? 1.0 : weights[i];
      const Vec3 pw = points[i] * wi;
      const double b = nu[r] * nv[c], bu = du[r] * nv[c], bv = nu[r] * dv[c];
      a += pw * b;
      au += pw * bu;
      av += pw * bv;
      w += wi * b;
      wu += wi * bu;
      wv += wi * bv;
    }
  }
  s = a / w;
  su = (au - s * wu) / w;
  sv = (av - s * wv) / w;
}

}  // namespace pt
