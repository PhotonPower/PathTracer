// Materials: how a surface scatters light (see the Python kernel: pt_bsdf.py). All directions
// point away from the surface point; n is the surface normal on the front side.
#include "pathtracer/pathtracer.hpp"

#include <algorithm>
#include <stdexcept>

namespace pt {

const char* const kKindNames[8] = {"absorber", "diffuse", "mirror", "glossy", "glass", "detector", "metal", "frosted"};

namespace {

/// Reflectance of a smooth dielectric interface for unpolarised light; eta = index on the
/// arriving side / index beyond. 1 for total reflection.
double fresnel(double cos, double eta) {
  const double sin2 = eta * eta * (1 - cos * cos);
  if (sin2 >= 1) return 1.0;
  const double cos_t = std::sqrt(std::max(0.0, 1 - sin2));
  const double rs = (eta * cos - cos_t) / (eta * cos + cos_t), rp = (cos - eta * cos_t) / (cos + eta * cos_t);
  return 0.5 * (rs * rs + rp * rp);
}

double ggx(double alpha, double cos_h) {
  const double d = cos_h * cos_h * (alpha * alpha - 1) + 1;
  return alpha * alpha / (kPi * d * d);
}

double g1(double alpha, double cos) {
  const double c = std::abs(cos);
  return 2 * c / (c + std::sqrt(alpha * alpha + (1 - alpha * alpha) * c * c));
}

/// A microfacet normal as wo sees them (Heitz 2018); n: the surface normal on the side of wo.
Vec3 visible_normal(double alpha, const Vec3& n, const Vec3& wo, double u1, double u2) {
  Vec3 a, b;
  frame(n, a, b);
  Vec3 v = normalized({alpha * dot(wo, a), alpha * dot(wo, b), dot(wo, n)});
  const double lensq = v.x * v.x + v.y * v.y;
  const Vec3 t1 = lensq > 0 ? Vec3{-v.y, v.x, 0} / std::sqrt(lensq) : Vec3{1, 0, 0};
  const Vec3 t2 = cross(v, t1);
  const double r = std::sqrt(u1), phi = 2 * kPi * u2;
  const double p1 = r * std::cos(phi);
  double p2 = r * std::sin(phi);
  const double s = 0.5 * (1 + v.z);
  p2 = (1 - s) * std::sqrt(std::max(0.0, 1 - p1 * p1)) + s * p2;
  const Vec3 nh = t1 * p1 + t2 * p2 + v * std::sqrt(std::max(0.0, 1 - p1 * p1 - p2 * p2));
  const Vec3 h = normalized({alpha * nh.x, alpha * nh.y, std::max(nh.z, 0.0)});
  return a * h.x + b * h.y + n * h.z;
}

Vec3 cosine(const Vec3& axis, double u1, double u2, double& cos) {
  cos = std::sqrt(1 - u1);
  return around(axis, cos, 2 * kPi * u2);
}

}  // namespace

Materials::Materials(std::vector<Material> rows) : rows_(std::move(rows)) {
  if (rows_.empty()) throw std::invalid_argument("no materials");
  for (std::size_t k = 0; k < rows_.size(); ++k) {
    Material& r = rows_[k];
    const std::string who = "material '" + r.name + "'";
    if (r.name.empty()) throw std::invalid_argument("a material has no name");
    for (std::size_t j = 0; j < k; ++j)
      if (rows_[j].name == r.name) throw std::invalid_argument(who + " is defined twice");
    if (r.kind < 0 || r.kind > kFrosted) throw std::invalid_argument(who + ": unknown kind");
    const bool glassy = r.kind == kGlass || r.kind == kFrosted;
    if (r.kind != kDiffuse) r.transmittance = 0;
    if (r.reflectance < 0 || r.transmittance < 0 || r.reflectance + r.transmittance > 1 + 1e-12)
      throw std::invalid_argument(who + ": reflectance + transmittance must lie in 0 … 1");
    if ((r.kind == kGlossy || r.kind == kMetal || r.kind == kFrosted) && !(r.roughness > 0 && r.roughness <= 1))
      throw std::invalid_argument(who + ": roughness must lie in 0 … 1 (not 0)");
    if (glassy && r.ior < 1) throw std::invalid_argument(who + ": refractive index below 1");
    if (r.absorption < 0) throw std::invalid_argument(who + ": negative absorption");
    if (!glassy) r.abbe = 0;
    if (r.abbe < 0) throw std::invalid_argument(who + ": negative Abbe number");
    if (r.kind == kAbsorber || r.kind == kDetector) r.spectrum.clear();
    std::sort(r.spectrum.begin(), r.spectrum.end());
    for (const auto& [nm, v] : r.spectrum)
      if (nm <= 0 || v < 0 || (!glassy && v + r.transmittance > 1 + 1e-12))
        throw std::invalid_argument(who + ": its spectrum needs wavelengths above 0 and values " +
                                    (glassy ? "that are not negative" : "in 0 … 1 (with the transmittance)"));
    const double rough = std::max(r.roughness, 1e-3);
    exponent_.push_back(2.0 / (rough * rough) - 2.0);
    alpha_.push_back(std::max(r.roughness * r.roughness, 1e-4));
  }
  set_wavelengths({587.56});
}

void Materials::set_wavelengths(const std::vector<double>& nm) {
  channels_ = static_cast<int>(nm.size());
  const std::size_t n = rows_.size();
  ior_.assign(n * channels_, 1.0);
  rho_.assign(n * channels_, 0.0);
  absorb_.assign(n * channels_, 0.0);
  const double f = 0.48613, d = 0.58756, c = 0.65627;  // F, d and C line / µm
  for (std::size_t m = 0; m < n; ++m) {
    const Material& r = rows_[m];
    const double b = r.abbe > 0 ? (r.ior - 1) / (r.abbe * (1 / (f * f) - 1 / (c * c))) : 0.0;
    for (int k = 0; k < channels_; ++k) {
      const double lam = nm[k] / 1000.0;
      ior_[m * channels_ + k] = r.ior + b * (1 / (lam * lam) - 1 / (d * d));
      double value = glassy(static_cast<int>(m)) ? r.absorption : r.reflectance;
      if (!r.spectrum.empty()) {  // straight lines between the pairs, the first / last value outside
        const auto& sp = r.spectrum;
        if (nm[k] <= sp.front().first)
          value = sp.front().second;
        else if (nm[k] >= sp.back().first)
          value = sp.back().second;
        else
          for (std::size_t i = 0; i + 1 < sp.size(); ++i)
            if (nm[k] <= sp[i + 1].first) {
              const double w = (nm[k] - sp[i].first) / (sp[i + 1].first - sp[i].first);
              value = sp[i].second + w * (sp[i + 1].second - sp[i].second);
              break;
            }
      }
      (glassy(static_cast<int>(m)) ? absorb_ : rho_)[m * channels_ + k] = value;
      if (glassy(static_cast<int>(m))) rho_[m * channels_ + k] = r.reflectance;
    }
  }
}

void Materials::eval(int m, const Vec3& n, const Vec3& wo, const Vec3& wi, int channel, bool radiance,
                     double outside, double& f, double& p_fwd, double& p_rev) const {
  f = p_fwd = p_rev = 0;
  const int kind = rows_[m].kind;
  const double co = dot(n, wo), ci = dot(n, wi);
  const bool same = co * ci > 0;
  if (kind == kDiffuse) {
    const double rho_ = rho(m, channel), tau = rows_[m].transmittance;
    const double part = same ? rho_ : tau, total = rho_ + tau;
    const double pick = total > 0 ? part / total : 0.0;
    f = part / kPi;
    p_fwd = pick * std::abs(ci) / kPi;
    p_rev = pick * std::abs(co) / kPi;
  } else if (kind == kGlossy) {
    if (!same) return;
    const double e = exponent_[m];
    // cosine between wi and the mirror direction of wo (symmetric in wo and wi)
    const double c = std::clamp(2 * co * ci - dot(wo, wi), 0.0, 1.0);
    const double lobe = std::pow(c, e);
    f = rho(m, channel) * (e + 2) / (2 * kPi) * lobe;
    p_fwd = p_rev = (e + 1) / (2 * kPi) * lobe;
  } else if (kind == kMetal) {
    if (!same) return;
    const double alpha = alpha_[m];
    const Vec3 h = normalized(wo + wi);
    const double dist = ggx(alpha, dot(n, h)), g_o = g1(alpha, co), g_i = g1(alpha, ci);
    const double f0 = rho(m, channel);
    const double fr = f0 + (1 - f0) * std::pow(1 - std::clamp(dot(wo, h), 0.0, 1.0), 5);
    f = fr * dist * g_o * g_i / (4 * std::abs(co * ci));
    p_fwd = g_o * dist / (4 * std::abs(co));
    p_rev = g_i * dist / (4 * std::abs(ci));
  } else if (kind == kFrosted) {
    if (co == 0 || ci == 0) return;
    const double alpha = alpha_[m], index = ior(m, channel);
    const double eta_o = co > 0 ? outside : index, eta_i = ci > 0 ? outside : index;  // on the side of wo, of wi
    const double beyond = co > 0 ? index : outside;                                 // beyond wo's side
    Vec3 h = same ? wo + wi : wo * eta_o + wi * eta_i;
    const double length = norm(h);
    if (!(length > 1e-12)) return;
    h = h / length;
    if (dot(h, n) * co < 0) h = -h;  // onto wo's side
    const double woh = dot(wo, h), wih = dot(wi, h);
    if (!(woh > 0) || !(same ? wih > 0 : wih < 0)) return;
    const double fr = fresnel(std::clamp(woh, 0.0, 1.0), eta_o / beyond);
    const double dist = ggx(alpha, dot(n, h)), g_o = g1(alpha, co), g_i = g1(alpha, ci);
    const double aco = std::abs(co), aci = std::abs(ci);
    if (same) {
      f = fr * dist * g_o * g_i / (4 * aco * aci);
      p_fwd = fr * g_o * dist / (4 * aco);
      p_rev = fr * g_i * dist / (4 * aci);
    } else {
      const double den = eta_o * woh + eta_i * wih;
      const double through = den != 0 ? (1 - fr) * dist * std::abs(woh * wih) / (den * den) : 0.0;
      const double scale = radiance ? eta_o : eta_i;
      f = through * g_o * g_i / (aco * aci) * scale * scale;
      p_fwd = through * g_o * eta_i * eta_i / aco;
      p_rev = through * g_i * eta_o * eta_o / aci;
    }
  }
}

void Materials::sample(int m, const Vec3& n, const Vec3& wo, Rng& rng, bool radiance, int channel, double outside,
                       bool straight, Vec3& wi, double& weight, double& pdf) const {
  const double u1 = rng.next(), u2 = rng.next(), u3 = rng.next();
  wi = {};
  weight = pdf = 0;
  if (straight) {  // a surface between two parts of the same medium
    wi = -wo;
    weight = 1;
    return;
  }
  const int kind = rows_[m].kind;
  const double co = dot(n, wo);
  const double side = co >= 0 ? 1.0 : -1.0;
  if (kind == kDiffuse) {
    const double rho_ = rho(m, channel), tau = rows_[m].transmittance, total = rho_ + tau;
    const bool reflect = u3 * total < rho_;
    double cos;
    wi = cosine(n * (side * (reflect ? 1.0 : -1.0)), u1, u2, cos);
    weight = total;
    pdf = total > 0 ? (reflect ? rho_ : tau) / total * cos / kPi : 0.0;
  } else if (kind == kMirror) {
    wi = n * (2 * co) - wo;
    weight = rho(m, channel);
  } else if (kind == kGlossy) {
    const double e = exponent_[m];
    const Vec3 mirror = n * (2 * co) - wo;
    const double cos_a = std::pow(u1, 1.0 / (e + 1));
    wi = around(mirror, cos_a, 2 * kPi * u2);
    const double ci = dot(n, wi);
    weight = ci * co > 0 ? rho(m, channel) * (e + 2) / (e + 1) * std::abs(ci) : 0.0;
    pdf = (e + 1) / (2 * kPi) * std::pow(cos_a, e);
  } else if (kind == kMetal || kind == kFrosted) {  // a microfacet seen from wo, then its mirror / lens
    const double alpha = alpha_[m];
    const Vec3 nn = n * side;
    const Vec3 h = visible_normal(alpha, nn, wo, u1, u2);
    const double woh = std::clamp(dot(wo, h), 0.0, 1.0);
    const double aco = std::max(std::abs(co), 1e-300);
    const double dist = ggx(alpha, dot(nn, h)), g_o = g1(alpha, co);
    const bool metal = kind == kMetal;
    const double index = ior(m, channel);
    const double eta = co >= 0 ? outside / index : index / outside;  // n(side of wo) / n(other side)
    const double f0 = rho(m, channel);
    const double fr = metal ? f0 + (1 - f0) * std::pow(1 - woh, 5) : fresnel(woh, eta);
    const bool reflect = metal || u3 < fr;
    const double cos_t = std::sqrt(std::max(0.0, 1 - eta * eta * (1 - woh * woh)));
    wi = reflect ? h * (2 * woh) - wo : wo * (-eta) + h * (eta * woh - cos_t);
    const double ci = dot(nn, wi);
    const bool ok = (reflect ? ci > 0 : ci < 0) && woh > 0;
    weight = ok ? g1(alpha, ci) * (metal ? fr : (reflect || !radiance ? 1.0 : eta * eta)) : 0.0;
    if (reflect) {
      pdf = (metal ? 1.0 : fr) * g_o * dist / (4 * aco);
    } else {
      const double eta_o = co >= 0 ? outside : index, eta_i = co >= 0 ? index : outside;
      const double den = eta_o * woh - eta_i * cos_t;
      pdf = den != 0 ? (1 - fr) * g_o * woh * dist / aco * eta_i * eta_i * cos_t / (den * den) : 0.0;
    }
  } else if (kind == kGlass) {
    const double index = ior(m, channel);
    const double eta = co >= 0 ? outside / index : index / outside;
    const double cos_o = std::abs(co);
    const double cos_t = std::sqrt(std::max(0.0, 1 - eta * eta * (1 - cos_o * cos_o)));
    const bool reflect = u3 < fresnel(cos_o, eta);
    const Vec3 nn = n * side;
    wi = reflect ? nn * (2 * cos_o) - wo : wo * (-eta) + nn * (eta * cos_o - cos_t);
    weight = reflect || !radiance ? 1.0 : eta * eta;
  }
}

}  // namespace pt
