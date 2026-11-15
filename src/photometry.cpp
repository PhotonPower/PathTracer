// Directional distributions of measured sources and the eye's sensitivity (pt_photometry.py).
#include "pathtracer/pathtracer.hpp"

#include <algorithm>
#include <stdexcept>

namespace pt {
namespace {

double interp(const std::vector<double>& x, const double* y, double at, bool zero_outside) {
  if (at < x.front()) return zero_outside ? 0.0 : y[0];
  if (at > x.back()) return zero_outside ? 0.0 : y[x.size() - 1];
  const std::size_t k = std::min<std::size_t>(std::upper_bound(x.begin(), x.end(), at) - x.begin(), x.size() - 1);
  if (k == 0) return y[0];
  const double w = (at - x[k - 1]) / (x[k] - x[k - 1]);
  return y[k - 1] + w * (y[k] - y[k - 1]);
}

}  // namespace

Distribution::Distribution(const std::vector<double>& theta, const std::vector<double>& phi,
                           const std::vector<double>& intensity) {
  const std::size_t nt = theta.size(), np = phi.size();
  if (nt < 2 || np < 1 || intensity.size() != nt * np)
    throw std::invalid_argument("distribution: angles and intensities do not match");
  for (std::size_t k = 1; k < nt; ++k)
    if (!(theta[k] > theta[k - 1])) throw std::invalid_argument("distribution: angles must rise");
  for (double v : intensity)
    if (v < 0) throw std::invalid_argument("distribution: intensities must not be negative");
  cells_p_ = np == 1 ? 1 : 180;
  std::vector<double> table(static_cast<std::size_t>(cells_t_) * cells_p_);
  cos_edges_.resize(cells_t_ + 1);
  for (int i = 0; i <= cells_t_; ++i) cos_edges_[i] = std::cos(kPi * i / cells_t_);
  std::vector<double> column(np);
  for (int it = 0; it < cells_t_; ++it) {
    const double centre = (it + 0.5) * 180.0 / cells_t_;
    for (std::size_t p = 0; p < np; ++p) column[p] = interp(theta, &intensity[p * nt], centre, true);
    for (int ip = 0; ip < cells_p_; ++ip) {
      double value = column[0];
      if (np > 1) {  // around the full turn: between the planes, the last joined to the first
        const double at = std::fmod((ip + 0.5) * 360.0 / cells_p_, 360.0);
        std::size_t hi = std::upper_bound(phi.begin(), phi.end(), at) - phi.begin();
        const std::size_t lo = hi == 0 ? np - 1 : hi - 1;
        hi = hi == np ? 0 : hi;
        double x0 = phi[lo], x1 = phi[hi];
        if (x0 > at) x0 -= 360.0;
        if (x1 < at) x1 += 360.0;
        const double w = x1 > x0 ? (at - x0) / (x1 - x0) : 0.0;
        value = column[lo] + w * (column[hi] - column[lo]);
      }
      table[static_cast<std::size_t>(it) * cells_p_ + ip] = value;
    }
  }
  double total = 0;
  for (int it = 0; it < cells_t_; ++it)
    for (int ip = 0; ip < cells_p_; ++ip)
      total += table[static_cast<std::size_t>(it) * cells_p_ + ip] * (cos_edges_[it] - cos_edges_[it + 1]) *
               (2 * kPi / cells_p_);
  if (!(total > 0)) throw std::invalid_argument("distribution: no light in any direction");
  density_.resize(table.size());
  cdf_.resize(table.size());
  double sum = 0;
  for (int it = 0; it < cells_t_; ++it)
    for (int ip = 0; ip < cells_p_; ++ip) {
      const std::size_t k = static_cast<std::size_t>(it) * cells_p_ + ip;
      density_[k] = table[k] / total;
      cdf_[k] = (sum += density_[k] * (cos_edges_[it] - cos_edges_[it + 1]) * (2 * kPi / cells_p_));
    }
}

double Distribution::eval(double cos, double phi) const {
  const double theta = std::acos(std::clamp(cos, -1.0, 1.0)) * 180.0 / kPi;
  const int it = std::min(static_cast<int>(theta * cells_t_ / 180.0), cells_t_ - 1);
  double turn = std::fmod(phi, 2 * kPi);
  if (turn < 0) turn += 2 * kPi;
  const int ip = std::min(static_cast<int>(turn * cells_p_ / (2 * kPi)), cells_p_ - 1);
  return density_[static_cast<std::size_t>(it) * cells_p_ + ip];
}

void Distribution::sample(Rng& rng, double& cos, double& phi, double& density) const {
  const std::size_t cell = std::min<std::size_t>(
      std::upper_bound(cdf_.begin(), cdf_.end(), rng.next() * cdf_.back()) - cdf_.begin(), cdf_.size() - 1);
  const int it = static_cast<int>(cell / cells_p_), ip = static_cast<int>(cell % cells_p_);
  cos = cos_edges_[it] - rng.next() * (cos_edges_[it] - cos_edges_[it + 1]);
  phi = (ip + rng.next()) * (2 * kPi / cells_p_);
  density = density_[cell];
}

double v_lambda(double nm) {
  // CIE 1924 photopic luminous efficiency, 380 … 780 nm in steps of 10 nm, and the maximum at 555 nm
  static const double v[] = {0.000039, 0.00012, 0.000396, 0.00121, 0.0040, 0.0116, 0.023, 0.038, 0.060, 0.09098,
                             0.13902, 0.20802, 0.323, 0.503, 0.710, 0.862, 0.954, 0.99495, 0.995, 0.952, 0.870,
                             0.757, 0.631, 0.503, 0.381, 0.265, 0.175, 0.107, 0.061, 0.032, 0.017, 0.00821,
                             0.004102, 0.002091, 0.001047, 0.00052, 0.000249, 0.00012, 0.00006, 0.00003, 0.000015};
  if (nm < 380 || nm > 780) return 0.0;
  if (nm >= 550 && nm <= 560) return nm <= 555 ? v[17] + (1.0 - v[17]) * (nm - 550) / 5 : 1.0 + (v[18] - 1.0) * (nm - 555) / 5;
  const double x = (nm - 380) / 10.0;
  const int k = std::min(static_cast<int>(x), 39);
  return v[k] + (v[k + 1] - v[k]) * (x - k);
}

}  // namespace pt
