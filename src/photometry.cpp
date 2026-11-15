// Directional distributions of measured sources and the eye's sensitivity (pt_photometry.py).
#include "pathtracer/pathtracer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <stdexcept>
#include <string>

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

namespace {

std::vector<std::string> lines_of(const std::string& text) {
  std::vector<std::string> out;
  std::string line;
  for (char c : text) {
    if (c == '\r') continue;
    if (c == '\n') {
      out.push_back(line);
      line.clear();
    } else {
      line.push_back(c);
    }
  }
  out.push_back(line);
  return out;
}

std::string trimmed(const std::string& s) {
  const auto a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t");
  return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

std::string upper(std::string s) {
  for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return s;
}

double number(const std::string& token, const char* what) {
  std::string t = token;
  std::replace(t.begin(), t.end(), ',', '.');
  std::size_t used = 0;
  double v = 0;
  try {
    v = std::stod(t, &used);
  } catch (const std::exception&) {
    throw std::invalid_argument(what);
  }
  if (used != t.size()) throw std::invalid_argument(what);
  return v;
}

/// Planes 0 … 90 or 0 … 180 mirrored to the full turn (planes 0 … 360 as they are).
void all_around(std::vector<double>& phi, std::vector<double>& cd, std::size_t nt) {
  auto row = [&](std::size_t k) { return std::vector<double>(cd.begin() + k * nt, cd.begin() + (k + 1) * nt); };
  double last = phi.back();
  if (std::abs(last - 90.0) < 1e-6) {  // a quarter: mirror at the 90° plane …
    for (std::size_t k = phi.size() - 1; k-- > 0;) {
      const auto r = row(k);
      phi.push_back(180.0 - phi[k]);
      cd.insert(cd.end(), r.begin(), r.end());
    }
    last = 180.0;
  }
  if (std::abs(last - 180.0) < 1e-6) {  // … and a half at the 0 – 180° plane
    for (std::size_t k = phi.size() - 1; k-- > 1;) {
      const auto r = row(k);
      phi.push_back(360.0 - phi[k]);
      cd.insert(cd.end(), r.begin(), r.end());
    }
  } else if (std::abs(last - 360.0) < 1e-6) {
    phi.pop_back();
    cd.resize(phi.size() * nt);
  } else {
    throw std::invalid_argument("luminaire file: planes up to " + std::to_string(last) + "° – expected 0, 90, 180 or 360");
  }
}

}  // namespace

LuminaireTable table_ies(const std::string& text) {
  const auto lines = lines_of(text);
  std::size_t tilt = lines.size();
  for (std::size_t k = 0; k < lines.size(); ++k)
    if (upper(trimmed(lines[k])).rfind("TILT", 0) == 0) {
      tilt = k;
      break;
    }
  if (tilt == lines.size()) throw std::invalid_argument("IES file: no TILT line");
  const auto eq = lines[tilt].find('=');
  const std::string kind = eq == std::string::npos ? std::string() : upper(trimmed(lines[tilt].substr(eq + 1)));
  std::vector<double> numbers;
  for (std::size_t k = tilt + 1; k < lines.size(); ++k) {
    std::string line = lines[k];
    std::replace(line.begin(), line.end(), ',', ' ');
    std::size_t at = 0;
    while (at < line.size()) {
      const auto a = line.find_first_not_of(" \t", at);
      if (a == std::string::npos) break;
      const auto b = line.find_first_of(" \t", a);
      numbers.push_back(number(line.substr(a, b == std::string::npos ? b : b - a), "IES file: text among the numbers"));
      at = b == std::string::npos ? line.size() : b;
    }
  }
  std::size_t first = 0;
  if (kind == "INCLUDE") {  // lamp geometry, number of pairs, angles, factors
    if (numbers.size() < 2) throw std::invalid_argument("IES file: too short");
    first = 2 + 2 * static_cast<std::size_t>(numbers[1]);
  } else if (kind != "NONE") {
    throw std::invalid_argument("IES file: TILT=" + kind + " is not supported");
  }
  if (numbers.size() < first + 13) throw std::invalid_argument("IES file: too short");
  const auto nt = static_cast<std::size_t>(numbers[first + 3]), np = static_cast<std::size_t>(numbers[first + 4]);
  if (static_cast<int>(numbers[first + 5]) != 1) throw std::invalid_argument("IES file: only photometry of type C is supported");
  const std::size_t data = first + 13;
  if (numbers.size() < data + nt + np + nt * np) throw std::invalid_argument("IES file: fewer values than its angles need");
  LuminaireTable t;
  t.theta.assign(numbers.begin() + data, numbers.begin() + data + nt);
  t.phi.assign(numbers.begin() + data + nt, numbers.begin() + data + nt + np);
  t.intensity.assign(numbers.begin() + data + nt + np, numbers.begin() + data + nt + np + nt * np);
  if (np > 1) all_around(t.phi, t.intensity, nt);
  return t;
}

LuminaireTable table_ldt(const std::string& text) {
  auto lines = lines_of(text);
  for (auto& l : lines) l = trimmed(l);
  const char* bad = "EULUMDAT file: not in the expected form";
  auto at = [&](std::size_t k) -> const std::string& {
    if (k >= lines.size()) throw std::invalid_argument(bad);
    return lines[k];
  };
  const int symmetry = static_cast<int>(number(at(2), bad)), planes = static_cast<int>(number(at(3), bad));
  const int ng = static_cast<int>(number(at(5), bad)), sets = static_cast<int>(number(at(25), bad));
  if (symmetry < 0 || symmetry > 4 || planes < 1 || ng < 2 || sets < 0) throw std::invalid_argument(bad);
  std::size_t pos = 26 + 6 * static_cast<std::size_t>(sets) + 10;
  std::vector<double> c(planes), gamma(ng);
  for (int k = 0; k < planes; ++k) c[k] = number(at(pos + k), bad);
  pos += planes;
  for (int k = 0; k < ng; ++k) gamma[k] = number(at(pos + k), bad);
  pos += ng;
  const int given = symmetry == 0 ? planes : symmetry == 1 ? 1 : symmetry == 4 ? planes / 4 + 1 : planes / 2 + 1;
  std::vector<double> cd(static_cast<std::size_t>(given) * ng);
  for (std::size_t k = 0; k < cd.size(); ++k) cd[k] = number(at(pos + k), bad);
  LuminaireTable t;
  t.theta = gamma;
  if (symmetry == 1) {
    t.phi = {0.0};
    t.intensity = cd;
  } else if (symmetry == 0) {
    t.phi = c;
    t.intensity = cd;
  } else if (symmetry == 3) {  // planes C90 … C270 given: mirror at C90 – C270
    int first = 0;
    for (int k = 0; k < planes; ++k)
      if (std::abs(c[k] - 90.0) < std::abs(c[first] - 90.0)) first = k;
    t.phi = c;
    for (double a : c) {
      const double want = (a >= 90.0 && a <= 270.0) ? a : std::fmod(180.0 - a + 360.0, 360.0);
      int best = 0;
      for (int k = 0; k < given; ++k)
        if (std::abs(c[first + k] - want) < std::abs(c[first + best] - want)) best = k;
      t.intensity.insert(t.intensity.end(), cd.begin() + static_cast<std::size_t>(best) * ng,
                         cd.begin() + static_cast<std::size_t>(best + 1) * ng);
    }
  } else {  // C0 … C180 or C0 … C90
    t.phi.assign(c.begin(), c.begin() + given);
    t.intensity = cd;
    all_around(t.phi, t.intensity, static_cast<std::size_t>(ng));
  }
  return t;
}

LuminaireTable load_luminaire(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::invalid_argument("luminaire file '" + path + "': cannot be read");
  std::string text;
  char buffer[65536];
  for (std::size_t n; (n = std::fread(buffer, 1, sizeof buffer, f)) > 0;) text.append(buffer, n);
  std::fclose(f);
  const std::string low = upper(path);
  const bool ends_ldt = low.size() >= 4 && low.compare(low.size() - 4, 4, ".LDT") == 0;
  const bool ends_ies = low.size() >= 4 && low.compare(low.size() - 4, 4, ".IES") == 0;
  if (ends_ldt || (!ends_ies && upper(text).find("TILT") == std::string::npos)) return table_ldt(text);
  return table_ies(text);
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
