// Bounding volume hierarchy over triangles of a mesh, built with the surface area heuristic
// (16 slices along the longest axis of the triangle centres; the median where no cut separates).
#include "pathtracer/pathtracer.hpp"

#include <algorithm>
#include <numeric>

namespace pt {
namespace {
constexpr int kLeaf = 4, kBins = 16;
inline Vec3 vmin(const Vec3& a, const Vec3& b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
inline Vec3 vmax(const Vec3& a, const Vec3& b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }
inline double surface(const Vec3& lo, const Vec3& hi) {
  const Vec3 d{std::max(0.0, hi.x - lo.x), std::max(0.0, hi.y - lo.y), std::max(0.0, hi.z - lo.z)};
  return d.x * d.y + d.y * d.z + d.z * d.x;
}
}  // namespace

Bvh::Bvh(const Mesh& mesh, std::vector<long long> tris) {
  const std::size_t n = tris.size();
  if (n == 0) return;
  std::vector<Vec3> lo(n), hi(n), centre(n);
  for (std::size_t k = 0; k < n; ++k) {
    const Vec3 a = mesh.v0[tris[k]], b = a + mesh.e1[tris[k]], c = a + mesh.e2[tris[k]];
    lo[k] = vmin(vmin(a, b), c);
    hi[k] = vmax(vmax(a, b), c);
    centre[k] = (a + b + c) / 3.0;
  }
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  node_.reserve(2 * n);
  build(order, 0, static_cast<int>(n), lo, hi, centre, 1);
  tri_.resize(n);
  v0_.resize(n);
  e1_.resize(n);
  e2_.resize(n);
  for (std::size_t slot = 0; slot < n; ++slot) {
    tri_[slot] = tris[order[slot]];
    v0_[slot] = mesh.v0[tri_[slot]];
    e1_[slot] = mesh.e1[tri_[slot]];
    e2_[slot] = mesh.e2[tri_[slot]];
  }
}

int Bvh::build(std::vector<int>& order, int start, int end, const std::vector<Vec3>& lo, const std::vector<Vec3>& hi,
               const std::vector<Vec3>& centre, int depth) {
  const int index = static_cast<int>(node_.size());
  node_.emplace_back();
  depth_ = std::max(depth_, depth);
  Vec3 blo{kInf, kInf, kInf}, bhi{-kInf, -kInf, -kInf}, clo = blo, chi = bhi;
  for (int k = start; k < end; ++k) {
    blo = vmin(blo, lo[order[k]]);
    bhi = vmax(bhi, hi[order[k]]);
    clo = vmin(clo, centre[order[k]]);
    chi = vmax(chi, centre[order[k]]);
  }
  node_[index].lo = blo;
  node_[index].hi = bhi;
  const int count = end - start;
  if (count <= kLeaf || depth > 60) {
    node_[index].start = start;
    node_[index].count = count;
    return index;
  }
  const Vec3 ext = chi - clo;
  const int axis = ext.x >= ext.y && ext.x >= ext.z ? 0 : ext.y >= ext.z ? 1 : 2;
  int mid = start + count / 2;
  bool cut = false;
  if (ext[axis] > 0) {
    int number[kBins] = {};
    Vec3 bin_lo[kBins], bin_hi[kBins];
    for (int b = 0; b < kBins; ++b) {
      bin_lo[b] = {kInf, kInf, kInf};
      bin_hi[b] = {-kInf, -kInf, -kInf};
    }
    auto bin_of = [&](int item) {
      return std::clamp(static_cast<int>((centre[item][axis] - clo[axis]) / ext[axis] * kBins), 0, kBins - 1);
    };
    for (int k = start; k < end; ++k) {
      const int b = bin_of(order[k]);
      ++number[b];
      bin_lo[b] = vmin(bin_lo[b], lo[order[k]]);
      bin_hi[b] = vmax(bin_hi[b], hi[order[k]]);
    }
    double right_area[kBins];
    Vec3 rlo{kInf, kInf, kInf}, rhi{-kInf, -kInf, -kInf};
    for (int b = kBins - 1; b > 0; --b) {
      rlo = vmin(rlo, bin_lo[b]);
      rhi = vmax(rhi, bin_hi[b]);
      right_area[b] = surface(rlo, rhi);
    }
    Vec3 llo{kInf, kInf, kInf}, lhi{-kInf, -kInf, -kInf};
    double best = kInf;
    int best_bin = -1, n_left = 0;
    for (int b = 0; b + 1 < kBins; ++b) {
      llo = vmin(llo, bin_lo[b]);
      lhi = vmax(lhi, bin_hi[b]);
      n_left += number[b];
      if (n_left == 0 || n_left == count) continue;
      const double cost = surface(llo, lhi) * n_left + right_area[b + 1] * (count - n_left);
      if (cost < best) {
        best = cost;
        best_bin = b;
      }
    }
    if (best_bin >= 0) {
      const auto middle = std::partition(order.begin() + start, order.begin() + end,
                                         [&](int item) { return bin_of(item) <= best_bin; });
      mid = static_cast<int>(middle - order.begin());
      cut = true;
    }
  }
  if (!cut)
    std::nth_element(order.begin() + start, order.begin() + mid, order.begin() + end,
                     [&](int a, int b) { return centre[a][axis] < centre[b][axis]; });
  const int left = build(order, start, mid, lo, hi, centre, depth + 1);
  const int right = build(order, mid, end, lo, hi, centre, depth + 1);
  node_[index].left = left;
  node_[index].right = right;
  return index;
}

}  // namespace pt
