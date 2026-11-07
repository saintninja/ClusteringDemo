#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "clustering/core/matrix.hpp"

namespace clustering {

inline constexpr int kNoiseLabel = -1;

struct ClusteringResult {
  std::vector<int> labels;
  Matrix centers;
  std::size_t iterations = 0;
  bool converged = false;

  [[nodiscard]] std::size_t ClusterCount() const;
  [[nodiscard]] std::size_t NoiseCount() const;
};

class Clusterer {
 public:
  virtual ~Clusterer() = default;

  [[nodiscard]] virtual ClusteringResult Fit(const Matrix& data) = 0;
  [[nodiscard]] virtual std::string Describe() const = 0;
};

}
