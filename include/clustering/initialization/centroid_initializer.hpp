#pragma once

#include <cstddef>
#include <string>

#include "clustering/core/matrix.hpp"
#include "clustering/distance/distance_metric.hpp"

namespace clustering {

class CentroidInitializer {
 public:
  virtual ~CentroidInitializer() = default;

  [[nodiscard]] virtual Matrix Initialize(const Matrix& data, std::size_t cluster_count,
                                          const DistanceMetric& metric) = 0;
  [[nodiscard]] virtual std::string Name() const = 0;
};

}
