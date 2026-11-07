#pragma once

#include <memory>
#include <string_view>

#include "clustering/core/matrix.hpp"

namespace clustering {

class DistanceMetric {
 public:
  virtual ~DistanceMetric() = default;

  [[nodiscard]] virtual double Distance(Point first, Point second) const = 0;
  [[nodiscard]] virtual std::string_view Name() const = 0;
};

using MetricPtr = std::shared_ptr<const DistanceMetric>;

}
