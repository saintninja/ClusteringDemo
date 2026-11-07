#pragma once

#include "clustering/distance/distance_metric.hpp"

namespace clustering {

class EuclideanDistance final : public DistanceMetric {
 public:
  [[nodiscard]] double Distance(Point first, Point second) const override;
  [[nodiscard]] std::string_view Name() const override { return "euclidean"; }
};

}
