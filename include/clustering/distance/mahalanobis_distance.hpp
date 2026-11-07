#pragma once

#include "clustering/distance/distance_metric.hpp"

namespace clustering {

class MahalanobisDistance final : public DistanceMetric {
 public:
  explicit MahalanobisDistance(Matrix inverse_covariance);

  [[nodiscard]] static MahalanobisDistance FitTo(const Matrix& data);

  [[nodiscard]] double Distance(Point first, Point second) const override;
  [[nodiscard]] std::string_view Name() const override { return "mahalanobis"; }

 private:
  Matrix inverse_covariance_;
};

}
