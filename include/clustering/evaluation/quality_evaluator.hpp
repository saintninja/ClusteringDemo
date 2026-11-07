#pragma once

#include <optional>
#include <span>

#include "clustering/core/matrix.hpp"
#include "clustering/distance/distance_metric.hpp"

namespace clustering {

struct QualityReport {
  std::optional<double> adjusted_rand_index;
  std::optional<double> adjusted_mutual_information;
  std::optional<double> silhouette_coefficient;
  std::optional<double> davies_bouldin_index;
};

class QualityEvaluator {
 public:
  explicit QualityEvaluator(MetricPtr metric);

  [[nodiscard]] QualityReport Evaluate(const Matrix& data, std::span<const int> predicted_labels,
                                       std::span<const int> truth_labels = {}) const;

 private:
  MetricPtr metric_;
};

}
