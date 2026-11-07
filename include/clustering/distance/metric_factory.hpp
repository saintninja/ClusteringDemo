#pragma once

#include <optional>
#include <string_view>

#include "clustering/distance/distance_metric.hpp"

namespace clustering {

enum class MetricKind { kEuclidean, kManhattan, kMahalanobis };

[[nodiscard]] std::optional<MetricKind> ParseMetricKind(std::string_view name);
[[nodiscard]] MetricPtr MakeMetric(MetricKind kind, const Matrix& data);

}
