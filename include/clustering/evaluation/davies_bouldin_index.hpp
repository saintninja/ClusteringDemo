#pragma once

#include <optional>
#include <span>

#include "clustering/core/matrix.hpp"
#include "clustering/distance/distance_metric.hpp"

namespace clustering {

[[nodiscard]] std::optional<double> DaviesBouldinIndex(const Matrix& data,
                                                       std::span<const int> labels,
                                                       const DistanceMetric& metric);

}
