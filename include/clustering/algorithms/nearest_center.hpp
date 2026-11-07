#pragma once

#include <cstddef>

#include "clustering/core/matrix.hpp"
#include "clustering/distance/distance_metric.hpp"

namespace clustering {

[[nodiscard]] std::size_t NearestCenter(Point object, const Matrix& centers,
                                        const DistanceMetric& metric);

}
