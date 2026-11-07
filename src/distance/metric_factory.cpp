#include "clustering/distance/metric_factory.hpp"

#include <utility>

#include "clustering/distance/euclidean_distance.hpp"
#include "clustering/distance/mahalanobis_distance.hpp"
#include "clustering/distance/manhattan_distance.hpp"

namespace clustering {

std::optional<MetricKind> ParseMetricKind(std::string_view name) {
  if (name == "euclidean") return MetricKind::kEuclidean;
  if (name == "manhattan") return MetricKind::kManhattan;
  if (name == "mahalanobis") return MetricKind::kMahalanobis;
  return std::nullopt;
}

MetricPtr MakeMetric(MetricKind kind, const Matrix& data) {
  switch (kind) {
    case MetricKind::kEuclidean:
      return std::make_shared<EuclideanDistance>();
    case MetricKind::kManhattan:
      return std::make_shared<ManhattanDistance>();
    case MetricKind::kMahalanobis:
      return std::make_shared<MahalanobisDistance>(MahalanobisDistance::FitTo(data));
  }
  std::unreachable();
}

}
