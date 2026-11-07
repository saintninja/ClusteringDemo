#include "clustering/algorithms/nearest_center.hpp"

namespace clustering {

std::size_t NearestCenter(Point object, const Matrix& centers, const DistanceMetric& metric) {
  std::size_t nearest = 0;
  double nearest_distance = metric.Distance(object, centers.Row(0));
  for (std::size_t center = 1; center < centers.RowCount(); ++center) {
    const double distance = metric.Distance(object, centers.Row(center));
    if (distance < nearest_distance) {
      nearest_distance = distance;
      nearest = center;
    }
  }
  return nearest;
}

}
