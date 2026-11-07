#include "clustering/distance/euclidean_distance.hpp"

#include <cmath>
#include <functional>
#include <numeric>

namespace clustering {

double EuclideanDistance::Distance(Point first, Point second) const {
  const double squared_sum = std::transform_reduce(
      first.begin(), first.end(), second.begin(), 0.0, std::plus<>{},
      [](double left, double right) { return (left - right) * (left - right); });
  return std::sqrt(squared_sum);
}

}
