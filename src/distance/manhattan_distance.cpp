#include "clustering/distance/manhattan_distance.hpp"

#include <cmath>
#include <functional>
#include <numeric>

namespace clustering {

double ManhattanDistance::Distance(Point first, Point second) const {
  return std::transform_reduce(
      first.begin(), first.end(), second.begin(), 0.0, std::plus<>{},
      [](double left, double right) { return std::abs(left - right); });
}

}
