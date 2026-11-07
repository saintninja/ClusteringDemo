#include "clustering/initialization/forgy_initializer.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#include "clustering/core/linear_algebra.hpp"
#include "clustering/core/validation.hpp"

namespace clustering {

Matrix ForgyInitializer::Initialize(const Matrix& data, std::size_t cluster_count,
                                    const DistanceMetric&) {
  RequireClusterCount(data, cluster_count);
  std::vector<std::size_t> all_objects(data.RowCount());
  std::iota(all_objects.begin(), all_objects.end(), std::size_t{0});
  std::vector<std::size_t> chosen_objects;
  std::sample(all_objects.begin(), all_objects.end(), std::back_inserter(chosen_objects),
              cluster_count, random_engine_);
  return SelectRows(data, chosen_objects);
}

}
