#include "clustering/algorithms/clusterer.hpp"

#include <algorithm>
#include <iterator>
#include <set>

namespace clustering {

std::size_t ClusteringResult::ClusterCount() const {
  std::set<int> distinct_clusters;
  std::ranges::copy_if(labels, std::inserter(distinct_clusters, distinct_clusters.end()),
                       [](int label) { return label != kNoiseLabel; });
  return distinct_clusters.size();
}

std::size_t ClusteringResult::NoiseCount() const {
  return static_cast<std::size_t>(std::ranges::count(labels, kNoiseLabel));
}

}
