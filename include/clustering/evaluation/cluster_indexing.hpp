#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace clustering {

struct ClusterIndexing {
  std::vector<int> dense_labels;
  std::vector<std::size_t> cluster_sizes;
  std::size_t clustered_object_count = 0;
};

[[nodiscard]] ClusterIndexing IndexClusters(std::span<const int> labels);

}
