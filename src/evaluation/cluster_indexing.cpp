#include "clustering/evaluation/cluster_indexing.hpp"

#include <map>

#include "clustering/algorithms/clusterer.hpp"

namespace clustering {

ClusterIndexing IndexClusters(std::span<const int> labels) {
  ClusterIndexing indexing;
  indexing.dense_labels.reserve(labels.size());
  std::map<int, int> dense_index;
  for (const int label : labels) {
    if (label == kNoiseLabel) {
      indexing.dense_labels.push_back(kNoiseLabel);
      continue;
    }
    const auto [position, inserted] =
        dense_index.try_emplace(label, static_cast<int>(dense_index.size()));
    if (inserted) {
      indexing.cluster_sizes.push_back(0);
    }
    indexing.dense_labels.push_back(position->second);
    ++indexing.cluster_sizes[static_cast<std::size_t>(position->second)];
    ++indexing.clustered_object_count;
  }
  return indexing;
}

}
