#include "clustering/evaluation/silhouette_coefficient.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "clustering/algorithms/clusterer.hpp"
#include "clustering/evaluation/cluster_indexing.hpp"

namespace clustering {

std::optional<double> SilhouetteCoefficient(const Matrix& data, std::span<const int> labels,
                                            const DistanceMetric& metric) {
  const ClusterIndexing indexing = IndexClusters(labels);
  const std::size_t cluster_count = indexing.cluster_sizes.size();
  if (cluster_count < 2 || cluster_count >= indexing.clustered_object_count) {
    return std::nullopt;
  }

  double silhouette_sum = 0.0;
  std::vector<double> distance_sums(cluster_count);
  for (std::size_t object = 0; object < data.RowCount(); ++object) {
    if (indexing.dense_labels[object] == kNoiseLabel) {
      continue;
    }
    std::ranges::fill(distance_sums, 0.0);
    for (std::size_t other = 0; other < data.RowCount(); ++other) {
      if (other == object || indexing.dense_labels[other] == kNoiseLabel) {
        continue;
      }
      distance_sums[static_cast<std::size_t>(indexing.dense_labels[other])] +=
          metric.Distance(data.Row(object), data.Row(other));
    }

    const auto own_cluster = static_cast<std::size_t>(indexing.dense_labels[object]);
    if (indexing.cluster_sizes[own_cluster] < 2) {
      continue;
    }
    const double mean_intra_distance =
        distance_sums[own_cluster] / static_cast<double>(indexing.cluster_sizes[own_cluster] - 1);
    double mean_nearest_other_distance = std::numeric_limits<double>::infinity();
    for (std::size_t cluster = 0; cluster < cluster_count; ++cluster) {
      if (cluster != own_cluster) {
        mean_nearest_other_distance = std::min(
            mean_nearest_other_distance,
            distance_sums[cluster] / static_cast<double>(indexing.cluster_sizes[cluster]));
      }
    }
    const double scale = std::max(mean_intra_distance, mean_nearest_other_distance);
    if (scale > 0.0) {
      silhouette_sum += (mean_nearest_other_distance - mean_intra_distance) / scale;
    }
  }
  return silhouette_sum / static_cast<double>(indexing.clustered_object_count);
}

}
