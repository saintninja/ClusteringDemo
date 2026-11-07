// ClusteringDemo - pure C++ library for demo clustering algos
// Copyright (C) 2026 Stanislav Furmavnin (GitHub: saintninja)
// SPDX-License-Identifier: LicenseRef-EIL-1.0
//
// This program is licensed under the Educational Integrity License (EIL) v1.0.
// You may use, modify, and distribute this file strictly for Permitted Uses
// (commercial or personal non-evaluated projects).
//
// ANY USE FOR ACADEMIC WORK OR SUBMISSION FOR CREDIT IS STRICTLY PROHIBITED.
// Any modification or redistribution must retain this license under Copyleft terms.
//
// See the full LICENSE file in the repository root for terms and conditions.
#include "clustering/evaluation/davies_bouldin_index.hpp"

#include <algorithm>
#include <vector>

#include "clustering/algorithms/clusterer.hpp"
#include "clustering/evaluation/cluster_indexing.hpp"

namespace clustering {

namespace {

Matrix ClusterCentroids(const Matrix& data, const ClusterIndexing& indexing) {
  Matrix centroids(indexing.cluster_sizes.size(), data.ColumnCount());
  for (std::size_t object = 0; object < data.RowCount(); ++object) {
    if (indexing.dense_labels[object] == kNoiseLabel) {
      continue;
    }
    const auto cluster = static_cast<std::size_t>(indexing.dense_labels[object]);
    for (std::size_t feature = 0; feature < data.ColumnCount(); ++feature) {
      centroids(cluster, feature) += data(object, feature);
    }
  }
  for (std::size_t cluster = 0; cluster < centroids.RowCount(); ++cluster) {
    for (std::size_t feature = 0; feature < centroids.ColumnCount(); ++feature) {
      centroids(cluster, feature) /= static_cast<double>(indexing.cluster_sizes[cluster]);
    }
  }
  return centroids;
}

std::vector<double> ClusterScatters(const Matrix& data, const ClusterIndexing& indexing,
                                    const Matrix& centroids, const DistanceMetric& metric) {
  std::vector<double> scatters(centroids.RowCount(), 0.0);
  for (std::size_t object = 0; object < data.RowCount(); ++object) {
    if (indexing.dense_labels[object] == kNoiseLabel) {
      continue;
    }
    const auto cluster = static_cast<std::size_t>(indexing.dense_labels[object]);
    scatters[cluster] += metric.Distance(data.Row(object), centroids.Row(cluster));
  }
  for (std::size_t cluster = 0; cluster < scatters.size(); ++cluster) {
    scatters[cluster] /= static_cast<double>(indexing.cluster_sizes[cluster]);
  }
  return scatters;
}

}

std::optional<double> DaviesBouldinIndex(const Matrix& data, std::span<const int> labels,
                                         const DistanceMetric& metric) {
  const ClusterIndexing indexing = IndexClusters(labels);
  const std::size_t cluster_count = indexing.cluster_sizes.size();
  if (cluster_count < 2) {
    return std::nullopt;
  }

  const Matrix centroids = ClusterCentroids(data, indexing);
  const std::vector<double> scatters = ClusterScatters(data, indexing, centroids, metric);

  double worst_ratio_sum = 0.0;
  for (std::size_t cluster = 0; cluster < cluster_count; ++cluster) {
    double worst_ratio = 0.0;
    for (std::size_t other = 0; other < cluster_count; ++other) {
      if (other == cluster) {
        continue;
      }
      const double separation = metric.Distance(centroids.Row(cluster), centroids.Row(other));
      if (separation > 0.0) {
        worst_ratio = std::max(worst_ratio, (scatters[cluster] + scatters[other]) / separation);
      }
    }
    worst_ratio_sum += worst_ratio;
  }
  return worst_ratio_sum / static_cast<double>(cluster_count);
}

}
