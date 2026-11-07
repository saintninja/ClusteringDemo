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
#include "clustering/algorithms/dbscan.hpp"

#include <format>
#include <vector>
#include <stdexcept>
#include <utility>

#include "clustering/core/validation.hpp"

namespace clustering {

Dbscan::Dbscan(DbscanOptions options, MetricPtr metric)
    : options_(options), metric_(std::move(metric)) {
  if (!(options_.epsilon > 0.0)) {
    throw std::invalid_argument("dbscan epsilon must be positive");
  }
  if (options_.min_points == 0) {
    throw std::invalid_argument("dbscan min_points must be positive");
  }
  if (!metric_) {
    throw std::invalid_argument("dbscan needs a distance metric");
  }
}

ClusteringResult Dbscan::Fit(const Matrix& data) {
  RequireNonEmpty(data);

  ClusteringResult result;
  result.labels.assign(data.RowCount(), kUnclassifiedLabel);
  int next_cluster = 0;

  for (std::size_t object = 0; object < data.RowCount(); ++object) {
    if (result.labels[object] != kUnclassifiedLabel) {
      continue;
    }
    std::vector<std::size_t> neighbors = NeighborsWithinEpsilon(data, object);
    if (neighbors.size() < options_.min_points) {
      result.labels[object] = kNoiseLabel;
      continue;
    }
    result.labels[object] = next_cluster;
    GrowCluster(data, std::move(neighbors), next_cluster, result.labels);
    ++next_cluster;
  }

  result.iterations = 1;
  result.converged = true;
  return result;
}

std::string Dbscan::Describe() const {
  return std::format("dbscan (epsilon={}, minPts={}, metric: {})", options_.epsilon,
                     options_.min_points, metric_->Name());
}

std::vector<std::size_t> Dbscan::NeighborsWithinEpsilon(const Matrix& data,
                                                        std::size_t object) const {
  std::vector<std::size_t> neighbors;
  for (std::size_t other = 0; other < data.RowCount(); ++other) {
    if (metric_->Distance(data.Row(object), data.Row(other)) <= options_.epsilon) {
      neighbors.push_back(other);
    }
  }
  return neighbors;
}

void Dbscan::GrowCluster(const Matrix& data, std::vector<std::size_t> frontier, int cluster,
                         std::vector<int>& labels) const {
  for (std::size_t position = 0; position < frontier.size(); ++position) {
    const std::size_t object = frontier[position];
    if (labels[object] == kNoiseLabel) {
      labels[object] = cluster;
      continue;
    }
    if (labels[object] != kUnclassifiedLabel) {
      continue;
    }
    labels[object] = cluster;
    const std::vector<std::size_t> neighbors = NeighborsWithinEpsilon(data, object);
    if (neighbors.size() >= options_.min_points) {
      for (const std::size_t neighbor : neighbors) {
        frontier.push_back(neighbor);
      }
    }
  }
}

}
