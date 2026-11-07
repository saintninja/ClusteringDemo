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
#include "clustering/initialization/kmeans_plus_plus_initializer.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

#include "clustering/core/linear_algebra.hpp"
#include "clustering/core/validation.hpp"

namespace clustering {

Matrix KMeansPlusPlusInitializer::Initialize(const Matrix& data, std::size_t cluster_count,
                                             const DistanceMetric& metric) {
  RequireClusterCount(data, cluster_count);
  std::vector<std::size_t> chosen_objects;
  chosen_objects.reserve(cluster_count);

  std::uniform_int_distribution<std::size_t> uniform_object(0, data.RowCount() - 1);
  chosen_objects.push_back(uniform_object(random_engine_));

  std::vector<double> squared_distance_to_nearest_center(
      data.RowCount(), std::numeric_limits<double>::infinity());
  while (chosen_objects.size() < cluster_count) {
    const Point newest_center = data.Row(chosen_objects.back());
    for (std::size_t object = 0; object < data.RowCount(); ++object) {
      const double distance = metric.Distance(data.Row(object), newest_center);
      squared_distance_to_nearest_center[object] =
          std::min(squared_distance_to_nearest_center[object], distance * distance);
    }
    chosen_objects.push_back(PickProportionallyToWeights(squared_distance_to_nearest_center));
  }
  return SelectRows(data, chosen_objects);
}

std::size_t KMeansPlusPlusInitializer::PickProportionallyToWeights(
    const std::vector<double>& weights) {
  const double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (total_weight <= 0.0) {
    std::uniform_int_distribution<std::size_t> uniform_object(0, weights.size() - 1);
    return uniform_object(random_engine_);
  }
  std::discrete_distribution<std::size_t> weighted_object(weights.begin(), weights.end());
  return weighted_object(random_engine_);
}

}
