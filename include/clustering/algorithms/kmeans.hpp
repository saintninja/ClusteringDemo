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
#pragma once

#include <memory>

#include "clustering/algorithms/clusterer.hpp"
#include "clustering/core/fixed_assignments.hpp"
#include "clustering/distance/distance_metric.hpp"
#include "clustering/initialization/centroid_initializer.hpp"

namespace clustering {

struct KMeansOptions {
  std::size_t cluster_count = 0;
  std::size_t max_iterations = 300;
};

class KMeans final : public Clusterer {
 public:
  KMeans(KMeansOptions options, std::unique_ptr<CentroidInitializer> initializer,
         MetricPtr metric, FixedAssignments fixed_assignments = {});

  [[nodiscard]] ClusteringResult Fit(const Matrix& data) override;
  [[nodiscard]] std::string Describe() const override;

 private:
  static constexpr int kUnassignedLabel = -2;

  [[nodiscard]] std::vector<int> InitialLabels(std::size_t object_count) const;
  [[nodiscard]] bool AssignObjectsToNearestCenters(const Matrix& data, const Matrix& centers,
                                                   std::vector<int>& labels) const;
  static void MoveCentersToClusterMeans(const Matrix& data, const std::vector<int>& labels,
                                        Matrix& centers);

  KMeansOptions options_;
  std::unique_ptr<CentroidInitializer> initializer_;
  MetricPtr metric_;
  FixedAssignments fixed_assignments_;
};

}
