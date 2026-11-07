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

#include <vector>

#include "clustering/algorithms/clusterer.hpp"
#include "clustering/distance/distance_metric.hpp"

namespace clustering {

struct DbscanOptions {
  double epsilon = 0.0;
  std::size_t min_points = 5;
};

class Dbscan final : public Clusterer {
 public:
  Dbscan(DbscanOptions options, MetricPtr metric);

  [[nodiscard]] ClusteringResult Fit(const Matrix& data) override;
  [[nodiscard]] std::string Describe() const override;

 private:
  static constexpr int kUnclassifiedLabel = -2;

  [[nodiscard]] std::vector<std::size_t> NeighborsWithinEpsilon(const Matrix& data,
                                                                std::size_t object) const;
  void GrowCluster(const Matrix& data, std::vector<std::size_t> frontier, int cluster,
                   std::vector<int>& labels) const;

  DbscanOptions options_;
  MetricPtr metric_;
};

}
