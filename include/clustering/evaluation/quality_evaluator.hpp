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

#include <optional>
#include <span>

#include "clustering/core/matrix.hpp"
#include "clustering/distance/distance_metric.hpp"

namespace clustering {

struct QualityReport {
  std::optional<double> adjusted_rand_index;
  std::optional<double> adjusted_mutual_information;
  std::optional<double> silhouette_coefficient;
  std::optional<double> davies_bouldin_index;
};

class QualityEvaluator {
 public:
  explicit QualityEvaluator(MetricPtr metric);

  [[nodiscard]] QualityReport Evaluate(const Matrix& data, std::span<const int> predicted_labels,
                                       std::span<const int> truth_labels = {}) const;

 private:
  MetricPtr metric_;
};

}
