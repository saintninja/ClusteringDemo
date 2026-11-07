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
#include "clustering/evaluation/quality_evaluator.hpp"

#include <stdexcept>
#include <utility>

#include "clustering/evaluation/adjusted_mutual_information.hpp"
#include "clustering/evaluation/adjusted_rand_index.hpp"
#include "clustering/evaluation/davies_bouldin_index.hpp"
#include "clustering/evaluation/silhouette_coefficient.hpp"

namespace clustering {

QualityEvaluator::QualityEvaluator(MetricPtr metric) : metric_(std::move(metric)) {
  if (!metric_) {
    throw std::invalid_argument("quality evaluator needs a distance metric");
  }
}

QualityReport QualityEvaluator::Evaluate(const Matrix& data, std::span<const int> predicted_labels,
                                         std::span<const int> truth_labels) const {
  if (predicted_labels.size() != data.RowCount()) {
    throw std::invalid_argument("predicted labels must be given for every object");
  }

  QualityReport report;
  if (!truth_labels.empty()) {
    report.adjusted_rand_index = AdjustedRandIndex(truth_labels, predicted_labels);
    report.adjusted_mutual_information = AdjustedMutualInformation(truth_labels, predicted_labels);
  }
  report.silhouette_coefficient = SilhouetteCoefficient(data, predicted_labels, *metric_);
  report.davies_bouldin_index = DaviesBouldinIndex(data, predicted_labels, *metric_);
  return report;
}

}
