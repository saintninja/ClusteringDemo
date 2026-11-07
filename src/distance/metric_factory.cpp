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
#include "clustering/distance/metric_factory.hpp"

#include <utility>

#include "clustering/distance/euclidean_distance.hpp"
#include "clustering/distance/mahalanobis_distance.hpp"
#include "clustering/distance/manhattan_distance.hpp"

namespace clustering {

std::optional<MetricKind> ParseMetricKind(std::string_view name) {
  if (name == "euclidean") return MetricKind::kEuclidean;
  if (name == "manhattan") return MetricKind::kManhattan;
  if (name == "mahalanobis") return MetricKind::kMahalanobis;
  return std::nullopt;
}

MetricPtr MakeMetric(MetricKind kind, const Matrix& data) {
  switch (kind) {
    case MetricKind::kEuclidean:
      return std::make_shared<EuclideanDistance>();
    case MetricKind::kManhattan:
      return std::make_shared<ManhattanDistance>();
    case MetricKind::kMahalanobis:
      return std::make_shared<MahalanobisDistance>(MahalanobisDistance::FitTo(data));
  }
  std::unreachable();
}

}
