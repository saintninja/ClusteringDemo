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
#include "clustering/algorithms/nearest_center.hpp"

namespace clustering {

std::size_t NearestCenter(Point object, const Matrix& centers, const DistanceMetric& metric) {
  std::size_t nearest = 0;
  double nearest_distance = metric.Distance(object, centers.Row(0));
  for (std::size_t center = 1; center < centers.RowCount(); ++center) {
    const double distance = metric.Distance(object, centers.Row(center));
    if (distance < nearest_distance) {
      nearest_distance = distance;
      nearest = center;
    }
  }
  return nearest;
}

}
