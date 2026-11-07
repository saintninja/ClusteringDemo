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
#include "clustering/algorithms/clusterer.hpp"

#include <algorithm>
#include <iterator>
#include <set>

namespace clustering {

std::size_t ClusteringResult::ClusterCount() const {
  std::set<int> distinct_clusters;
  std::ranges::copy_if(labels, std::inserter(distinct_clusters, distinct_clusters.end()),
                       [](int label) { return label != kNoiseLabel; });
  return distinct_clusters.size();
}

std::size_t ClusteringResult::NoiseCount() const {
  return static_cast<std::size_t>(std::ranges::count(labels, kNoiseLabel));
}

}
