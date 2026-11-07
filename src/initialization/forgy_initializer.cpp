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
#include "clustering/initialization/forgy_initializer.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#include "clustering/core/linear_algebra.hpp"
#include "clustering/core/validation.hpp"

namespace clustering {

Matrix ForgyInitializer::Initialize(const Matrix& data, std::size_t cluster_count,
                                    const DistanceMetric&) {
  RequireClusterCount(data, cluster_count);
  std::vector<std::size_t> all_objects(data.RowCount());
  std::iota(all_objects.begin(), all_objects.end(), std::size_t{0});
  std::vector<std::size_t> chosen_objects;
  std::sample(all_objects.begin(), all_objects.end(), std::back_inserter(chosen_objects),
              cluster_count, random_engine_);
  return SelectRows(data, chosen_objects);
}

}
