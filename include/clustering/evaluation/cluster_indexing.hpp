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

#include <cstddef>
#include <span>
#include <vector>

namespace clustering {

struct ClusterIndexing {
  std::vector<int> dense_labels;
  std::vector<std::size_t> cluster_sizes;
  std::size_t clustered_object_count = 0;
};

[[nodiscard]] ClusterIndexing IndexClusters(std::span<const int> labels);

}
