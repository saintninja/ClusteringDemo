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
#include "clustering/evaluation/cluster_indexing.hpp"

#include <map>

#include "clustering/algorithms/clusterer.hpp"

namespace clustering {

ClusterIndexing IndexClusters(std::span<const int> labels) {
  ClusterIndexing indexing;
  indexing.dense_labels.reserve(labels.size());
  std::map<int, int> dense_index;
  for (const int label : labels) {
    if (label == kNoiseLabel) {
      indexing.dense_labels.push_back(kNoiseLabel);
      continue;
    }
    const auto [position, inserted] =
        dense_index.try_emplace(label, static_cast<int>(dense_index.size()));
    if (inserted) {
      indexing.cluster_sizes.push_back(0);
    }
    indexing.dense_labels.push_back(position->second);
    ++indexing.cluster_sizes[static_cast<std::size_t>(position->second)];
    ++indexing.clustered_object_count;
  }
  return indexing;
}

}
