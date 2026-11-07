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
#include "clustering/initialization/training_labels.hpp"

#include <stdexcept>

namespace clustering {

TrainingLabels::TrainingLabels(std::vector<int> class_of_object)
    : class_of_object_(std::move(class_of_object)) {
  if (class_of_object_.empty()) {
    throw std::invalid_argument("training set must contain at least one object");
  }
  for (const int training_class : class_of_object_) {
    cluster_of_class_.try_emplace(training_class, 0);
  }
  std::size_t next_cluster = 0;
  for (auto& [training_class, cluster] : cluster_of_class_) {
    cluster = next_cluster++;
  }
}

std::size_t TrainingLabels::ClusterOfObject(std::size_t training_object) const {
  return cluster_of_class_.at(class_of_object_.at(training_object));
}

FixedAssignments TrainingLabels::ToFixedAssignments(std::size_t total_object_count) const {
  if (total_object_count < ObjectCount()) {
    throw std::invalid_argument("dataset is smaller than its training set");
  }
  FixedAssignments assignments(total_object_count);
  for (std::size_t object = 0; object < ObjectCount(); ++object) {
    assignments[object] = static_cast<int>(ClusterOfObject(object));
  }
  return assignments;
}

}
