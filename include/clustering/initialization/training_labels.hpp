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
#include <map>
#include <vector>

#include "clustering/core/fixed_assignments.hpp"

namespace clustering {

class TrainingLabels {
 public:
  explicit TrainingLabels(std::vector<int> class_of_object);

  [[nodiscard]] std::size_t ObjectCount() const noexcept { return class_of_object_.size(); }
  [[nodiscard]] std::size_t ClassCount() const noexcept { return cluster_of_class_.size(); }
  [[nodiscard]] std::size_t ClusterOfObject(std::size_t training_object) const;
  [[nodiscard]] FixedAssignments ToFixedAssignments(std::size_t total_object_count) const;

 private:
  std::vector<int> class_of_object_;
  std::map<int, std::size_t> cluster_of_class_;
};

}
