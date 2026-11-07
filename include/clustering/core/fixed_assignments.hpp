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
#include <optional>
#include <stdexcept>
#include <vector>

namespace clustering {

using FixedAssignments = std::vector<std::optional<int>>;

[[nodiscard]] inline bool IsFixed(const FixedAssignments& assignments, std::size_t object) {
  return !assignments.empty() && assignments[object].has_value();
}

inline void RequireValidFixedAssignments(const FixedAssignments& assignments,
                                         std::size_t object_count, std::size_t cluster_count) {
  if (assignments.empty()) {
    return;
  }
  if (assignments.size() != object_count) {
    throw std::invalid_argument("fixed assignments must be given for every object");
  }
  for (const auto& assignment : assignments) {
    if (assignment && (*assignment < 0 || static_cast<std::size_t>(*assignment) >= cluster_count)) {
      throw std::invalid_argument("fixed assignment refers to a cluster that does not exist");
    }
  }
}

}
