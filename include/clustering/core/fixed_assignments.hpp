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
