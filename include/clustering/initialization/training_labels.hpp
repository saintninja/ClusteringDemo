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
