#include "clustering/initialization/seeded_initializer.hpp"

#include <algorithm>
#include <format>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "clustering/core/linear_algebra.hpp"
#include "clustering/core/validation.hpp"

namespace clustering {

SeededInitializer::SeededInitializer(TrainingLabels training, std::uint64_t seed)
    : training_(std::move(training)), random_engine_(seed) {}

Matrix SeededInitializer::Initialize(const Matrix& data, std::size_t cluster_count,
                                     const DistanceMetric&) {
  RequireClusterCount(data, cluster_count);
  if (data.RowCount() < training_.ObjectCount()) {
    throw std::invalid_argument("dataset must start with the training objects");
  }
  if (training_.ClassCount() > cluster_count) {
    throw std::invalid_argument("training set has more classes than requested clusters");
  }

  std::vector<std::vector<std::size_t>> members_of_class(training_.ClassCount());
  for (std::size_t object = 0; object < training_.ObjectCount(); ++object) {
    members_of_class[training_.ClusterOfObject(object)].push_back(object);
  }

  Matrix centers;
  for (const auto& members : members_of_class) {
    centers.AppendRow(MeanOfRows(data, members));
  }

  const std::size_t missing_center_count = cluster_count - training_.ClassCount();
  std::vector<std::size_t> candidates(data.RowCount() - training_.ObjectCount());
  std::iota(candidates.begin(), candidates.end(), training_.ObjectCount());
  if (missing_center_count > candidates.size()) {
    throw std::invalid_argument("not enough unlabelled objects to complete the centers");
  }
  std::vector<std::size_t> extra_objects;
  std::sample(candidates.begin(), candidates.end(), std::back_inserter(extra_objects),
              missing_center_count, random_engine_);
  for (const std::size_t object : extra_objects) {
    centers.AppendRow(data.Row(object));
  }
  return centers;
}

std::string SeededInitializer::Name() const {
  return std::format("seeded (semi-supervised, {} training objects, {} classes)",
                     training_.ObjectCount(), training_.ClassCount());
}

}
