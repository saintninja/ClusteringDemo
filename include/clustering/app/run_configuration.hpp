#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "clustering/distance/metric_factory.hpp"
#include "clustering/initialization/initializer_factory.hpp"

namespace clustering {

enum class AlgorithmKind { kKMeans, kGaussianMixture, kDbscan, kAffinityPropagation };

struct RunConfiguration {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::vector<std::string> feature_columns;
  std::optional<std::string> label_column;
  std::optional<std::size_t> sample_size;
  bool standardize_features = false;
  std::uint64_t seed = 42;

  AlgorithmKind algorithm = AlgorithmKind::kKMeans;
  MetricKind metric = MetricKind::kEuclidean;
  MetricKind evaluation_metric = MetricKind::kEuclidean;

  std::size_t cluster_count = 0;
  InitializerKind initializer = InitializerKind::kForgy;
  std::size_t max_iterations = 300;

  std::optional<std::filesystem::path> training_path;
  std::optional<std::string> training_label_column;
  std::optional<double> training_fraction;

  std::optional<double> epsilon;
  std::size_t min_points = 5;

  double damping = 0.7;
  std::optional<double> preference;

  [[nodiscard]] bool UsesTrainingSet() const {
    return training_path.has_value() || training_fraction.has_value();
  }
};

}
