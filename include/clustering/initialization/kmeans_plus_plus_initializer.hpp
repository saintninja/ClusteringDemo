#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "clustering/initialization/centroid_initializer.hpp"

namespace clustering {

class KMeansPlusPlusInitializer final : public CentroidInitializer {
 public:
  explicit KMeansPlusPlusInitializer(std::uint64_t seed) : random_engine_(seed) {}

  [[nodiscard]] Matrix Initialize(const Matrix& data, std::size_t cluster_count,
                                  const DistanceMetric& metric) override;
  [[nodiscard]] std::string Name() const override { return "kmeans++"; }

 private:
  [[nodiscard]] std::size_t PickProportionallyToWeights(const std::vector<double>& weights);

  std::mt19937_64 random_engine_;
};

}
