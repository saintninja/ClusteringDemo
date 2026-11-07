#pragma once

#include <cstdint>
#include <random>

#include "clustering/initialization/centroid_initializer.hpp"
#include "clustering/initialization/training_labels.hpp"

namespace clustering {

class SeededInitializer final : public CentroidInitializer {
 public:
  SeededInitializer(TrainingLabels training, std::uint64_t seed);

  [[nodiscard]] Matrix Initialize(const Matrix& data, std::size_t cluster_count,
                                  const DistanceMetric& metric) override;
  [[nodiscard]] std::string Name() const override;

 private:
  TrainingLabels training_;
  std::mt19937_64 random_engine_;
};

}
