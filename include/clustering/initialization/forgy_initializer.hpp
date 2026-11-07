#pragma once

#include <cstdint>
#include <random>

#include "clustering/initialization/centroid_initializer.hpp"

namespace clustering {

class ForgyInitializer final : public CentroidInitializer {
 public:
  explicit ForgyInitializer(std::uint64_t seed) : random_engine_(seed) {}

  [[nodiscard]] Matrix Initialize(const Matrix& data, std::size_t cluster_count,
                                  const DistanceMetric& metric) override;
  [[nodiscard]] std::string Name() const override { return "forgy"; }

 private:
  std::mt19937_64 random_engine_;
};

}
