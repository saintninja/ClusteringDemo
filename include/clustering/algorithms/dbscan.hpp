#pragma once

#include <vector>

#include "clustering/algorithms/clusterer.hpp"
#include "clustering/distance/distance_metric.hpp"

namespace clustering {

struct DbscanOptions {
  double epsilon = 0.0;
  std::size_t min_points = 5;
};

class Dbscan final : public Clusterer {
 public:
  Dbscan(DbscanOptions options, MetricPtr metric);

  [[nodiscard]] ClusteringResult Fit(const Matrix& data) override;
  [[nodiscard]] std::string Describe() const override;

 private:
  static constexpr int kUnclassifiedLabel = -2;

  [[nodiscard]] std::vector<std::size_t> NeighborsWithinEpsilon(const Matrix& data,
                                                                std::size_t object) const;
  void GrowCluster(const Matrix& data, std::vector<std::size_t> frontier, int cluster,
                   std::vector<int>& labels) const;

  DbscanOptions options_;
  MetricPtr metric_;
};

}
