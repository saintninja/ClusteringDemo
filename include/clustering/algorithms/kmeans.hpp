#pragma once

#include <memory>

#include "clustering/algorithms/clusterer.hpp"
#include "clustering/core/fixed_assignments.hpp"
#include "clustering/distance/distance_metric.hpp"
#include "clustering/initialization/centroid_initializer.hpp"

namespace clustering {

struct KMeansOptions {
  std::size_t cluster_count = 0;
  std::size_t max_iterations = 300;
};

class KMeans final : public Clusterer {
 public:
  KMeans(KMeansOptions options, std::unique_ptr<CentroidInitializer> initializer,
         MetricPtr metric, FixedAssignments fixed_assignments = {});

  [[nodiscard]] ClusteringResult Fit(const Matrix& data) override;
  [[nodiscard]] std::string Describe() const override;

 private:
  static constexpr int kUnassignedLabel = -2;

  [[nodiscard]] std::vector<int> InitialLabels(std::size_t object_count) const;
  [[nodiscard]] bool AssignObjectsToNearestCenters(const Matrix& data, const Matrix& centers,
                                                   std::vector<int>& labels) const;
  static void MoveCentersToClusterMeans(const Matrix& data, const std::vector<int>& labels,
                                        Matrix& centers);

  KMeansOptions options_;
  std::unique_ptr<CentroidInitializer> initializer_;
  MetricPtr metric_;
  FixedAssignments fixed_assignments_;
};

}
