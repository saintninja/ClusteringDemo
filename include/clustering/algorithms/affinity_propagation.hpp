#pragma once

#include <optional>
#include <vector>

#include "clustering/algorithms/clusterer.hpp"
#include "clustering/distance/distance_metric.hpp"

namespace clustering {

struct AffinityPropagationOptions {
  double damping = 0.7;
  std::size_t max_iterations = 200;
  std::size_t convergence_iterations = 15;
  std::optional<double> preference{};
};

class AffinityPropagation final : public Clusterer {
 public:
  static constexpr std::size_t kMaximumObjectCount = 6000;

  AffinityPropagation(AffinityPropagationOptions options, MetricPtr metric);

  [[nodiscard]] ClusteringResult Fit(const Matrix& data) override;
  [[nodiscard]] std::string Describe() const override;

 private:
  [[nodiscard]] Matrix BuildSimilarities(const Matrix& data) const;
  [[nodiscard]] double ResolvePreference(const Matrix& similarities) const;
  void UpdateResponsibilities(const Matrix& similarities, const Matrix& availabilities,
                              Matrix& responsibilities) const;
  void UpdateAvailabilities(const Matrix& responsibilities, Matrix& availabilities) const;
  [[nodiscard]] static std::vector<std::size_t> FindExemplars(const Matrix& responsibilities,
                                                              const Matrix& availabilities);
  [[nodiscard]] static std::vector<int> AssignToExemplars(
      const Matrix& similarities, const std::vector<std::size_t>& exemplars);

  AffinityPropagationOptions options_;
  MetricPtr metric_;
};

}
