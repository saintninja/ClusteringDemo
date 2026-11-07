#include "clustering/app/clusterer_factory.hpp"

#include <utility>

#include "clustering/algorithms/affinity_propagation.hpp"
#include "clustering/algorithms/dbscan.hpp"
#include "clustering/algorithms/gaussian_mixture.hpp"
#include "clustering/algorithms/kmeans.hpp"
#include "clustering/initialization/initializer_factory.hpp"

namespace clustering {

namespace {

std::unique_ptr<CentroidInitializer> MakeConfiguredInitializer(
    const RunConfiguration& configuration, const std::optional<TrainingLabels>& training) {
  return MakeInitializer(
      {.kind = configuration.initializer, .seed = configuration.seed, .training = training});
}

FixedAssignments FixedAssignmentsFor(const std::optional<TrainingLabels>& training,
                                     std::size_t object_count) {
  return training ? training->ToFixedAssignments(object_count) : FixedAssignments{};
}

}

std::unique_ptr<Clusterer> MakeClusterer(const RunConfiguration& configuration,
                                         const std::optional<TrainingLabels>& training,
                                         std::size_t object_count, MetricPtr metric) {
  switch (configuration.algorithm) {
    case AlgorithmKind::kKMeans:
      return std::make_unique<KMeans>(
          KMeansOptions{.cluster_count = configuration.cluster_count,
                        .max_iterations = configuration.max_iterations},
          MakeConfiguredInitializer(configuration, training), std::move(metric),
          FixedAssignmentsFor(training, object_count));
    case AlgorithmKind::kGaussianMixture:
      return std::make_unique<GaussianMixture>(
          GaussianMixtureOptions{.component_count = configuration.cluster_count,
                                 .max_iterations = configuration.max_iterations},
          MakeConfiguredInitializer(configuration, training), std::move(metric),
          FixedAssignmentsFor(training, object_count));
    case AlgorithmKind::kDbscan:
      return std::make_unique<Dbscan>(
          DbscanOptions{.epsilon = configuration.epsilon.value_or(0.0),
                        .min_points = configuration.min_points},
          std::move(metric));
    case AlgorithmKind::kAffinityPropagation:
      return std::make_unique<AffinityPropagation>(
          AffinityPropagationOptions{.damping = configuration.damping,
                                     .max_iterations = configuration.max_iterations,
                                     .preference = configuration.preference},
          std::move(metric));
  }
  std::unreachable();
}

}
