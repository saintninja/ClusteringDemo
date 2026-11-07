#include "clustering/app/report_formatter.hpp"

#include <format>
#include <optional>

namespace clustering {

namespace {

std::string FormatMetric(const std::optional<double>& value) {
  return value ? std::format("{:.4f}", *value) : "n/a";
}

std::string FormatQuality(const std::string& title, const QualityReport& quality) {
  return std::format(
      "{}\n"
      "  ARI:             {}\n"
      "  AMI:             {}\n"
      "  Silhouette:      {}\n"
      "  Davies-Bouldin:  {}\n",
      title, FormatMetric(quality.adjusted_rand_index),
      FormatMetric(quality.adjusted_mutual_information),
      FormatMetric(quality.silhouette_coefficient), FormatMetric(quality.davies_bouldin_index));
}

std::string FormatObjectCounts(const PipelineOutcome& outcome) {
  if (outcome.training_object_count == 0) {
    return std::format("{} (rows skipped because of missing values: {})", outcome.object_count,
                       outcome.skipped_row_count);
  }
  return std::format("{} (training: {}, clustered: {}; rows skipped because of missing values: {})",
                     outcome.object_count, outcome.training_object_count,
                     outcome.object_count - outcome.training_object_count,
                     outcome.skipped_row_count);
}

}

std::string FormatOutcome(const PipelineOutcome& outcome) {
  const ClusteringResult& clustering = outcome.clustering;
  std::string report = std::format(
      "Algorithm:       {}\n"
      "Objects:         {}\n"
      "Clusters found:  {}, noise objects: {}\n"
      "Iterations:      {} ({})\n"
      "Time:            {:.2f} ms\n",
      outcome.algorithm_description, FormatObjectCounts(outcome), clustering.ClusterCount(),
      clustering.NoiseCount(), clustering.iterations,
      clustering.converged ? "converged" : "iteration limit reached", outcome.clustering_milliseconds);

  if (outcome.quality_of_clustered_objects) {
    report += FormatQuality("Quality (all objects, training included):", outcome.quality);
    report += FormatQuality("Quality (clustered objects only):", *outcome.quality_of_clustered_objects);
  } else {
    report += FormatQuality("Quality:", outcome.quality);
  }
  return report;
}

}
