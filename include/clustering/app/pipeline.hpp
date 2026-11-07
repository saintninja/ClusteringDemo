#pragma once

#include <optional>
#include <string>

#include "clustering/algorithms/clusterer.hpp"
#include "clustering/app/run_configuration.hpp"
#include "clustering/core/result.hpp"
#include "clustering/evaluation/quality_evaluator.hpp"

namespace clustering {

struct PipelineOutcome {
  std::string algorithm_description;
  std::size_t object_count = 0;
  std::size_t training_object_count = 0;
  std::size_t skipped_row_count = 0;
  ClusteringResult clustering;
  QualityReport quality;
  std::optional<QualityReport> quality_of_clustered_objects;
  double clustering_milliseconds = 0.0;
};

[[nodiscard]] Result<PipelineOutcome> RunPipeline(const RunConfiguration& configuration);

}
