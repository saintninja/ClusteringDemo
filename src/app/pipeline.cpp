#include "clustering/app/pipeline.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <numeric>

#include "clustering/app/clusterer_factory.hpp"
#include "clustering/app/clustering_problem.hpp"
#include "clustering/core/linear_algebra.hpp"
#include "clustering/io/clustered_csv_writer.hpp"
#include "clustering/preprocessing/standardizer.hpp"

namespace clustering {

namespace {

Result<void> EnsureOutputDiffersFromInputs(const RunConfiguration& configuration) {
  std::error_code ignored;
  if (!std::filesystem::exists(configuration.output_path, ignored)) {
    return {};
  }
  const auto same_as_output = [&](const std::filesystem::path& other) {
    return std::filesystem::equivalent(other, configuration.output_path, ignored);
  };
  if (same_as_output(configuration.input_path) ||
      (configuration.training_path && same_as_output(*configuration.training_path))) {
    return Failure("output file must differ from the input files");
  }
  return {};
}

Result<InputTables> ReadInputTables(const RunConfiguration& configuration) {
  InputTables tables;
  auto input = ReadCsv(configuration.input_path);
  if (!input) {
    return std::unexpected(input.error());
  }
  tables.input = std::move(*input);
  if (configuration.training_path) {
    auto training = ReadCsv(*configuration.training_path);
    if (!training) {
      return std::unexpected(training.error());
    }
    tables.training = std::move(*training);
  }
  return tables;
}

MetricPtr EvaluationMetricFor(const RunConfiguration& configuration, const Matrix& features,
                              const MetricPtr& clustering_metric) {
  return configuration.evaluation_metric == configuration.metric
             ? clustering_metric
             : MakeMetric(configuration.evaluation_metric, features);
}

QualityReport EvaluateClusteredObjectsOnly(const QualityEvaluator& evaluator,
                                           const Matrix& features,
                                           const ClusteringResult& clustering,
                                           std::span<const int> truth_labels,
                                           std::size_t training_object_count) {
  std::vector<std::size_t> clustered_rows(features.RowCount() - training_object_count);
  std::iota(clustered_rows.begin(), clustered_rows.end(), training_object_count);
  const std::span<const int> predicted =
      std::span<const int>(clustering.labels).subspan(training_object_count);
  const std::span<const int> truth =
      truth_labels.empty() ? truth_labels : truth_labels.subspan(training_object_count);
  return evaluator.Evaluate(SelectRows(features, clustered_rows), predicted, truth);
}

Result<PipelineOutcome> Execute(const RunConfiguration& configuration) {
  if (auto distinct = EnsureOutputDiffersFromInputs(configuration); !distinct) {
    return std::unexpected(distinct.error());
  }
  const auto tables = ReadInputTables(configuration);
  if (!tables) {
    return std::unexpected(tables.error());
  }
  const auto problem = BuildProblem(configuration, *tables);
  if (!problem) {
    return std::unexpected(problem.error());
  }

  const Matrix features =
      configuration.standardize_features ? Standardize(problem->features) : problem->features;
  const MetricPtr clustering_metric = MakeMetric(configuration.metric, features);
  const QualityEvaluator evaluator(EvaluationMetricFor(configuration, features, clustering_metric));
  const auto clusterer =
      MakeClusterer(configuration, problem->training, features.RowCount(), clustering_metric);

  PipelineOutcome outcome;
  outcome.algorithm_description = clusterer->Describe();
  outcome.object_count = features.RowCount();
  outcome.training_object_count = problem->TrainingObjectCount();
  outcome.skipped_row_count = problem->skipped_row_count;

  const auto started_at = std::chrono::steady_clock::now();
  outcome.clustering = clusterer->Fit(features);
  const std::chrono::duration<double, std::milli> elapsed =
      std::chrono::steady_clock::now() - started_at;
  outcome.clustering_milliseconds = elapsed.count();

  outcome.quality = evaluator.Evaluate(features, outcome.clustering.labels, problem->truth_labels);
  if (problem->training) {
    outcome.quality_of_clustered_objects =
        EvaluateClusteredObjectsOnly(evaluator, features, outcome.clustering,
                                     problem->truth_labels, outcome.training_object_count);
  }

  if (auto written = WriteClusteredCsv(configuration.output_path, tables->input, problem->sources,
                                       outcome.clustering.labels);
      !written) {
    return std::unexpected(written.error());
  }
  return outcome;
}

}

Result<PipelineOutcome> RunPipeline(const RunConfiguration& configuration) {
  try {
    return Execute(configuration);
  } catch (const std::exception& error) {
    return Failure(error.what());
  }
}

}
