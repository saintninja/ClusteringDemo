// ClusteringDemo - pure C++ library for demo clustering algos
// Copyright (C) 2026 Stanislav Furmavnin (GitHub: saintninja)
// SPDX-License-Identifier: LicenseRef-EIL-1.0
//
// This program is licensed under the Educational Integrity License (EIL) v1.0.
// You may use, modify, and distribute this file strictly for Permitted Uses
// (commercial or personal non-evaluated projects).
//
// ANY USE FOR ACADEMIC WORK OR SUBMISSION FOR CREDIT IS STRICTLY PROHIBITED.
// Any modification or redistribution must retain this license under Copyleft terms.
//
// See the full LICENSE file in the repository root for terms and conditions.
#include "clustering/app/clustering_problem.hpp"

#include <format>

#include "clustering/initialization/training_split.hpp"
#include "clustering/io/dataset_loader.hpp"
#include "clustering/io/label_encoder.hpp"

namespace clustering {

namespace {

struct TrainingSplit {
  Dataset training;
  const CsvTable* training_table;
  Dataset to_cluster;
};

Result<TrainingSplit> ResolveTrainingSplit(const RunConfiguration& configuration,
                                           const InputTables& tables, Dataset input) {
  if (configuration.training_path) {
    const std::optional<std::string>& label_column =
        configuration.training_label_column ? configuration.training_label_column
                                            : configuration.label_column;
    auto training = LoadDataset(*tables.training, {.feature_columns = configuration.feature_columns,
                                                   .label_column = label_column,
                                                   .seed = configuration.seed});
    if (!training) {
      return std::unexpected(training.error());
    }
    return TrainingSplit{std::move(*training), &*tables.training, std::move(input)};
  }

  const std::vector<int> classes = EncodeLabels(input.label_texts);
  const std::vector<std::size_t> training_rows =
      SelectStratifiedTrainingRows(classes, *configuration.training_fraction, configuration.seed);
  DatasetPartition partition = PartitionDataset(input, training_rows);
  if (partition.remaining.features.IsEmpty()) {
    return Failure("no objects left to cluster after selecting the training set");
  }
  return TrainingSplit{std::move(partition.selected), &tables.input, std::move(partition.remaining)};
}

ClusteringProblem PlainProblem(const InputTables& tables, const Dataset& input) {
  ClusteringProblem problem;
  problem.features = input.features;
  problem.skipped_row_count = input.skipped_row_count;
  if (!input.label_texts.empty()) {
    problem.truth_labels = EncodeLabels(input.label_texts);
  }
  for (const std::size_t row : input.source_rows) {
    problem.sources.push_back({&tables.input, row, RowOrigin::kClustered});
  }
  return problem;
}

ClusteringProblem SemiSupervisedProblem(const InputTables& tables, const TrainingSplit& split,
                                        std::size_t input_skipped_rows) {
  ClusteringProblem problem;
  problem.skipped_row_count = input_skipped_rows + split.training.skipped_row_count;

  std::vector<std::string> label_texts = split.training.label_texts;
  for (std::size_t object = 0; object < split.training.features.RowCount(); ++object) {
    problem.features.AppendRow(split.training.features.Row(object));
    problem.sources.push_back(
        {split.training_table, split.training.source_rows[object], RowOrigin::kTraining});
  }
  for (std::size_t object = 0; object < split.to_cluster.features.RowCount(); ++object) {
    problem.features.AppendRow(split.to_cluster.features.Row(object));
    problem.sources.push_back(
        {&tables.input, split.to_cluster.source_rows[object], RowOrigin::kClustered});
  }

  const bool truth_is_known = !split.to_cluster.label_texts.empty();
  label_texts.insert(label_texts.end(), split.to_cluster.label_texts.begin(),
                     split.to_cluster.label_texts.end());
  const std::vector<int> codes = EncodeLabels(label_texts);
  problem.training.emplace(std::vector<int>(
      codes.begin(), codes.begin() + static_cast<std::ptrdiff_t>(split.training.features.RowCount())));
  if (truth_is_known) {
    problem.truth_labels = codes;
  }
  return problem;
}

}

Result<ClusteringProblem> BuildProblem(const RunConfiguration& configuration,
                                       const InputTables& tables) {
  auto input = LoadDataset(tables.input, {.feature_columns = configuration.feature_columns,
                                          .label_column = configuration.label_column,
                                          .sample_size = configuration.sample_size,
                                          .seed = configuration.seed});
  if (!input) {
    return std::unexpected(input.error());
  }
  if (!configuration.UsesTrainingSet()) {
    return PlainProblem(tables, *input);
  }

  const std::size_t input_skipped_rows = input->skipped_row_count;
  auto split = ResolveTrainingSplit(configuration, tables, std::move(*input));
  if (!split) {
    return std::unexpected(split.error());
  }
  return SemiSupervisedProblem(tables, *split, input_skipped_rows);
}

}
