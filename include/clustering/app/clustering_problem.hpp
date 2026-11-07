#pragma once

#include <optional>
#include <vector>

#include "clustering/app/run_configuration.hpp"
#include "clustering/core/matrix.hpp"
#include "clustering/core/result.hpp"
#include "clustering/initialization/training_labels.hpp"
#include "clustering/io/clustered_csv_writer.hpp"
#include "clustering/io/csv_table.hpp"

namespace clustering {

struct InputTables {
  CsvTable input;
  std::optional<CsvTable> training;
};

struct ClusteringProblem {
  Matrix features;
  std::vector<int> truth_labels;
  std::optional<TrainingLabels> training;
  std::vector<SourceRow> sources;
  std::size_t skipped_row_count = 0;

  [[nodiscard]] std::size_t TrainingObjectCount() const {
    return training ? training->ObjectCount() : 0;
  }
};

[[nodiscard]] Result<ClusteringProblem> BuildProblem(const RunConfiguration& configuration,
                                                     const InputTables& tables);

}
