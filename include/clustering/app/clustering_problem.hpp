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
