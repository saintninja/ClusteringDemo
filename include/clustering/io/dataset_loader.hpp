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

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "clustering/core/matrix.hpp"
#include "clustering/core/result.hpp"
#include "clustering/io/csv_table.hpp"

namespace clustering {

struct DatasetRequest {
  std::vector<std::string> feature_columns;
  std::optional<std::string> label_column{};
  std::optional<std::size_t> sample_size{};
  std::uint64_t seed = 0;
};

struct Dataset {
  Matrix features;
  std::vector<std::string> label_texts;
  std::vector<std::size_t> source_rows;
  std::size_t skipped_row_count = 0;
};

struct DatasetPartition {
  Dataset selected;
  Dataset remaining;
};

[[nodiscard]] Result<Dataset> LoadDataset(const CsvTable& table, const DatasetRequest& request);
[[nodiscard]] DatasetPartition PartitionDataset(const Dataset& dataset,
                                                std::span<const std::size_t> selected_rows);

}
