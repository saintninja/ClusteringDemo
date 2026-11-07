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

#include <filesystem>
#include <span>

#include "clustering/core/result.hpp"
#include "clustering/io/csv_table.hpp"

namespace clustering {

enum class RowOrigin { kTraining, kClustered };

struct SourceRow {
  const CsvTable* table;
  std::size_t row;
  RowOrigin origin;
};

[[nodiscard]] Result<void> WriteClusteredCsv(const std::filesystem::path& path,
                                             const CsvTable& header_table,
                                             std::span<const SourceRow> sources,
                                             std::span<const int> labels);

}
