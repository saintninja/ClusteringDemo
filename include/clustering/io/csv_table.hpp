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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "clustering/core/result.hpp"

namespace clustering {

struct CsvTable {
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;

  [[nodiscard]] std::optional<std::size_t> ColumnIndex(std::string_view name) const;
};

[[nodiscard]] Result<CsvTable> ReadCsv(const std::filesystem::path& path);
[[nodiscard]] Result<void> WriteCsv(const std::filesystem::path& path, const CsvTable& table);

}
