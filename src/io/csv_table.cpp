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
#include "clustering/io/csv_table.hpp"

#include <algorithm>
#include <format>
#include <fstream>

namespace clustering {

namespace {

constexpr std::string_view kByteOrderMark = "\xEF\xBB\xBF";

std::string_view Trimmed(std::string_view text) {
  constexpr std::string_view kWhitespace = " \t\r\n";
  const std::size_t first = text.find_first_not_of(kWhitespace);
  if (first == std::string_view::npos) {
    return {};
  }
  const std::size_t last = text.find_last_not_of(kWhitespace);
  return text.substr(first, last - first + 1);
}

std::vector<std::string> SplitFields(std::string_view line) {
  std::vector<std::string> fields;
  std::size_t start = 0;
  while (true) {
    const std::size_t comma = line.find(',', start);
    if (comma == std::string_view::npos) {
      fields.emplace_back(Trimmed(line.substr(start)));
      return fields;
    }
    fields.emplace_back(Trimmed(line.substr(start, comma - start)));
    start = comma + 1;
  }
}

}

std::optional<std::size_t> CsvTable::ColumnIndex(std::string_view name) const {
  const auto position = std::ranges::find(header, name);
  if (position == header.end()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(position - header.begin());
}

Result<CsvTable> ReadCsv(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    return Failure(std::format("cannot open '{}' for reading", path.string()));
  }

  CsvTable table;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line_number == 1 && line.starts_with(kByteOrderMark)) {
      line.erase(0, kByteOrderMark.size());
    }
    if (Trimmed(line).empty()) {
      continue;
    }
    std::vector<std::string> fields = SplitFields(line);
    if (table.header.empty()) {
      table.header = std::move(fields);
      continue;
    }
    if (fields.size() != table.header.size()) {
      return Failure(std::format("line {} has {} fields, expected {}", line_number, fields.size(),
                                 table.header.size()));
    }
    table.rows.push_back(std::move(fields));
  }

  if (table.header.empty()) {
    return Failure(std::format("'{}' is empty", path.string()));
  }
  return table;
}

Result<void> WriteCsv(const std::filesystem::path& path, const CsvTable& table) {
  std::ofstream output(path);
  if (!output) {
    return Failure(std::format("cannot open '{}' for writing", path.string()));
  }
  const auto write_row = [&output](const std::vector<std::string>& row) {
    for (std::size_t column = 0; column < row.size(); ++column) {
      output << (column == 0 ? "" : ",") << row[column];
    }
    output << '\n';
  };
  write_row(table.header);
  for (const auto& row : table.rows) {
    write_row(row);
  }
  if (!output) {
    return Failure(std::format("failed while writing '{}'", path.string()));
  }
  return {};
}

}
