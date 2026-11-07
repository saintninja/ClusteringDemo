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
#include "clustering/io/clustered_csv_writer.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <string>

namespace clustering {

namespace {

std::string UniqueColumnName(const std::vector<std::string>& existing_names, std::string name) {
  while (std::ranges::find(existing_names, name) != existing_names.end()) {
    name += '_';
  }
  return name;
}

std::string_view OriginText(RowOrigin origin) {
  return origin == RowOrigin::kTraining ? "training" : "clustered";
}

std::vector<std::optional<std::size_t>> ColumnsInHeaderOrder(const CsvTable& source,
                                                             const CsvTable& header_table) {
  std::vector<std::optional<std::size_t>> columns;
  for (const std::string& name : header_table.header) {
    columns.push_back(source.ColumnIndex(name));
  }
  return columns;
}

}

Result<void> WriteClusteredCsv(const std::filesystem::path& path, const CsvTable& header_table,
                               std::span<const SourceRow> sources, std::span<const int> labels) {
  const bool has_training_rows = std::ranges::any_of(
      sources, [](const SourceRow& source) { return source.origin == RowOrigin::kTraining; });

  CsvTable clustered_table;
  clustered_table.header = header_table.header;
  clustered_table.header.push_back(UniqueColumnName(header_table.header, "cluster"));
  if (has_training_rows) {
    clustered_table.header.push_back(UniqueColumnName(header_table.header, "origin"));
  }

  std::map<const CsvTable*, std::vector<std::optional<std::size_t>>> column_maps;
  clustered_table.rows.reserve(sources.size());
  for (std::size_t position = 0; position < sources.size(); ++position) {
    const SourceRow& source = sources[position];
    const auto [map_position, inserted] = column_maps.try_emplace(source.table);
    if (inserted) {
      map_position->second = ColumnsInHeaderOrder(*source.table, header_table);
    }

    std::vector<std::string> row;
    row.reserve(clustered_table.header.size());
    for (const auto& column : map_position->second) {
      row.push_back(column ? source.table->rows[source.row][*column] : std::string());
    }
    row.push_back(std::to_string(labels[position]));
    if (has_training_rows) {
      row.emplace_back(OriginText(source.origin));
    }
    clustered_table.rows.push_back(std::move(row));
  }
  return WriteCsv(path, clustered_table);
}

}
