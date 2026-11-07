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
