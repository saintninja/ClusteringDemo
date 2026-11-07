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
