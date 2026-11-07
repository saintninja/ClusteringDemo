#include "clustering/io/dataset_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <map>
#include <numeric>
#include <random>

namespace clustering {

namespace {

bool IsMissingToken(std::string_view cell) {
  return cell.empty() || cell == "NA" || cell == "N/A" || cell == "NaN" || cell == "nan" ||
         cell == "null" || cell == "NULL";
}

std::optional<double> ParseFiniteNumber(const std::string& text) {
  if (text.empty()) {
    return std::nullopt;
  }
  char* parsed_end = nullptr;
  const double value = std::strtod(text.c_str(), &parsed_end);
  if (parsed_end != text.c_str() + text.size() || !std::isfinite(value)) {
    return std::nullopt;
  }
  return value;
}

struct ColumnLookup {
  std::vector<std::size_t> feature_indices;
  std::optional<std::size_t> label_index;
};

Result<ColumnLookup> ResolveColumns(const CsvTable& table, const DatasetRequest& request) {
  if (request.feature_columns.empty()) {
    return Failure("at least one feature column must be selected");
  }
  ColumnLookup lookup;
  for (const std::string& name : request.feature_columns) {
    const auto index = table.ColumnIndex(name);
    if (!index) {
      return Failure(std::format("column '{}' not found in the input file", name));
    }
    lookup.feature_indices.push_back(*index);
  }
  if (request.label_column) {
    lookup.label_index = table.ColumnIndex(*request.label_column);
    if (!lookup.label_index) {
      return Failure(std::format("label column '{}' not found in the input file",
                                 *request.label_column));
    }
  }
  return lookup;
}

std::vector<std::size_t> ChooseSample(std::size_t available, std::optional<std::size_t> sample_size,
                                      std::uint64_t seed) {
  std::vector<std::size_t> positions(available);
  std::iota(positions.begin(), positions.end(), std::size_t{0});
  if (sample_size && *sample_size < available) {
    std::mt19937_64 random_engine(seed);
    std::ranges::shuffle(positions, random_engine);
    positions.resize(*sample_size);
    std::ranges::sort(positions);
  }
  return positions;
}

}

Result<Dataset> LoadDataset(const CsvTable& table, const DatasetRequest& request) {
  const auto lookup = ResolveColumns(table, request);
  if (!lookup) {
    return std::unexpected(lookup.error());
  }

  std::vector<std::vector<double>> valid_features;
  std::vector<std::string> valid_label_texts;
  std::vector<std::size_t> valid_source_rows;
  std::size_t skipped_rows = 0;

  for (std::size_t row = 0; row < table.rows.size(); ++row) {
    const auto& cells = table.rows[row];
    std::vector<double> features;
    bool has_missing_value = false;
    for (const std::size_t column : lookup->feature_indices) {
      if (IsMissingToken(cells[column])) {
        has_missing_value = true;
        break;
      }
      const auto number = ParseFiniteNumber(cells[column]);
      if (!number) {
        return Failure(std::format("column '{}' contains non-numeric value '{}' in data row {}",
                                   table.header[column], cells[column], row + 1));
      }
      features.push_back(*number);
    }
    if (lookup->label_index && IsMissingToken(cells[*lookup->label_index])) {
      has_missing_value = true;
    }
    if (has_missing_value) {
      ++skipped_rows;
      continue;
    }
    valid_features.push_back(std::move(features));
    if (lookup->label_index) {
      valid_label_texts.push_back(cells[*lookup->label_index]);
    }
    valid_source_rows.push_back(row);
  }

  if (valid_features.empty()) {
    return Failure("no usable rows: every row has missing values in the selected columns");
  }

  Dataset dataset;
  dataset.skipped_row_count = skipped_rows;
  for (const std::size_t position :
       ChooseSample(valid_features.size(), request.sample_size, request.seed)) {
    dataset.features.AppendRow(valid_features[position]);
    dataset.source_rows.push_back(valid_source_rows[position]);
    if (lookup->label_index) {
      dataset.label_texts.push_back(valid_label_texts[position]);
    }
  }
  return dataset;
}

DatasetPartition PartitionDataset(const Dataset& dataset, std::span<const std::size_t> selected_rows) {
  std::vector<bool> is_selected(dataset.features.RowCount(), false);
  for (const std::size_t row : selected_rows) {
    is_selected[row] = true;
  }

  DatasetPartition partition;
  partition.remaining.skipped_row_count = dataset.skipped_row_count;
  for (std::size_t row = 0; row < dataset.features.RowCount(); ++row) {
    Dataset& target = is_selected[row] ? partition.selected : partition.remaining;
    target.features.AppendRow(dataset.features.Row(row));
    target.source_rows.push_back(dataset.source_rows[row]);
    if (!dataset.label_texts.empty()) {
      target.label_texts.push_back(dataset.label_texts[row]);
    }
  }
  return partition;
}

}
