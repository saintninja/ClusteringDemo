#include "clustering/evaluation/adjusted_mutual_information.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "clustering/evaluation/contingency_table.hpp"

namespace clustering {

namespace {

std::vector<double> LogFactorials(std::size_t largest_argument) {
  std::vector<double> table(largest_argument + 1, 0.0);
  for (std::size_t value = 1; value <= largest_argument; ++value) {
    table[value] = table[value - 1] + std::log(static_cast<double>(value));
  }
  return table;
}

double Entropy(const ContingencyTable& table, bool of_rows) {
  const auto total = static_cast<double>(table.Total());
  const std::size_t group_count = of_rows ? table.RowCount() : table.ColumnCount();
  double entropy = 0.0;
  for (std::size_t group = 0; group < group_count; ++group) {
    const auto size = static_cast<double>(of_rows ? table.RowSum(group) : table.ColumnSum(group));
    entropy -= (size / total) * std::log(size / total);
  }
  return entropy;
}

double MutualInformation(const ContingencyTable& table) {
  const auto total = static_cast<double>(table.Total());
  double information = 0.0;
  for (std::size_t row = 0; row < table.RowCount(); ++row) {
    for (std::size_t column = 0; column < table.ColumnCount(); ++column) {
      const auto joint = static_cast<double>(table.Count(row, column));
      if (joint == 0.0) {
        continue;
      }
      const auto row_size = static_cast<double>(table.RowSum(row));
      const auto column_size = static_cast<double>(table.ColumnSum(column));
      information += (joint / total) * std::log(total * joint / (row_size * column_size));
    }
  }
  return information;
}

double ExpectedMutualInformation(const ContingencyTable& table) {
  const std::size_t total = table.Total();
  const auto total_real = static_cast<double>(total);
  const std::vector<double> log_factorial = LogFactorials(total);

  double expected = 0.0;
  for (std::size_t row = 0; row < table.RowCount(); ++row) {
    const std::size_t row_size = table.RowSum(row);
    for (std::size_t column = 0; column < table.ColumnCount(); ++column) {
      const std::size_t column_size = table.ColumnSum(column);
      const std::size_t lowest_overlap =
          std::max<std::size_t>(1, row_size + column_size > total ? row_size + column_size - total : 0);
      const std::size_t highest_overlap = std::min(row_size, column_size);
      for (std::size_t overlap = lowest_overlap; overlap <= highest_overlap; ++overlap) {
        const double log_probability =
            log_factorial[row_size] + log_factorial[column_size] +
            log_factorial[total - row_size] + log_factorial[total - column_size] -
            log_factorial[total] - log_factorial[overlap] - log_factorial[row_size - overlap] -
            log_factorial[column_size - overlap] -
            log_factorial[total + overlap - row_size - column_size];
        const auto overlap_real = static_cast<double>(overlap);
        const double information_term =
            std::log(total_real * overlap_real /
                     (static_cast<double>(row_size) * static_cast<double>(column_size)));
        expected += (overlap_real / total_real) * information_term * std::exp(log_probability);
      }
    }
  }
  return expected;
}

}

double AdjustedMutualInformation(std::span<const int> truth_labels,
                                 std::span<const int> predicted_labels) {
  const ContingencyTable table(truth_labels, predicted_labels);
  if (table.RowCount() == 1 && table.ColumnCount() == 1) {
    return 1.0;
  }

  const double mutual_information = MutualInformation(table);
  const double expected_information = ExpectedMutualInformation(table);
  const double normalizer = 0.5 * (Entropy(table, true) + Entropy(table, false));

  constexpr double kEpsilon = std::numeric_limits<double>::epsilon();
  double denominator = normalizer - expected_information;
  denominator = denominator < 0.0 ? std::min(denominator, -kEpsilon) : std::max(denominator, kEpsilon);
  return (mutual_information - expected_information) / denominator;
}

}
