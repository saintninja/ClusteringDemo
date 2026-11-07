#include "clustering/evaluation/contingency_table.hpp"

#include <map>
#include <stdexcept>

namespace clustering {

namespace {

std::map<int, std::size_t> DenseIndexOfLabels(std::span<const int> labels) {
  std::map<int, std::size_t> dense_index;
  for (const int label : labels) {
    dense_index.try_emplace(label, dense_index.size());
  }
  return dense_index;
}

}

ContingencyTable::ContingencyTable(std::span<const int> first_labels,
                                   std::span<const int> second_labels)
    : total_(first_labels.size()) {
  if (first_labels.size() != second_labels.size()) {
    throw std::invalid_argument("label sequences must have the same length");
  }
  if (first_labels.empty()) {
    throw std::invalid_argument("label sequences must not be empty");
  }

  const auto first_index = DenseIndexOfLabels(first_labels);
  const auto second_index = DenseIndexOfLabels(second_labels);
  row_sums_.assign(first_index.size(), 0);
  column_sums_.assign(second_index.size(), 0);
  counts_.assign(first_index.size() * second_index.size(), 0);

  for (std::size_t object = 0; object < total_; ++object) {
    const std::size_t row = first_index.at(first_labels[object]);
    const std::size_t column = second_index.at(second_labels[object]);
    ++counts_[row * second_index.size() + column];
    ++row_sums_[row];
    ++column_sums_[column];
  }
}

}
