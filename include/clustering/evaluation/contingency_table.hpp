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

#include <cstddef>
#include <span>
#include <vector>

namespace clustering {

class ContingencyTable {
 public:
  ContingencyTable(std::span<const int> first_labels, std::span<const int> second_labels);

  [[nodiscard]] std::size_t RowCount() const noexcept { return row_sums_.size(); }
  [[nodiscard]] std::size_t ColumnCount() const noexcept { return column_sums_.size(); }
  [[nodiscard]] std::size_t Total() const noexcept { return total_; }
  [[nodiscard]] std::size_t Count(std::size_t row, std::size_t column) const {
    return counts_[row * ColumnCount() + column];
  }
  [[nodiscard]] std::size_t RowSum(std::size_t row) const { return row_sums_[row]; }
  [[nodiscard]] std::size_t ColumnSum(std::size_t column) const { return column_sums_[column]; }

 private:
  std::size_t total_;
  std::vector<std::size_t> counts_;
  std::vector<std::size_t> row_sums_;
  std::vector<std::size_t> column_sums_;
};

}
