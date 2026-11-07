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
#include "clustering/evaluation/adjusted_rand_index.hpp"

#include <cmath>

#include "clustering/evaluation/contingency_table.hpp"

namespace clustering {

namespace {

double PairCount(std::size_t objects) {
  const auto count = static_cast<double>(objects);
  return 0.5 * count * (count - 1.0);
}

}

double AdjustedRandIndex(std::span<const int> truth_labels, std::span<const int> predicted_labels) {
  const ContingencyTable table(truth_labels, predicted_labels);
  if (table.Total() < 2) {
    return 1.0;
  }

  double paired_in_both = 0.0;
  double paired_in_truth = 0.0;
  double paired_in_prediction = 0.0;
  for (std::size_t row = 0; row < table.RowCount(); ++row) {
    paired_in_truth += PairCount(table.RowSum(row));
    for (std::size_t column = 0; column < table.ColumnCount(); ++column) {
      paired_in_both += PairCount(table.Count(row, column));
    }
  }
  for (std::size_t column = 0; column < table.ColumnCount(); ++column) {
    paired_in_prediction += PairCount(table.ColumnSum(column));
  }

  const double expected_by_chance =
      paired_in_truth * paired_in_prediction / PairCount(table.Total());
  const double maximum_possible = 0.5 * (paired_in_truth + paired_in_prediction);
  const double denominator = maximum_possible - expected_by_chance;
  if (std::abs(denominator) < 1e-15) {
    return 1.0;
  }
  return (paired_in_both - expected_by_chance) / denominator;
}

}
