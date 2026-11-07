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
#include "clustering/preprocessing/standardizer.hpp"

#include <cmath>
#include <vector>

#include "clustering/core/linear_algebra.hpp"

namespace clustering {

Matrix Standardize(const Matrix& data) {
  const std::vector<double> means = ColumnMeans(data);
  std::vector<double> standard_deviations(data.ColumnCount(), 0.0);
  for (std::size_t row = 0; row < data.RowCount(); ++row) {
    for (std::size_t column = 0; column < data.ColumnCount(); ++column) {
      const double deviation = data(row, column) - means[column];
      standard_deviations[column] += deviation * deviation;
    }
  }
  for (double& value : standard_deviations) {
    value = std::sqrt(value / static_cast<double>(data.RowCount()));
    if (value == 0.0) {
      value = 1.0;
    }
  }

  Matrix standardized(data.RowCount(), data.ColumnCount());
  for (std::size_t row = 0; row < data.RowCount(); ++row) {
    for (std::size_t column = 0; column < data.ColumnCount(); ++column) {
      standardized(row, column) = (data(row, column) - means[column]) / standard_deviations[column];
    }
  }
  return standardized;
}

}
