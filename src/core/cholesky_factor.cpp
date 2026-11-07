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
#include "clustering/core/cholesky_factor.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace clustering {

std::optional<CholeskyFactor> CholeskyFactor::Compute(const Matrix& symmetric_positive_definite) {
  const std::size_t dimension = symmetric_positive_definite.RowCount();
  Matrix lower(dimension, dimension);
  for (std::size_t row = 0; row < dimension; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      double remainder = symmetric_positive_definite(row, column);
      for (std::size_t inner = 0; inner < column; ++inner) {
        remainder -= lower(row, inner) * lower(column, inner);
      }
      if (row == column) {
        if (!(remainder > 0.0)) {
          return std::nullopt;
        }
        lower(row, column) = std::sqrt(remainder);
      } else {
        lower(row, column) = remainder / lower(column, column);
      }
    }
  }
  return CholeskyFactor(std::move(lower));
}

double CholeskyFactor::LogDeterminant() const {
  double sum_of_logs = 0.0;
  for (std::size_t index = 0; index < Dimension(); ++index) {
    sum_of_logs += std::log(lower_(index, index));
  }
  return 2.0 * sum_of_logs;
}

void CholeskyFactor::SolveLowerInPlace(std::span<double> values) const {
  for (std::size_t row = 0; row < Dimension(); ++row) {
    for (std::size_t column = 0; column < row; ++column) {
      values[row] -= lower_(row, column) * values[column];
    }
    values[row] /= lower_(row, row);
  }
}

void CholeskyFactor::SolveUpperTransposedInPlace(std::span<double> values) const {
  for (std::size_t row = Dimension(); row-- > 0;) {
    for (std::size_t column = row + 1; column < Dimension(); ++column) {
      values[row] -= lower_(column, row) * values[column];
    }
    values[row] /= lower_(row, row);
  }
}

Matrix CholeskyFactor::Inverse() const {
  Matrix inverse(Dimension(), Dimension());
  std::vector<double> column_buffer(Dimension());
  for (std::size_t column = 0; column < Dimension(); ++column) {
    std::ranges::fill(column_buffer, 0.0);
    column_buffer[column] = 1.0;
    SolveLowerInPlace(column_buffer);
    SolveUpperTransposedInPlace(column_buffer);
    for (std::size_t row = 0; row < Dimension(); ++row) {
      inverse(row, column) = column_buffer[row];
    }
  }
  return inverse;
}

}
