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
#include "clustering/core/linear_algebra.hpp"

#include <algorithm>

namespace clustering {

std::vector<double> ColumnMeans(const Matrix& data) {
  std::vector<double> means(data.ColumnCount(), 0.0);
  for (std::size_t row = 0; row < data.RowCount(); ++row) {
    for (std::size_t column = 0; column < data.ColumnCount(); ++column) {
      means[column] += data(row, column);
    }
  }
  const auto object_count = static_cast<double>(data.RowCount());
  for (double& mean : means) {
    mean /= object_count;
  }
  return means;
}

std::vector<double> MeanOfRows(const Matrix& data, std::span<const std::size_t> row_indices) {
  std::vector<double> means(data.ColumnCount(), 0.0);
  for (const std::size_t row : row_indices) {
    for (std::size_t column = 0; column < data.ColumnCount(); ++column) {
      means[column] += data(row, column);
    }
  }
  for (double& mean : means) {
    mean /= static_cast<double>(row_indices.size());
  }
  return means;
}

Matrix SelectRows(const Matrix& data, std::span<const std::size_t> row_indices) {
  Matrix selected;
  for (const std::size_t row : row_indices) {
    selected.AppendRow(data.Row(row));
  }
  return selected;
}

Matrix SampleCovariance(const Matrix& data) {
  const auto means = ColumnMeans(data);
  const std::size_t dimension = data.ColumnCount();
  Matrix covariance(dimension, dimension);
  for (std::size_t row = 0; row < data.RowCount(); ++row) {
    for (std::size_t first = 0; first < dimension; ++first) {
      const double first_deviation = data(row, first) - means[first];
      for (std::size_t second = 0; second < dimension; ++second) {
        covariance(first, second) += first_deviation * (data(row, second) - means[second]);
      }
    }
  }
  const double degrees_of_freedom = std::max(1.0, static_cast<double>(data.RowCount()) - 1.0);
  for (std::size_t first = 0; first < dimension; ++first) {
    for (std::size_t second = 0; second < dimension; ++second) {
      covariance(first, second) /= degrees_of_freedom;
    }
  }
  return covariance;
}

double Trace(const Matrix& square_matrix) {
  double trace = 0.0;
  for (std::size_t index = 0; index < square_matrix.RowCount(); ++index) {
    trace += square_matrix(index, index);
  }
  return trace;
}

void AddToDiagonal(Matrix& square_matrix, double amount) {
  for (std::size_t index = 0; index < square_matrix.RowCount(); ++index) {
    square_matrix(index, index) += amount;
  }
}

}
