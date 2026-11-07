#pragma once

#include <span>
#include <vector>

#include "clustering/core/matrix.hpp"

namespace clustering {

[[nodiscard]] std::vector<double> ColumnMeans(const Matrix& data);
[[nodiscard]] std::vector<double> MeanOfRows(const Matrix& data,
                                             std::span<const std::size_t> row_indices);
[[nodiscard]] Matrix SelectRows(const Matrix& data, std::span<const std::size_t> row_indices);
[[nodiscard]] Matrix SampleCovariance(const Matrix& data);
[[nodiscard]] double Trace(const Matrix& square_matrix);
void AddToDiagonal(Matrix& square_matrix, double amount);

}
