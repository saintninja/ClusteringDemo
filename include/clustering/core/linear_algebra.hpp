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
