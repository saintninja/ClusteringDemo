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

#include <optional>
#include <span>
#include <utility>

#include "clustering/core/matrix.hpp"

namespace clustering {

class CholeskyFactor {
 public:
  [[nodiscard]] static std::optional<CholeskyFactor> Compute(
      const Matrix& symmetric_positive_definite);

  [[nodiscard]] std::size_t Dimension() const noexcept { return lower_.RowCount(); }
  [[nodiscard]] double LogDeterminant() const;
  [[nodiscard]] Matrix Inverse() const;

  void SolveLowerInPlace(std::span<double> values) const;
  void SolveUpperTransposedInPlace(std::span<double> values) const;

 private:
  explicit CholeskyFactor(Matrix lower) : lower_(std::move(lower)) {}

  Matrix lower_;
};

}
