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
