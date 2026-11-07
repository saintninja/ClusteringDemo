#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace clustering {

using Point = std::span<const double>;

class Matrix {
 public:
  Matrix() = default;
  Matrix(std::size_t row_count, std::size_t column_count, double initial_value = 0.0)
      : row_count_(row_count),
        column_count_(column_count),
        values_(row_count * column_count, initial_value) {}

  [[nodiscard]] std::size_t RowCount() const noexcept { return row_count_; }
  [[nodiscard]] std::size_t ColumnCount() const noexcept { return column_count_; }
  [[nodiscard]] bool IsEmpty() const noexcept { return row_count_ == 0; }

  [[nodiscard]] Point Row(std::size_t row) const noexcept {
    return {values_.data() + row * column_count_, column_count_};
  }

  [[nodiscard]] std::span<double> MutableRow(std::size_t row) noexcept {
    return {values_.data() + row * column_count_, column_count_};
  }

  [[nodiscard]] double& operator()(std::size_t row, std::size_t column) noexcept {
    return values_[row * column_count_ + column];
  }

  [[nodiscard]] double operator()(std::size_t row, std::size_t column) const noexcept {
    return values_[row * column_count_ + column];
  }

  void AppendRow(Point row);

 private:
  std::size_t row_count_ = 0;
  std::size_t column_count_ = 0;
  std::vector<double> values_;
};

}
