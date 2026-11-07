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
