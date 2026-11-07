#include "clustering/distance/mahalanobis_distance.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "clustering/core/cholesky_factor.hpp"
#include "clustering/core/linear_algebra.hpp"
#include "clustering/core/validation.hpp"

namespace clustering {

namespace {

constexpr double kRelativeRidge = 1e-9;
constexpr double kMinimumRidge = 1e-12;

}

MahalanobisDistance::MahalanobisDistance(Matrix inverse_covariance)
    : inverse_covariance_(std::move(inverse_covariance)) {}

MahalanobisDistance MahalanobisDistance::FitTo(const Matrix& data) {
  RequireNonEmpty(data);
  Matrix covariance = SampleCovariance(data);
  const double average_variance = Trace(covariance) / static_cast<double>(covariance.RowCount());
  AddToDiagonal(covariance, std::max(kRelativeRidge * average_variance, kMinimumRidge));
  const auto factor = CholeskyFactor::Compute(covariance);
  if (!factor) {
    throw std::invalid_argument("covariance matrix of the data is not positive definite");
  }
  return MahalanobisDistance(factor->Inverse());
}

double MahalanobisDistance::Distance(Point first, Point second) const {
  double squared_distance = 0.0;
  for (std::size_t row = 0; row < first.size(); ++row) {
    double weighted_difference = 0.0;
    for (std::size_t column = 0; column < first.size(); ++column) {
      weighted_difference += inverse_covariance_(row, column) * (first[column] - second[column]);
    }
    squared_distance += (first[row] - second[row]) * weighted_difference;
  }
  return std::sqrt(std::max(squared_distance, 0.0));
}

}
