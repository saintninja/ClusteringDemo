#include <cmath>

#include "clustering/core/cholesky_factor.hpp"
#include "clustering/core/linear_algebra.hpp"
#include "test_support.hpp"

namespace clustering::tests {

namespace {

Matrix SymmetricExample() {
  Matrix matrix(2, 2);
  matrix(0, 0) = 4.0;
  matrix(0, 1) = 2.0;
  matrix(1, 0) = 2.0;
  matrix(1, 1) = 3.0;
  return matrix;
}

void CholeskyInverseMatchesAnalyticalInverse() {
  const auto factor = CholeskyFactor::Compute(SymmetricExample());
  Expect(factor.has_value(), "positive definite matrix must factorize");
  const Matrix inverse = factor->Inverse();
  ExpectNear(inverse(0, 0), 0.375, 1e-12, "inverse(0,0)");
  ExpectNear(inverse(0, 1), -0.25, 1e-12, "inverse(0,1)");
  ExpectNear(inverse(1, 0), -0.25, 1e-12, "inverse(1,0)");
  ExpectNear(inverse(1, 1), 0.5, 1e-12, "inverse(1,1)");
  ExpectNear(factor->LogDeterminant(), std::log(8.0), 1e-12, "log determinant");
}

void CholeskyRejectsIndefiniteMatrix() {
  Matrix indefinite(2, 2);
  indefinite(0, 0) = 1.0;
  indefinite(0, 1) = 2.0;
  indefinite(1, 0) = 2.0;
  indefinite(1, 1) = 1.0;
  Expect(!CholeskyFactor::Compute(indefinite).has_value(), "indefinite matrix must be rejected");
}

void SampleCovarianceOfSymmetricCross() {
  Matrix cross;
  for (const auto& row : {std::vector<double>{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
    cross.AppendRow(row);
  }
  const Matrix covariance = SampleCovariance(cross);
  ExpectNear(covariance(0, 0), 2.0 / 3.0, 1e-12, "variance of x");
  ExpectNear(covariance(1, 1), 2.0 / 3.0, 1e-12, "variance of y");
  ExpectNear(covariance(0, 1), 0.0, 1e-12, "covariance of x and y");
}

}

void AddLinearAlgebraTests(TestSuite& suite) {
  suite.Add("cholesky inverse matches analytical inverse", CholeskyInverseMatchesAnalyticalInverse);
  suite.Add("cholesky rejects indefinite matrix", CholeskyRejectsIndefiniteMatrix);
  suite.Add("sample covariance of symmetric cross", SampleCovarianceOfSymmetricCross);
}

}
