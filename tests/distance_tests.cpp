#include <cmath>
#include <vector>

#include "clustering/distance/euclidean_distance.hpp"
#include "clustering/distance/mahalanobis_distance.hpp"
#include "clustering/distance/manhattan_distance.hpp"
#include "test_support.hpp"

namespace clustering::tests {

namespace {

void EuclideanDistanceOfThreeFourFive() {
  const std::vector<double> origin{0.0, 0.0};
  const std::vector<double> target{3.0, 4.0};
  ExpectNear(EuclideanDistance().Distance(origin, target), 5.0, 1e-12, "euclidean");
}

void ManhattanDistanceOfThreeFour() {
  const std::vector<double> origin{0.0, 0.0};
  const std::vector<double> target{3.0, -4.0};
  ExpectNear(ManhattanDistance().Distance(origin, target), 7.0, 1e-12, "manhattan");
}

void MahalanobisDistanceUsesInverseCovariance() {
  Matrix inverse_covariance(2, 2);
  inverse_covariance(0, 0) = 0.25;
  inverse_covariance(1, 1) = 1.0;
  const MahalanobisDistance metric(inverse_covariance);
  const std::vector<double> origin{0.0, 0.0};
  ExpectNear(metric.Distance(std::vector<double>{2.0, 0.0}, origin), 1.0, 1e-12, "stretched axis");
  ExpectNear(metric.Distance(std::vector<double>{0.0, 1.0}, origin), 1.0, 1e-12, "unit axis");
}

void MahalanobisDistanceFittedToData() {
  Matrix cross;
  for (const auto& row : {std::vector<double>{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
    cross.AppendRow(row);
  }
  const MahalanobisDistance metric = MahalanobisDistance::FitTo(cross);
  const std::vector<double> origin{0.0, 0.0};
  ExpectNear(metric.Distance(std::vector<double>{1.0, 0.0}, origin), std::sqrt(1.5), 1e-6,
             "fitted mahalanobis");
}

void MahalanobisStaysFiniteForCollinearData() {
  Matrix collinear;
  for (const auto& row : {std::vector<double>{1, 1}, {2, 2}, {3, 3}, {4, 4}}) {
    collinear.AppendRow(row);
  }
  const MahalanobisDistance metric = MahalanobisDistance::FitTo(collinear);
  const double along_the_line = metric.Distance(std::vector<double>{1.0, 1.0}, std::vector<double>{2.0, 2.0});
  const double across_the_line = metric.Distance(std::vector<double>{1.0, 2.0}, std::vector<double>{2.0, 1.0});
  Expect(std::isfinite(along_the_line) && along_the_line > 0.0, "distance along the line is finite");
  Expect(across_the_line > 1000.0 * along_the_line, "ridge makes the unseen direction very costly");
}

}

void AddDistanceTests(TestSuite& suite) {
  suite.Add("euclidean distance", EuclideanDistanceOfThreeFourFive);
  suite.Add("manhattan distance", ManhattanDistanceOfThreeFour);
  suite.Add("mahalanobis distance with explicit inverse covariance",
            MahalanobisDistanceUsesInverseCovariance);
  suite.Add("mahalanobis distance fitted to data", MahalanobisDistanceFittedToData);
  suite.Add("mahalanobis stays finite for collinear data", MahalanobisStaysFiniteForCollinearData);
}

}
