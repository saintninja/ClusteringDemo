#include <algorithm>
#include <ranges>
#include <vector>

#include "clustering/distance/euclidean_distance.hpp"
#include "clustering/initialization/forgy_initializer.hpp"
#include "clustering/initialization/kmeans_plus_plus_initializer.hpp"
#include "clustering/initialization/seeded_initializer.hpp"
#include "clustering/initialization/training_split.hpp"
#include "test_support.hpp"

namespace clustering::tests {

namespace {

Matrix TwoDistantGroups() {
  Matrix data;
  for (int index = 0; index < 5; ++index) {
    data.AppendRow(std::vector<double>{0.1 * index, 0.0});
  }
  for (int index = 0; index < 5; ++index) {
    data.AppendRow(std::vector<double>{1000.0 + 0.1 * index, 0.0});
  }
  return data;
}

bool ContainsRow(const Matrix& data, Point row) {
  for (std::size_t index = 0; index < data.RowCount(); ++index) {
    if (std::ranges::equal(data.Row(index), row)) {
      return true;
    }
  }
  return false;
}

void ForgyPicksDistinctObjectsFromTheData() {
  const Matrix data = TwoDistantGroups();
  ForgyInitializer initializer(1);
  const Matrix centers = initializer.Initialize(data, 4, EuclideanDistance());
  Expect(centers.RowCount() == 4, "forgy returns k centers");
  for (std::size_t index = 0; index < centers.RowCount(); ++index) {
    Expect(ContainsRow(data, centers.Row(index)), "forgy center must be a data object");
  }
}

void KMeansPlusPlusSpreadsCentersAcrossGroups() {
  const Matrix data = TwoDistantGroups();
  KMeansPlusPlusInitializer initializer(3);
  const Matrix centers = initializer.Initialize(data, 2, EuclideanDistance());
  const bool one_per_group = (centers(0, 0) < 500.0) != (centers(1, 0) < 500.0);
  Expect(one_per_group, "kmeans++ must place one center in each distant group");
}

Matrix TwoGroupsWithTrainingPrefix() {
  Matrix data;
  for (const auto& row : {std::vector<double>{0.0, 0.0}, {0.2, 0.0}, {1000.0, 0.0}, {1000.2, 0.0},
                          {0.1, 0.0}, {0.3, 0.0}, {1000.1, 0.0}, {1000.3, 0.0}}) {
    data.AppendRow(row);
  }
  return data;
}

void SeededInitializerUsesTrainingClassMeans() {
  const Matrix data = TwoGroupsWithTrainingPrefix();
  SeededInitializer initializer(TrainingLabels({7, 7, 3, 3}), 1);
  const Matrix centers = initializer.Initialize(data, 2, EuclideanDistance());
  ExpectNear(centers(0, 0), 1000.1, 1e-9, "class 3 is the first cluster in sorted order");
  ExpectNear(centers(1, 0), 0.1, 1e-9, "class 7 is the second cluster in sorted order");
}

void SeededInitializerCompletesMissingCentersFromUnlabelledObjects() {
  const Matrix data = TwoGroupsWithTrainingPrefix();
  SeededInitializer initializer(TrainingLabels({0, 0}), 1);
  const Matrix centers = initializer.Initialize(data, 3, EuclideanDistance());
  Expect(centers.RowCount() == 3, "seeded initializer completes the center set");
  ExpectNear(centers(0, 0), 0.1, 1e-9, "first center is the training class mean");
}

void SeededInitializerRejectsTooManyClasses() {
  const Matrix data = TwoGroupsWithTrainingPrefix();
  SeededInitializer initializer(TrainingLabels({0, 1}), 1);
  ExpectThrows([&] { static_cast<void>(initializer.Initialize(data, 1, EuclideanDistance())); },
               "more training classes than clusters");
}

void TrainingLabelsMapClassesToClustersInSortedOrder() {
  const TrainingLabels training({5, 2, 5, 9});
  Expect(training.ClassCount() == 3, "three classes");
  Expect(training.ClusterOfObject(1) == 0 && training.ClusterOfObject(0) == 1 &&
             training.ClusterOfObject(3) == 2,
         "clusters follow sorted class order");
  const FixedAssignments fixed = training.ToFixedAssignments(6);
  Expect(fixed.size() == 6 && fixed[0] == 1 && !fixed[4].has_value(), "only training objects are fixed");
}

void StratifiedTrainingSelectionCoversEveryClass() {
  const std::vector<int> classes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2};
  const std::vector<std::size_t> rows = SelectStratifiedTrainingRows(classes, 0.1, 5);
  Expect(std::ranges::is_sorted(rows), "selected rows are sorted");
  for (const int label : {0, 1, 2}) {
    const bool covered = std::ranges::any_of(
        rows, [&](std::size_t row) { return classes[row] == label; });
    Expect(covered, "every class needs at least one training object");
  }
}

}

void AddInitializerTests(TestSuite& suite) {
  suite.Add("forgy picks distinct objects from the data", ForgyPicksDistinctObjectsFromTheData);
  suite.Add("kmeans++ spreads centers across groups", KMeansPlusPlusSpreadsCentersAcrossGroups);
  suite.Add("seeded initializer uses training class means", SeededInitializerUsesTrainingClassMeans);
  suite.Add("seeded initializer completes missing centers",
            SeededInitializerCompletesMissingCentersFromUnlabelledObjects);
  suite.Add("seeded initializer rejects too many classes", SeededInitializerRejectsTooManyClasses);
  suite.Add("training labels map classes to clusters in sorted order",
            TrainingLabelsMapClassesToClustersInSortedOrder);
  suite.Add("stratified training selection covers every class", StratifiedTrainingSelectionCoversEveryClass);
}

}
