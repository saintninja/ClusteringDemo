#include <memory>
#include <vector>

#include "clustering/algorithms/affinity_propagation.hpp"
#include "clustering/algorithms/dbscan.hpp"
#include "clustering/algorithms/gaussian_mixture.hpp"
#include "clustering/algorithms/kmeans.hpp"
#include "clustering/distance/euclidean_distance.hpp"
#include "clustering/distance/manhattan_distance.hpp"
#include "clustering/evaluation/adjusted_rand_index.hpp"
#include "clustering/initialization/forgy_initializer.hpp"
#include "clustering/initialization/kmeans_plus_plus_initializer.hpp"
#include "clustering/initialization/seeded_initializer.hpp"
#include "test_support.hpp"
#include "synthetic_data.hpp"

namespace clustering::tests {

namespace {

MetricPtr Euclidean() { return std::make_shared<EuclideanDistance>(); }

void KMeansRecoversSeparatedBlobs() {
  const LabelledBlobs blobs = MakeThreeBlobs();
  KMeans kmeans({.cluster_count = 3}, std::make_unique<KMeansPlusPlusInitializer>(11), Euclidean());
  const ClusteringResult result = kmeans.Fit(blobs.points);
  Expect(result.converged, "k-means converges on separated blobs");
  ExpectNear(AdjustedRandIndex(blobs.labels, result.labels), 1.0, 1e-12, "k-means ari");
}

void KMeansWorksWithManhattanMetric() {
  const LabelledBlobs blobs = MakeThreeBlobs();
  KMeans kmeans({.cluster_count = 3}, std::make_unique<KMeansPlusPlusInitializer>(5),
                std::make_shared<ManhattanDistance>());
  ExpectNear(AdjustedRandIndex(blobs.labels, kmeans.Fit(blobs.points).labels), 1.0, 1e-12,
             "manhattan k-means ari");
}

void KMeansIsReproducibleForTheSameSeed() {
  const LabelledBlobs blobs = MakeThreeBlobs();
  const auto fit = [&] {
    KMeans kmeans({.cluster_count = 3}, std::make_unique<KMeansPlusPlusInitializer>(99), Euclidean());
    return kmeans.Fit(blobs.points).labels;
  };
  Expect(fit() == fit(), "same seed must give identical labels");
}

void KMeansRejectsTooManyClusters() {
  const LabelledBlobs blobs = MakeThreeBlobs(7, 2);
  KMeans kmeans({.cluster_count = 100}, std::make_unique<KMeansPlusPlusInitializer>(1), Euclidean());
  ExpectThrows([&] { static_cast<void>(kmeans.Fit(blobs.points)); }, "k larger than object count");
}

Matrix TwoGroupsWithMislabelledTrainingObject() {
  Matrix data;
  for (const auto& row : {std::vector<double>{0.0, 0.0}, {1000.0, 0.0}, {1.0, 0.0}, {2.0, 0.0},
                          {0.5, 0.5}, {999.0, 0.0}, {998.0, 0.0}, {1000.5, 0.5}, {1.5, 0.5}}) {
    data.AppendRow(row);
  }
  return data;
}

void KMeansKeepsTrainingLabelsFixed() {
  const Matrix data = TwoGroupsWithMislabelledTrainingObject();
  const TrainingLabels training({0, 1, 1});
  KMeans kmeans({.cluster_count = 2}, std::make_unique<SeededInitializer>(training, 1), Euclidean(),
                training.ToFixedAssignments(data.RowCount()));
  const ClusteringResult result = kmeans.Fit(data);
  Expect(result.converged, "semi-supervised k-means converges");
  Expect(result.labels[0] == 0 && result.labels[1] == 1, "training labels are kept");
  Expect(result.labels[2] == 1, "training object stays in its cluster although it lies next to the other");
  Expect(result.labels[3] == 0 && result.labels[8] == 0, "free object near the first group follows the centers");
  Expect(result.labels[5] == 1 && result.labels[7] == 1, "free objects near the second group");
}

void GaussianMixtureKeepsTrainingLabelsFixed() {
  const LabelledBlobs blobs = MakeThreeBlobs();
  std::vector<int> training_classes{blobs.labels[0], blobs.labels[60], blobs.labels[120]};
  Matrix data;
  for (const std::size_t row : {std::size_t{0}, std::size_t{60}, std::size_t{120}}) {
    data.AppendRow(blobs.points.Row(row));
  }
  for (std::size_t row = 0; row < blobs.points.RowCount(); ++row) {
    data.AppendRow(blobs.points.Row(row));
  }
  const TrainingLabels training(training_classes);
  GaussianMixture mixture({.component_count = 3}, std::make_unique<SeededInitializer>(training, 1),
                          Euclidean(), training.ToFixedAssignments(data.RowCount()));
  const ClusteringResult result = mixture.Fit(data);
  for (std::size_t object = 0; object < training.ObjectCount(); ++object) {
    Expect(result.labels[object] == static_cast<int>(training.ClusterOfObject(object)),
           "gmm keeps training labels");
  }
}

void FixedAssignmentsAreValidated() {
  const Matrix data = TwoGroupsWithMislabelledTrainingObject();
  FixedAssignments wrong_size(3);
  KMeans wrong_size_kmeans({.cluster_count = 2}, std::make_unique<ForgyInitializer>(1), Euclidean(),
                           wrong_size);
  ExpectThrows([&] { static_cast<void>(wrong_size_kmeans.Fit(data)); }, "fixed assignments size");
  FixedAssignments unknown_cluster(data.RowCount());
  unknown_cluster[0] = 5;
  KMeans unknown_cluster_kmeans({.cluster_count = 2}, std::make_unique<ForgyInitializer>(1),
                                Euclidean(), unknown_cluster);
  ExpectThrows([&] { static_cast<void>(unknown_cluster_kmeans.Fit(data)); }, "unknown cluster");
}

void GaussianMixtureRecoversSeparatedBlobs() {
  const LabelledBlobs blobs = MakeThreeBlobs();
  GaussianMixture mixture({.component_count = 3}, std::make_unique<KMeansPlusPlusInitializer>(2),
                          Euclidean());
  const ClusteringResult result = mixture.Fit(blobs.points);
  Expect(result.converged, "em converges on separated blobs");
  ExpectNear(AdjustedRandIndex(blobs.labels, result.labels), 1.0, 1e-12, "gmm ari");
  Expect(result.centers.RowCount() == 3, "gmm exposes component means");
}

void DbscanFindsBlobsAndMarksNoise() {
  LabelledBlobs blobs = MakeThreeBlobs();
  blobs.points.AppendRow(std::vector<double>{60.0, 60.0});
  Dbscan dbscan({.epsilon = 1.2, .min_points = 5}, Euclidean());
  const ClusteringResult result = dbscan.Fit(blobs.points);
  Expect(result.labels.back() == kNoiseLabel, "isolated object must be noise");
  Expect(result.ClusterCount() == 3, "dbscan finds three clusters");
}

void DbscanTreatsSparseDataAsNoise() {
  const LabelledBlobs blobs = MakeThreeBlobs();
  Dbscan dbscan({.epsilon = 0.001, .min_points = 5}, Euclidean());
  const ClusteringResult result = dbscan.Fit(blobs.points);
  Expect(result.ClusterCount() == 0, "no cluster with tiny epsilon");
  Expect(result.NoiseCount() == blobs.points.RowCount(), "every object is noise");
}

void DbscanValidatesParameters() {
  ExpectThrows([] { Dbscan({.epsilon = 0.0, .min_points = 5}, Euclidean()); }, "epsilon > 0");
  ExpectThrows([] { Dbscan({.epsilon = 1.0, .min_points = 0}, Euclidean()); }, "minPts > 0");
}

void AffinityPropagationFindsBlobExemplars() {
  const LabelledBlobs blobs = MakeThreeBlobs(7, 30);
  AffinityPropagation propagation({.preference = -300.0}, Euclidean());
  const ClusteringResult result = propagation.Fit(blobs.points);
  Expect(result.converged, "affinity propagation converges");
  Expect(result.ClusterCount() == 3, "three exemplars expected");
  ExpectNear(AdjustedRandIndex(blobs.labels, result.labels), 1.0, 1e-12, "affinity ari");
  Expect(result.centers.RowCount() == 3, "exemplars are exposed as centers");
}

void AffinityPropagationRejectsOversizedInput() {
  Matrix large(AffinityPropagation::kMaximumObjectCount + 1, 1);
  AffinityPropagation propagation({}, Euclidean());
  ExpectThrows([&] { static_cast<void>(propagation.Fit(large)); }, "too many objects");
}

}

void AddAlgorithmTests(TestSuite& suite) {
  suite.Add("k-means recovers separated blobs", KMeansRecoversSeparatedBlobs);
  suite.Add("k-means works with manhattan metric", KMeansWorksWithManhattanMetric);
  suite.Add("k-means is reproducible for the same seed", KMeansIsReproducibleForTheSameSeed);
  suite.Add("k-means rejects too many clusters", KMeansRejectsTooManyClusters);
  suite.Add("k-means keeps training labels fixed", KMeansKeepsTrainingLabelsFixed);
  suite.Add("gmm keeps training labels fixed", GaussianMixtureKeepsTrainingLabelsFixed);
  suite.Add("fixed assignments are validated", FixedAssignmentsAreValidated);
  suite.Add("gmm recovers separated blobs", GaussianMixtureRecoversSeparatedBlobs);
  suite.Add("dbscan finds blobs and marks noise", DbscanFindsBlobsAndMarksNoise);
  suite.Add("dbscan treats sparse data as noise", DbscanTreatsSparseDataAsNoise);
  suite.Add("dbscan validates parameters", DbscanValidatesParameters);
  suite.Add("affinity propagation finds blob exemplars", AffinityPropagationFindsBlobExemplars);
  suite.Add("affinity propagation rejects oversized input", AffinityPropagationRejectsOversizedInput);
}

}
