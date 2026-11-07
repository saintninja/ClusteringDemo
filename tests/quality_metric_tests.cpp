#include <vector>

#include "clustering/algorithms/clusterer.hpp"
#include "clustering/distance/euclidean_distance.hpp"
#include "clustering/evaluation/adjusted_mutual_information.hpp"
#include "clustering/evaluation/adjusted_rand_index.hpp"
#include "clustering/evaluation/davies_bouldin_index.hpp"
#include "clustering/evaluation/quality_evaluator.hpp"
#include "clustering/evaluation/silhouette_coefficient.hpp"
#include "test_support.hpp"

namespace clustering::tests {

namespace {

const std::vector<int> kTruth{0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
const std::vector<int> kPrediction{0, 0, 1, 1, 1, 2, 2, 2, 2, 2};
const std::vector<int> kGoodClusters{0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
const std::vector<int> kBlurredClusters{0, 0, 1, 1, 1, 1, 2, 2, 2, 2};

Matrix ThreeSeparatedGroups() {
  Matrix data;
  for (const auto& row : {std::vector<double>{0, 0}, {1, 0}, {0, 1}, {5, 5}, {6, 5}, {5, 6},
                          {10, 0}, {11, 0}, {10, 1}, {11, 1}}) {
    data.AppendRow(row);
  }
  return data;
}

void AdjustedRandIndexMatchesReference() {
  ExpectNear(AdjustedRandIndex(kTruth, kPrediction), 0.460431654676, 1e-9, "ari");
}

void AdjustedMutualInformationMatchesReference() {
  ExpectNear(AdjustedMutualInformation(kTruth, kPrediction), 0.466656778282, 1e-9, "ami");
}

void ExternalMetricsIgnoreLabelPermutation() {
  const std::vector<int> renamed{2, 2, 2, 0, 0, 0, 1, 1, 1, 1};
  ExpectNear(AdjustedRandIndex(kTruth, renamed), 1.0, 1e-12, "ari of renamed labels");
  ExpectNear(AdjustedMutualInformation(kTruth, renamed), 1.0, 1e-12, "ami of renamed labels");
}

void SilhouetteMatchesReference() {
  const EuclideanDistance metric;
  const Matrix data = ThreeSeparatedGroups();
  ExpectNear(*SilhouetteCoefficient(data, kGoodClusters, metric), 0.836936918638, 1e-9,
             "silhouette of good clusters");
  ExpectNear(*SilhouetteCoefficient(data, kBlurredClusters, metric), 0.591165298222, 1e-9,
             "silhouette of blurred clusters");
}

void DaviesBouldinMatchesReference() {
  const EuclideanDistance metric;
  const Matrix data = ThreeSeparatedGroups();
  ExpectNear(*DaviesBouldinIndex(data, kGoodClusters, metric), 0.189922177044, 1e-9,
             "dbi of good clusters");
  ExpectNear(*DaviesBouldinIndex(data, kBlurredClusters, metric), 0.528702834921, 1e-9,
             "dbi of blurred clusters");
}

void InternalMetricsAreUndefinedForSingleCluster() {
  const EuclideanDistance metric;
  const Matrix data = ThreeSeparatedGroups();
  const std::vector<int> single_cluster(10, 0);
  Expect(!SilhouetteCoefficient(data, single_cluster, metric).has_value(), "single cluster silhouette");
  Expect(!DaviesBouldinIndex(data, single_cluster, metric).has_value(), "single cluster dbi");
}

void InternalMetricsSkipNoise() {
  const EuclideanDistance metric;
  Matrix data = ThreeSeparatedGroups();
  const std::vector<double> outlier{100.0, 100.0};
  data.AppendRow(outlier);
  std::vector<int> labels = kGoodClusters;
  labels.push_back(kNoiseLabel);
  ExpectNear(*SilhouetteCoefficient(data, labels, metric), 0.836936918638, 1e-9,
             "silhouette ignoring noise");
  ExpectNear(*DaviesBouldinIndex(data, labels, metric), 0.189922177044, 1e-9, "dbi ignoring noise");
}

void EvaluatorSkipsExternalMetricsWithoutTruth() {
  const QualityEvaluator evaluator(std::make_shared<EuclideanDistance>());
  const QualityReport report = evaluator.Evaluate(ThreeSeparatedGroups(), kGoodClusters);
  Expect(!report.adjusted_rand_index.has_value(), "ari needs ground truth");
  Expect(!report.adjusted_mutual_information.has_value(), "ami needs ground truth");
  Expect(report.silhouette_coefficient.has_value(), "silhouette does not need ground truth");
  Expect(report.davies_bouldin_index.has_value(), "dbi does not need ground truth");
}

}

void AddQualityMetricTests(TestSuite& suite) {
  suite.Add("adjusted rand index matches sklearn", AdjustedRandIndexMatchesReference);
  suite.Add("adjusted mutual information matches sklearn", AdjustedMutualInformationMatchesReference);
  suite.Add("external metrics ignore label permutation", ExternalMetricsIgnoreLabelPermutation);
  suite.Add("silhouette matches sklearn", SilhouetteMatchesReference);
  suite.Add("davies-bouldin matches sklearn", DaviesBouldinMatchesReference);
  suite.Add("internal metrics are undefined for a single cluster",
            InternalMetricsAreUndefinedForSingleCluster);
  suite.Add("internal metrics skip noise", InternalMetricsSkipNoise);
  suite.Add("evaluator skips external metrics without ground truth",
            EvaluatorSkipsExternalMetricsWithoutTruth);
}

}
