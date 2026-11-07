#include <filesystem>
#include <fstream>

#include "clustering/app/pipeline.hpp"
#include "clustering/io/csv_table.hpp"
#include "test_support.hpp"

namespace clustering::tests {

namespace {

const std::filesystem::path kDataDirectory = CLUSTERING_DATA_DIR;

RunConfiguration ConfigurationFor(const std::string& data_file, const std::string& output_name) {
  RunConfiguration configuration;
  configuration.input_path = kDataDirectory / data_file;
  configuration.output_path = std::filesystem::temp_directory_path() / output_name;
  configuration.feature_columns = {"x", "y"};
  configuration.label_column = "color";
  return configuration;
}

void KMeansPlusPlusClustersBlobFile() {
  RunConfiguration configuration = ConfigurationFor("blob.csv", "clustering_blob_kmeans.csv");
  configuration.cluster_count = 4;
  configuration.initializer = InitializerKind::kKMeansPlusPlus;
  const auto outcome = RunPipeline(configuration);
  Expect(outcome.has_value(), "pipeline must succeed");
  Expect(*outcome->quality.adjusted_rand_index > 0.8, "blob ari");
  Expect(outcome->quality.silhouette_coefficient.has_value(), "silhouette is reported");
  Expect(std::filesystem::exists(configuration.output_path), "output file is written");
}

void SemiSupervisedGaussianMixtureRecoversBoxes() {
  RunConfiguration configuration = ConfigurationFor("boxes.csv", "clustering_boxes_gmm.csv");
  configuration.algorithm = AlgorithmKind::kGaussianMixture;
  configuration.cluster_count = 30;
  configuration.initializer = InitializerKind::kSeeded;
  configuration.training_fraction = 0.02;
  const auto outcome = RunPipeline(configuration);
  Expect(outcome.has_value(), "pipeline must succeed");
  Expect(outcome->training_object_count > 0, "training objects are reported");
  Expect(*outcome->quality.adjusted_rand_index > 0.99, "boxes ari");
  Expect(*outcome->quality_of_clustered_objects->adjusted_rand_index > 0.99,
         "ari of the clustered objects alone");
}

void SemiSupervisedKMeansUsesSeparateTrainingFile() {
  const auto blob = ReadCsv(kDataDirectory / "blob.csv");
  CsvTable training_table;
  training_table.header = blob->header;
  for (std::size_t row = 0; row < blob->rows.size(); row += 20) {
    training_table.rows.push_back(blob->rows[row]);
  }
  const auto training_path = std::filesystem::temp_directory_path() / "clustering_blob_training.csv";
  Expect(WriteCsv(training_path, training_table).has_value(), "training file is written");

  RunConfiguration configuration = ConfigurationFor("blob.csv", "clustering_blob_ssl.csv");
  configuration.cluster_count = 4;
  configuration.initializer = InitializerKind::kSeeded;
  configuration.training_path = training_path;
  const auto outcome = RunPipeline(configuration);
  Expect(outcome.has_value(), "pipeline must succeed");
  Expect(outcome->training_object_count == training_table.rows.size(), "training size");
  Expect(outcome->object_count == blob->rows.size() + training_table.rows.size(),
         "training set is merged with the clustered objects");

  const auto merged = ReadCsv(configuration.output_path);
  Expect(merged->header.back() == "origin", "output has the origin column");
  Expect(merged->rows.size() == outcome->object_count, "all objects are written");
  Expect(merged->rows.front().back() == "training", "training rows come first");
}

void DbscanClustersBoxesAndReportsNoise() {
  RunConfiguration configuration = ConfigurationFor("boxes.csv", "clustering_boxes_dbscan.csv");
  configuration.algorithm = AlgorithmKind::kDbscan;
  configuration.epsilon = 6.0;
  configuration.min_points = 5;
  const auto outcome = RunPipeline(configuration);
  Expect(outcome.has_value(), "pipeline must succeed");
  Expect(outcome->clustering.ClusterCount() == 30, "thirty boxes");
  Expect(outcome->clustering.NoiseCount() > 0, "some noise objects");
}

void PenguinsAreClusteredWithoutLabels() {
  RunConfiguration configuration;
  configuration.input_path = kDataDirectory / "penguins_sex.csv";
  configuration.output_path = std::filesystem::temp_directory_path() / "clustering_penguins.csv";
  configuration.feature_columns = {"culmen_length_mm", "culmen_depth_mm", "flipper_length_mm",
                                   "body_mass_g"};
  configuration.cluster_count = 3;
  configuration.initializer = InitializerKind::kKMeansPlusPlus;
  configuration.standardize_features = true;
  const auto outcome = RunPipeline(configuration);
  Expect(outcome.has_value(), "pipeline must succeed");
  Expect(outcome->skipped_row_count == 2, "two rows with missing measurements are skipped");
  Expect(!outcome->quality.adjusted_rand_index.has_value(), "no ground truth, no ari");
  Expect(outcome->quality.davies_bouldin_index.has_value(), "internal metrics are still computed");
}

void PipelineNeverOverwritesTheInputFile() {
  RunConfiguration configuration = ConfigurationFor("blob.csv", "unused.csv");
  configuration.output_path = configuration.input_path;
  configuration.cluster_count = 4;
  Expect(!RunPipeline(configuration).has_value(), "output equal to input must be rejected");
}

void PipelineReportsInvalidAlgorithmParameters() {
  RunConfiguration configuration = ConfigurationFor("blob.csv", "clustering_invalid.csv");
  configuration.cluster_count = 4;
  configuration.initializer = InitializerKind::kSeeded;
  configuration.label_column.reset();
  Expect(!RunPipeline(configuration).has_value(), "seeded init without labels must fail");
}

}

void AddPipelineTests(TestSuite& suite) {
  suite.Add("k-means++ clusters the blob file", KMeansPlusPlusClustersBlobFile);
  suite.Add("semi-supervised gmm recovers the boxes file", SemiSupervisedGaussianMixtureRecoversBoxes);
  suite.Add("semi-supervised k-means uses a separate training file", SemiSupervisedKMeansUsesSeparateTrainingFile);
  suite.Add("dbscan clusters the boxes file and reports noise", DbscanClustersBoxesAndReportsNoise);
  suite.Add("penguins are clustered without labels", PenguinsAreClusteredWithoutLabels);
  suite.Add("pipeline never overwrites the input file", PipelineNeverOverwritesTheInputFile);
  suite.Add("pipeline reports invalid algorithm parameters", PipelineReportsInvalidAlgorithmParameters);
}

}
