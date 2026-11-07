#include <filesystem>
#include <fstream>
#include <string>

#include "clustering/app/command_line.hpp"
#include "clustering/io/clustered_csv_writer.hpp"
#include "clustering/io/csv_table.hpp"
#include "clustering/io/dataset_loader.hpp"
#include "test_support.hpp"

namespace clustering::tests {

namespace {

std::filesystem::path WriteTemporaryFile(const std::string& name, const std::string& content) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream(path, std::ios::binary) << content;
  return path;
}

void CsvReaderHandlesByteOrderMarkAndWindowsLineEndings() {
  const auto path = WriteTemporaryFile("clustering_bom.csv", "\xEF\xBB\xBF" "a,b\r\n1,2\r\n3,4\r\n");
  const auto table = ReadCsv(path);
  Expect(table.has_value(), "file must be readable");
  Expect(table->header.front() == "a", "byte order mark must be stripped");
  Expect(table->rows.size() == 2 && table->rows[1][1] == "4", "rows must be parsed");
}

void CsvReaderReportsRaggedRows() {
  const auto path = WriteTemporaryFile("clustering_ragged.csv", "a,b\n1,2,3\n");
  Expect(!ReadCsv(path).has_value(), "ragged row must be an error");
}

void LoaderSkipsRowsWithMissingValues() {
  const auto path = WriteTemporaryFile("clustering_missing.csv", "x,y,class\n1,2,u\nNA,3,v\n4,,v\n5,6,v\n");
  const auto table = ReadCsv(path);
  const auto dataset =
      LoadDataset(*table, {.feature_columns = {"x", "y"}, .label_column = std::string("class")});
  Expect(dataset.has_value(), "dataset must load");
  Expect(dataset->features.RowCount() == 2, "two complete rows remain");
  Expect(dataset->skipped_row_count == 2, "two incomplete rows are skipped");
  Expect(dataset->label_texts == std::vector<std::string>({"u", "v"}), "label texts are kept");
  Expect(dataset->source_rows == std::vector<std::size_t>({0, 3}), "source rows are remembered");
}

void LoaderRejectsNonNumericFeatures() {
  const auto path = WriteTemporaryFile("clustering_text.csv", "x,y\n1,abc\n");
  const auto table = ReadCsv(path);
  Expect(!LoadDataset(*table, {.feature_columns = {"x", "y"}}).has_value(), "text is not numeric");
}

void LoaderRejectsUnknownColumn() {
  const auto path = WriteTemporaryFile("clustering_unknown.csv", "x,y\n1,2\n");
  const auto table = ReadCsv(path);
  Expect(!LoadDataset(*table, {.feature_columns = {"x", "z"}}).has_value(), "unknown column");
}

void LoaderSamplesRequestedNumberOfRows() {
  std::string content = "x\n";
  for (int value = 0; value < 50; ++value) {
    content += std::to_string(value) + "\n";
  }
  const auto table = ReadCsv(WriteTemporaryFile("clustering_sample.csv", content));
  const auto dataset = LoadDataset(*table, {.feature_columns = {"x"}, .sample_size = 10, .seed = 3});
  Expect(dataset->features.RowCount() == 10, "sample size must be respected");
}

void CommandLineParsesFullConfiguration() {
  const std::vector<std::string_view> arguments{
      "--input", "in.csv", "--output", "out.csv", "--columns", "x,y", "--algorithm", "gmm",
      "--k", "4", "--init", "ssl", "--training", "train.csv", "--training-labels", "class",
      "--metric", "mahalanobis", "--standardize", "--seed", "9"};
  const auto configuration = ParseCommandLine(arguments);
  Expect(configuration.has_value(), "valid arguments must parse");
  Expect(configuration->algorithm == AlgorithmKind::kGaussianMixture, "algorithm");
  Expect(configuration->initializer == InitializerKind::kSeeded, "initializer alias");
  Expect(configuration->metric == MetricKind::kMahalanobis, "metric");
  Expect(configuration->feature_columns.size() == 2, "columns");
  Expect(configuration->standardize_features && configuration->seed == 9, "flags");
  Expect(configuration->UsesTrainingSet() && *configuration->training_label_column == "class",
         "training options");
}

void CommandLineValidatesTrainingOptions() {
  const std::vector<std::string_view> seeded_without_training{
      "--input", "a", "--output", "b", "--columns", "x", "--k", "2", "--init", "seeded"};
  Expect(!ParseCommandLine(seeded_without_training).has_value(), "seeded needs a training set");
  const std::vector<std::string_view> training_without_seeded{
      "--input", "a", "--output", "b", "--columns", "x", "--k", "2", "--labels", "c",
      "--training-fraction", "0.1"};
  Expect(!ParseCommandLine(training_without_seeded).has_value(), "training set needs --init seeded");
  const std::vector<std::string_view> training_for_dbscan{
      "--input", "a", "--output", "b", "--columns", "x", "--algorithm", "dbscan", "--eps", "1",
      "--init", "seeded", "--labels", "c", "--training-fraction", "0.1"};
  Expect(!ParseCommandLine(training_for_dbscan).has_value(), "dbscan has no training mode");
  const std::vector<std::string_view> both_sources{
      "--input", "a", "--output", "b", "--columns", "x", "--k", "2", "--init", "seeded", "--labels",
      "c", "--training", "t", "--training-fraction", "0.1"};
  Expect(!ParseCommandLine(both_sources).has_value(), "training sources are exclusive");
}

void ClusteredCsvMergesTrainingAndClusteredRows() {
  CsvTable input;
  input.header = {"x", "note"};
  input.rows = {{"1", "a"}, {"2", "b"}};
  CsvTable training;
  training.header = {"note", "x", "class"};
  training.rows = {{"t", "9", "k"}};
  const std::vector<SourceRow> sources{{&training, 0, RowOrigin::kTraining},
                                       {&input, 1, RowOrigin::kClustered}};
  const std::vector<int> labels{4, 5};
  const auto path = std::filesystem::temp_directory_path() / "clustering_merged.csv";
  Expect(WriteClusteredCsv(path, input, sources, labels).has_value(), "writing succeeds");
  const auto merged = ReadCsv(path);
  Expect(merged->header == std::vector<std::string>({"x", "note", "cluster", "origin"}), "header");
  Expect(merged->rows[0] == std::vector<std::string>({"9", "t", "4", "training"}),
         "training columns are mapped by name");
  Expect(merged->rows[1] == std::vector<std::string>({"2", "b", "5", "clustered"}), "clustered row");
}

void CommandLineRequiresAlgorithmSpecificOptions() {
  const std::vector<std::string_view> kmeans_without_k{"--input", "a", "--output", "b", "--columns", "x"};
  Expect(!ParseCommandLine(kmeans_without_k).has_value(), "k-means needs --k");
  const std::vector<std::string_view> dbscan_without_eps{"--input", "a", "--output", "b",
                                                         "--columns", "x", "--algorithm", "dbscan"};
  Expect(!ParseCommandLine(dbscan_without_eps).has_value(), "dbscan needs --eps");
}

void CommandLineRejectsBadInput() {
  const std::vector<std::string_view> unknown_option{"--bogus", "1"};
  Expect(!ParseCommandLine(unknown_option).has_value(), "unknown option");
  const std::vector<std::string_view> bad_number{"--input", "a", "--output", "b", "--columns", "x", "--k", "three"};
  Expect(!ParseCommandLine(bad_number).has_value(), "non-numeric --k");
}

}

void AddInputOutputTests(TestSuite& suite) {
  suite.Add("csv reader handles byte order mark and windows line endings",
            CsvReaderHandlesByteOrderMarkAndWindowsLineEndings);
  suite.Add("csv reader reports ragged rows", CsvReaderReportsRaggedRows);
  suite.Add("loader skips rows with missing values", LoaderSkipsRowsWithMissingValues);
  suite.Add("loader rejects non-numeric features", LoaderRejectsNonNumericFeatures);
  suite.Add("loader rejects unknown column", LoaderRejectsUnknownColumn);
  suite.Add("loader samples requested number of rows", LoaderSamplesRequestedNumberOfRows);
  suite.Add("command line parses full configuration", CommandLineParsesFullConfiguration);
  suite.Add("command line requires algorithm specific options", CommandLineRequiresAlgorithmSpecificOptions);
  suite.Add("command line rejects bad input", CommandLineRejectsBadInput);
  suite.Add("command line validates training options", CommandLineValidatesTrainingOptions);
  suite.Add("clustered csv merges training and clustered rows", ClusteredCsvMergesTrainingAndClusteredRows);
}

}
