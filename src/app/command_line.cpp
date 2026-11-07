#include "clustering/app/command_line.hpp"

#include <charconv>
#include <filesystem>
#include <cmath>
#include <cstdlib>
#include <format>
#include <functional>
#include <map>

namespace clustering {

namespace {

using OptionHandler = std::function<Result<void>(std::string_view)>;

template <std::integral Integer>
Result<Integer> ParseInteger(std::string_view option, std::string_view text) {
  Integer value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    return Failure(std::format("option {} expects an integer, got '{}'", option, text));
  }
  return value;
}

Result<double> ParseReal(std::string_view option, std::string_view text) {
  const std::string copy(text);
  char* parsed_end = nullptr;
  const double value = std::strtod(copy.c_str(), &parsed_end);
  if (copy.empty() || parsed_end != copy.c_str() + copy.size() || !std::isfinite(value)) {
    return Failure(std::format("option {} expects a number, got '{}'", option, text));
  }
  return value;
}

std::vector<std::string> SplitByComma(std::string_view text) {
  std::vector<std::string> parts;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    parts.emplace_back(text.substr(start, end - start));
    start = end + 1;
  }
  return parts;
}

template <typename Target, typename Value>
Result<void> Store(Target& target, Result<Value> parsed) {
  if (!parsed) {
    return std::unexpected(parsed.error());
  }
  target = *parsed;
  return {};
}

template <typename Kind>
Result<void> StoreKind(Kind& target, std::string_view option, std::string_view text,
                       std::optional<Kind> (*parse)(std::string_view)) {
  const std::optional<Kind> kind = parse(text);
  if (!kind) {
    return Failure(std::format("unknown value '{}' for option {}", text, option));
  }
  target = *kind;
  return {};
}

std::optional<AlgorithmKind> ParseAlgorithmKind(std::string_view name) {
  if (name == "kmeans") return AlgorithmKind::kKMeans;
  if (name == "gmm") return AlgorithmKind::kGaussianMixture;
  if (name == "dbscan") return AlgorithmKind::kDbscan;
  if (name == "affinity") return AlgorithmKind::kAffinityPropagation;
  return std::nullopt;
}

std::map<std::string_view, OptionHandler> BuildHandlers(RunConfiguration& configuration) {
  return {
      {"--input", [&](std::string_view value) -> Result<void> {
         configuration.input_path = std::string(value);
         return {};
       }},
      {"--output", [&](std::string_view value) -> Result<void> {
         configuration.output_path = std::string(value);
         return {};
       }},
      {"--columns", [&](std::string_view value) -> Result<void> {
         configuration.feature_columns = SplitByComma(value);
         return {};
       }},
      {"--labels", [&](std::string_view value) -> Result<void> {
         configuration.label_column = std::string(value);
         return {};
       }},
      {"--algorithm", [&](std::string_view value) {
         return StoreKind(configuration.algorithm, "--algorithm", value, ParseAlgorithmKind);
       }},
      {"--metric", [&](std::string_view value) {
         return StoreKind(configuration.metric, "--metric", value, ParseMetricKind);
       }},
      {"--eval-metric", [&](std::string_view value) {
         return StoreKind(configuration.evaluation_metric, "--eval-metric", value, ParseMetricKind);
       }},
      {"--init", [&](std::string_view value) {
         return StoreKind(configuration.initializer, "--init", value, ParseInitializerKind);
       }},
      {"--k", [&](std::string_view value) {
         return Store(configuration.cluster_count, ParseInteger<std::size_t>("--k", value));
       }},
      {"--training", [&](std::string_view value) -> Result<void> {
         configuration.training_path = std::filesystem::path(std::string(value));
         return {};
       }},
      {"--training-labels", [&](std::string_view value) -> Result<void> {
         configuration.training_label_column = std::string(value);
         return {};
       }},
      {"--training-fraction", [&](std::string_view value) {
         return Store(configuration.training_fraction, ParseReal("--training-fraction", value));
       }},
      {"--max-iter", [&](std::string_view value) {
         return Store(configuration.max_iterations, ParseInteger<std::size_t>("--max-iter", value));
       }},
      {"--eps", [&](std::string_view value) {
         return Store(configuration.epsilon, ParseReal("--eps", value));
       }},
      {"--min-pts", [&](std::string_view value) {
         return Store(configuration.min_points, ParseInteger<std::size_t>("--min-pts", value));
       }},
      {"--damping", [&](std::string_view value) {
         return Store(configuration.damping, ParseReal("--damping", value));
       }},
      {"--preference", [&](std::string_view value) {
         return Store(configuration.preference, ParseReal("--preference", value));
       }},
      {"--sample", [&](std::string_view value) {
         return Store(configuration.sample_size, ParseInteger<std::size_t>("--sample", value));
       }},
      {"--seed", [&](std::string_view value) {
         return Store(configuration.seed, ParseInteger<std::uint64_t>("--seed", value));
       }},
  };
}

Result<void> ValidateTrainingOptions(const RunConfiguration& configuration) {
  if (configuration.training_path && configuration.training_fraction) {
    return Failure("options --training and --training-fraction are mutually exclusive");
  }
  const bool seeded = configuration.initializer == InitializerKind::kSeeded;
  if (seeded && !configuration.UsesTrainingSet()) {
    return Failure("--init seeded needs a training set: use --training or --training-fraction");
  }
  if (!seeded && configuration.UsesTrainingSet()) {
    return Failure("a training set is used only together with --init seeded");
  }
  const bool supports_training = configuration.algorithm == AlgorithmKind::kKMeans ||
                                 configuration.algorithm == AlgorithmKind::kGaussianMixture;
  if (configuration.UsesTrainingSet() && !supports_training) {
    return Failure("training sets are supported only by kmeans and gmm");
  }
  if (configuration.training_fraction &&
      !(*configuration.training_fraction > 0.0 && *configuration.training_fraction < 1.0)) {
    return Failure("option --training-fraction must be between 0 and 1 exclusive");
  }
  if (configuration.training_path && !configuration.training_label_column &&
      !configuration.label_column) {
    return Failure("option --training-labels is required with --training");
  }
  if (configuration.training_fraction && !configuration.label_column) {
    return Failure("option --training-fraction needs --labels with ground-truth classes");
  }
  return {};
}

Result<void> ValidateConfiguration(const RunConfiguration& configuration) {
  if (configuration.input_path.empty() || configuration.output_path.empty() ||
      configuration.feature_columns.empty()) {
    return Failure("options --input, --output and --columns are required");
  }
  const bool needs_cluster_count = configuration.algorithm == AlgorithmKind::kKMeans ||
                                   configuration.algorithm == AlgorithmKind::kGaussianMixture;
  if (needs_cluster_count && configuration.cluster_count == 0) {
    return Failure("option --k is required for kmeans and gmm");
  }
  if (configuration.algorithm == AlgorithmKind::kDbscan && !configuration.epsilon) {
    return Failure("option --eps is required for dbscan");
  }
  return ValidateTrainingOptions(configuration);
}

}

Result<RunConfiguration> ParseCommandLine(std::span<const std::string_view> arguments) {
  RunConfiguration configuration;
  const auto handlers = BuildHandlers(configuration);

  for (std::size_t position = 0; position < arguments.size(); ++position) {
    const std::string_view option = arguments[position];
    if (option == "--standardize") {
      configuration.standardize_features = true;
      continue;
    }
    const auto handler = handlers.find(option);
    if (handler == handlers.end()) {
      return Failure(std::format("unknown option '{}'", option));
    }
    if (position + 1 >= arguments.size()) {
      return Failure(std::format("option {} needs a value", option));
    }
    if (auto stored = handler->second(arguments[++position]); !stored) {
      return std::unexpected(stored.error());
    }
  }

  if (auto valid = ValidateConfiguration(configuration); !valid) {
    return std::unexpected(valid.error());
  }
  return configuration;
}

std::string_view UsageText() {
  return R"(Usage: clustering_cli --input <file.csv> --output <result.csv> --columns <c1,c2,...> [options]

Data:
  --labels <column>       column with ground-truth classes of the input (enables ARI and AMI)
  --standardize           z-score the selected features before clustering
  --sample <n>            cluster a random sample of n objects
  --seed <n>              random seed (default 42)

Algorithm:
  --algorithm kmeans|gmm|dbscan|affinity      (default kmeans)
  --metric euclidean|manhattan|mahalanobis    (default euclidean)
  --eval-metric <metric>                      metric for silhouette and Davies-Bouldin

  kmeans, gmm:  --k <clusters>  --init forgy|kmeans++|seeded  [--max-iter 300]
  dbscan:       --eps <radius>  [--min-pts 5]
  affinity:     [--damping 0.7]  [--preference <value>]  [--max-iter 300]

Semi-supervised start (--init seeded, kmeans and gmm only): initial centers are the class means of a
training set whose labels stay fixed; the training set is merged into the output.
  --training <file.csv> --training-labels <column>   labelled training set from a separate file
  --training-fraction <0..1>                         take a stratified part of the labelled input as training
)";
}

}
