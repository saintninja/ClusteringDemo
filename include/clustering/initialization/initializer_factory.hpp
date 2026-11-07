#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

#include "clustering/initialization/centroid_initializer.hpp"
#include "clustering/initialization/training_labels.hpp"

namespace clustering {

enum class InitializerKind { kForgy, kKMeansPlusPlus, kSeeded };

struct InitializerSettings {
  InitializerKind kind = InitializerKind::kForgy;
  std::uint64_t seed = 0;
  std::optional<TrainingLabels> training{};
};

[[nodiscard]] std::optional<InitializerKind> ParseInitializerKind(std::string_view name);
[[nodiscard]] std::unique_ptr<CentroidInitializer> MakeInitializer(
    const InitializerSettings& settings);

}
