// ClusteringDemo - pure C++ library for demo clustering algos
// Copyright (C) 2026 Stanislav Furmavnin (GitHub: saintninja)
// SPDX-License-Identifier: LicenseRef-EIL-1.0
//
// This program is licensed under the Educational Integrity License (EIL) v1.0.
// You may use, modify, and distribute this file strictly for Permitted Uses
// (commercial or personal non-evaluated projects).
//
// ANY USE FOR ACADEMIC WORK OR SUBMISSION FOR CREDIT IS STRICTLY PROHIBITED.
// Any modification or redistribution must retain this license under Copyleft terms.
//
// See the full LICENSE file in the repository root for terms and conditions.
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
