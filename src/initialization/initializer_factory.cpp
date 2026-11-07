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
#include "clustering/initialization/initializer_factory.hpp"

#include <stdexcept>
#include <utility>

#include "clustering/initialization/forgy_initializer.hpp"
#include "clustering/initialization/kmeans_plus_plus_initializer.hpp"
#include "clustering/initialization/seeded_initializer.hpp"

namespace clustering {

std::optional<InitializerKind> ParseInitializerKind(std::string_view name) {
  if (name == "forgy") return InitializerKind::kForgy;
  if (name == "kmeans++") return InitializerKind::kKMeansPlusPlus;
  if (name == "seeded" || name == "ssl") return InitializerKind::kSeeded;
  return std::nullopt;
}

std::unique_ptr<CentroidInitializer> MakeInitializer(const InitializerSettings& settings) {
  switch (settings.kind) {
    case InitializerKind::kForgy:
      return std::make_unique<ForgyInitializer>(settings.seed);
    case InitializerKind::kKMeansPlusPlus:
      return std::make_unique<KMeansPlusPlusInitializer>(settings.seed);
    case InitializerKind::kSeeded:
      if (!settings.training) {
        throw std::invalid_argument("seeded initialization requires a training set");
      }
      return std::make_unique<SeededInitializer>(*settings.training, settings.seed);
  }
  std::unreachable();
}

}
