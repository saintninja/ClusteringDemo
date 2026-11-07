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
#include <random>

#include "clustering/initialization/centroid_initializer.hpp"
#include "clustering/initialization/training_labels.hpp"

namespace clustering {

class SeededInitializer final : public CentroidInitializer {
 public:
  SeededInitializer(TrainingLabels training, std::uint64_t seed);

  [[nodiscard]] Matrix Initialize(const Matrix& data, std::size_t cluster_count,
                                  const DistanceMetric& metric) override;
  [[nodiscard]] std::string Name() const override;

 private:
  TrainingLabels training_;
  std::mt19937_64 random_engine_;
};

}
