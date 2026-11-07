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

#include <cstddef>
#include <string>
#include <vector>

#include "clustering/core/matrix.hpp"

namespace clustering {

inline constexpr int kNoiseLabel = -1;

struct ClusteringResult {
  std::vector<int> labels;
  Matrix centers;
  std::size_t iterations = 0;
  bool converged = false;

  [[nodiscard]] std::size_t ClusterCount() const;
  [[nodiscard]] std::size_t NoiseCount() const;
};

class Clusterer {
 public:
  virtual ~Clusterer() = default;

  [[nodiscard]] virtual ClusteringResult Fit(const Matrix& data) = 0;
  [[nodiscard]] virtual std::string Describe() const = 0;
};

}
