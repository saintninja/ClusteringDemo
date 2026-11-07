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

#include <optional>
#include <string>

#include "clustering/algorithms/clusterer.hpp"
#include "clustering/app/run_configuration.hpp"
#include "clustering/core/result.hpp"
#include "clustering/evaluation/quality_evaluator.hpp"

namespace clustering {

struct PipelineOutcome {
  std::string algorithm_description;
  std::size_t object_count = 0;
  std::size_t training_object_count = 0;
  std::size_t skipped_row_count = 0;
  ClusteringResult clustering;
  QualityReport quality;
  std::optional<QualityReport> quality_of_clustered_objects;
  double clustering_milliseconds = 0.0;
};

[[nodiscard]] Result<PipelineOutcome> RunPipeline(const RunConfiguration& configuration);

}
