#pragma once

#include <memory>
#include <optional>

#include "clustering/algorithms/clusterer.hpp"
#include "clustering/app/run_configuration.hpp"
#include "clustering/distance/distance_metric.hpp"
#include "clustering/initialization/training_labels.hpp"

namespace clustering {

[[nodiscard]] std::unique_ptr<Clusterer> MakeClusterer(
    const RunConfiguration& configuration, const std::optional<TrainingLabels>& training,
    std::size_t object_count, MetricPtr metric);

}
