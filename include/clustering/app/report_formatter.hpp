#pragma once

#include <string>

#include "clustering/app/pipeline.hpp"

namespace clustering {

[[nodiscard]] std::string FormatOutcome(const PipelineOutcome& outcome);

}
