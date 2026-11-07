#pragma once

#include <span>
#include <string_view>

#include "clustering/app/run_configuration.hpp"
#include "clustering/core/result.hpp"

namespace clustering {

[[nodiscard]] Result<RunConfiguration> ParseCommandLine(std::span<const std::string_view> arguments);
[[nodiscard]] std::string_view UsageText();

}
