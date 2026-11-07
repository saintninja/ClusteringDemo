#pragma once

#include <span>

namespace clustering {

[[nodiscard]] double AdjustedRandIndex(std::span<const int> truth_labels,
                                       std::span<const int> predicted_labels);

}
