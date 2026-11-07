#pragma once

#include <span>

namespace clustering {

[[nodiscard]] double AdjustedMutualInformation(std::span<const int> truth_labels,
                                               std::span<const int> predicted_labels);

}
