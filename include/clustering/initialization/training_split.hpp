#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace clustering {

[[nodiscard]] std::vector<std::size_t> SelectStratifiedTrainingRows(
    std::span<const int> class_labels, double training_fraction, std::uint64_t seed);

}
