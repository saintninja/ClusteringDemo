#include "clustering/initialization/training_split.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>

namespace clustering {

std::vector<std::size_t> SelectStratifiedTrainingRows(std::span<const int> class_labels,
                                                      double training_fraction,
                                                      std::uint64_t seed) {
  if (!(training_fraction > 0.0 && training_fraction <= 1.0)) {
    throw std::invalid_argument("training fraction must be in the interval (0, 1]");
  }
  std::map<int, std::vector<std::size_t>> objects_by_class;
  for (std::size_t object = 0; object < class_labels.size(); ++object) {
    objects_by_class[class_labels[object]].push_back(object);
  }

  std::mt19937_64 random_engine(seed);
  std::vector<std::size_t> training_rows;
  for (const auto& [class_label, objects] : objects_by_class) {
    const auto desired_count = static_cast<std::size_t>(
        std::llround(training_fraction * static_cast<double>(objects.size())));
    const std::size_t training_count = std::clamp<std::size_t>(desired_count, 1, objects.size());
    std::sample(objects.begin(), objects.end(), std::back_inserter(training_rows), training_count,
                random_engine);
  }
  std::ranges::sort(training_rows);
  return training_rows;
}

}
