#pragma once

#include <random>
#include <vector>

#include "clustering/core/matrix.hpp"

namespace clustering::tests {

struct LabelledBlobs {
  clustering::Matrix points;
  std::vector<int> labels;
};

inline LabelledBlobs MakeThreeBlobs(unsigned long long seed = 7, std::size_t points_per_blob = 60) {
  const std::vector<std::vector<double>> centers{{0.0, 0.0}, {10.0, 0.0}, {5.0, 9.0}};
  std::mt19937_64 random_engine(seed);
  std::normal_distribution<double> noise(0.0, 0.5);

  LabelledBlobs blobs;
  for (std::size_t blob = 0; blob < centers.size(); ++blob) {
    for (std::size_t index = 0; index < points_per_blob; ++index) {
      const std::vector<double> point{centers[blob][0] + noise(random_engine),
                                      centers[blob][1] + noise(random_engine)};
      blobs.points.AppendRow(point);
      blobs.labels.push_back(static_cast<int>(blob));
    }
  }
  return blobs;
}

}
