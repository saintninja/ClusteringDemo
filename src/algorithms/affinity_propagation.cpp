#include "clustering/algorithms/affinity_propagation.hpp"

#include <algorithm>
#include <format>
#include <limits>
#include <stdexcept>
#include <utility>

#include "clustering/core/linear_algebra.hpp"
#include "clustering/core/validation.hpp"

namespace clustering {

AffinityPropagation::AffinityPropagation(AffinityPropagationOptions options, MetricPtr metric)
    : options_(options), metric_(std::move(metric)) {
  if (options_.damping < 0.5 || options_.damping >= 1.0) {
    throw std::invalid_argument("affinity propagation damping must be in the range [0.5, 1)");
  }
  if (options_.max_iterations == 0 || options_.convergence_iterations == 0) {
    throw std::invalid_argument("affinity propagation iteration limits must be positive");
  }
  if (!metric_) {
    throw std::invalid_argument("affinity propagation needs a distance metric");
  }
}

ClusteringResult AffinityPropagation::Fit(const Matrix& data) {
  RequireNonEmpty(data);
  if (data.RowCount() < 2) {
    throw std::invalid_argument("affinity propagation needs at least two objects");
  }
  if (data.RowCount() > kMaximumObjectCount) {
    throw std::invalid_argument(
        std::format("affinity propagation needs O(n^2) memory; {} objects exceed the limit of {}, "
                    "use a random sample",
                    data.RowCount(), kMaximumObjectCount));
  }

  const Matrix similarities = BuildSimilarities(data);
  Matrix responsibilities(data.RowCount(), data.RowCount());
  Matrix availabilities(data.RowCount(), data.RowCount());

  ClusteringResult result;
  std::vector<std::size_t> exemplars;
  std::size_t stable_iterations = 0;
  while (result.iterations < options_.max_iterations) {
    ++result.iterations;
    UpdateResponsibilities(similarities, availabilities, responsibilities);
    UpdateAvailabilities(responsibilities, availabilities);

    std::vector<std::size_t> current_exemplars = FindExemplars(responsibilities, availabilities);
    const bool exemplars_unchanged = !current_exemplars.empty() && current_exemplars == exemplars;
    stable_iterations = exemplars_unchanged ? stable_iterations + 1 : 0;
    exemplars = std::move(current_exemplars);

    if (stable_iterations >= options_.convergence_iterations) {
      result.converged = true;
      break;
    }
  }

  result.labels = AssignToExemplars(similarities, exemplars);
  result.centers = SelectRows(data, exemplars);
  return result;
}

std::string AffinityPropagation::Describe() const {
  return std::format("affinity propagation (damping={}, metric: {})", options_.damping,
                     metric_->Name());
}

Matrix AffinityPropagation::BuildSimilarities(const Matrix& data) const {
  const std::size_t object_count = data.RowCount();
  Matrix similarities(object_count, object_count);
  for (std::size_t first = 0; first < object_count; ++first) {
    for (std::size_t second = first + 1; second < object_count; ++second) {
      const double distance = metric_->Distance(data.Row(first), data.Row(second));
      similarities(first, second) = -distance * distance;
      similarities(second, first) = -distance * distance;
    }
  }
  const double preference = ResolvePreference(similarities);
  for (std::size_t object = 0; object < object_count; ++object) {
    similarities(object, object) = preference;
  }
  return similarities;
}

double AffinityPropagation::ResolvePreference(const Matrix& similarities) const {
  if (options_.preference) {
    return *options_.preference;
  }
  const std::size_t object_count = similarities.RowCount();
  std::vector<double> pairwise;
  pairwise.reserve(object_count * (object_count - 1) / 2);
  for (std::size_t first = 0; first < object_count; ++first) {
    for (std::size_t second = first + 1; second < object_count; ++second) {
      pairwise.push_back(similarities(first, second));
    }
  }
  const auto median_position = pairwise.begin() + static_cast<std::ptrdiff_t>(pairwise.size() / 2);
  std::ranges::nth_element(pairwise, median_position);
  return *median_position;
}

void AffinityPropagation::UpdateResponsibilities(const Matrix& similarities,
                                                 const Matrix& availabilities,
                                                 Matrix& responsibilities) const {
  const std::size_t object_count = similarities.RowCount();
  constexpr double kNegativeInfinity = -std::numeric_limits<double>::infinity();
  for (std::size_t object = 0; object < object_count; ++object) {
    const Point similarity_row = similarities.Row(object);
    const Point availability_row = availabilities.Row(object);

    double best = kNegativeInfinity;
    double second_best = kNegativeInfinity;
    std::size_t best_candidate = 0;
    for (std::size_t candidate = 0; candidate < object_count; ++candidate) {
      const double value = availability_row[candidate] + similarity_row[candidate];
      if (value > best) {
        second_best = best;
        best = value;
        best_candidate = candidate;
      } else if (value > second_best) {
        second_best = value;
      }
    }

    std::span<double> responsibility_row = responsibilities.MutableRow(object);
    for (std::size_t candidate = 0; candidate < object_count; ++candidate) {
      const double strongest_competitor = candidate == best_candidate ? second_best : best;
      const double fresh = similarity_row[candidate] - strongest_competitor;
      responsibility_row[candidate] =
          options_.damping * responsibility_row[candidate] + (1.0 - options_.damping) * fresh;
    }
  }
}

void AffinityPropagation::UpdateAvailabilities(const Matrix& responsibilities,
                                               Matrix& availabilities) const {
  const std::size_t object_count = responsibilities.RowCount();
  std::vector<double> positive_responsibility_sums(object_count, 0.0);
  for (std::size_t object = 0; object < object_count; ++object) {
    const Point responsibility_row = responsibilities.Row(object);
    for (std::size_t candidate = 0; candidate < object_count; ++candidate) {
      if (candidate != object) {
        positive_responsibility_sums[candidate] += std::max(0.0, responsibility_row[candidate]);
      }
    }
  }

  for (std::size_t object = 0; object < object_count; ++object) {
    const Point responsibility_row = responsibilities.Row(object);
    std::span<double> availability_row = availabilities.MutableRow(object);
    for (std::size_t candidate = 0; candidate < object_count; ++candidate) {
      double fresh = positive_responsibility_sums[candidate];
      if (candidate != object) {
        fresh = std::min(0.0, responsibilities(candidate, candidate) + fresh -
                                  std::max(0.0, responsibility_row[candidate]));
      }
      availability_row[candidate] =
          options_.damping * availability_row[candidate] + (1.0 - options_.damping) * fresh;
    }
  }
}

std::vector<std::size_t> AffinityPropagation::FindExemplars(const Matrix& responsibilities,
                                                            const Matrix& availabilities) {
  std::vector<std::size_t> exemplars;
  for (std::size_t object = 0; object < responsibilities.RowCount(); ++object) {
    if (availabilities(object, object) + responsibilities(object, object) > 0.0) {
      exemplars.push_back(object);
    }
  }
  return exemplars;
}

std::vector<int> AffinityPropagation::AssignToExemplars(
    const Matrix& similarities, const std::vector<std::size_t>& exemplars) {
  std::vector<int> labels(similarities.RowCount(), kNoiseLabel);
  if (exemplars.empty()) {
    return labels;
  }
  for (std::size_t object = 0; object < similarities.RowCount(); ++object) {
    const auto own_position = std::ranges::find(exemplars, object);
    if (own_position != exemplars.end()) {
      labels[object] = static_cast<int>(own_position - exemplars.begin());
      continue;
    }
    std::size_t best = 0;
    for (std::size_t position = 1; position < exemplars.size(); ++position) {
      if (similarities(object, exemplars[position]) > similarities(object, exemplars[best])) {
        best = position;
      }
    }
    labels[object] = static_cast<int>(best);
  }
  return labels;
}

}
