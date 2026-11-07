#include "clustering/algorithms/gaussian_mixture.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

#include "clustering/algorithms/nearest_center.hpp"
#include "clustering/core/linear_algebra.hpp"
#include "clustering/core/validation.hpp"

namespace clustering {

namespace {

constexpr double kMinimumResponsibilitySum = 1e-10;
constexpr int kRegularizationAttempts = 6;
constexpr double kRegularizationGrowth = 10.0;

}

GaussianMixture::GaussianMixture(GaussianMixtureOptions options,
                                 std::unique_ptr<CentroidInitializer> initializer,
                                 MetricPtr metric, FixedAssignments fixed_assignments)
    : options_(options),
      initializer_(std::move(initializer)),
      metric_(std::move(metric)),
      fixed_assignments_(std::move(fixed_assignments)) {
  if (options_.component_count == 0 || options_.max_iterations == 0) {
    throw std::invalid_argument("gmm needs a positive component count and iteration limit");
  }
  if (!(options_.tolerance > 0.0) || !(options_.covariance_regularization > 0.0)) {
    throw std::invalid_argument("gmm tolerance and covariance regularization must be positive");
  }
  if (!initializer_ || !metric_) {
    throw std::invalid_argument("gmm needs an initializer and a distance metric");
  }
}

ClusteringResult GaussianMixture::Fit(const Matrix& data) {
  RequireClusterCount(data, options_.component_count);
  RequireValidFixedAssignments(fixed_assignments_, data.RowCount(), options_.component_count);

  const Matrix initial_centers =
      initializer_->Initialize(data, options_.component_count, *metric_);
  Matrix responsibilities = AssignHardResponsibilities(data, initial_centers);
  std::vector<Component> components = MaximizeParameters(data, responsibilities);

  ClusteringResult result;
  double previous_log_likelihood = -std::numeric_limits<double>::infinity();
  while (result.iterations < options_.max_iterations) {
    ++result.iterations;
    const double log_likelihood = ExpectResponsibilities(data, components, responsibilities);
    components = MaximizeParameters(data, responsibilities);
    if (std::abs(log_likelihood - previous_log_likelihood) < options_.tolerance) {
      result.converged = true;
      break;
    }
    previous_log_likelihood = log_likelihood;
  }

  static_cast<void>(ExpectResponsibilities(data, components, responsibilities));
  result.labels = MostLikelyComponents(responsibilities);
  result.centers = MeansOf(components);
  return result;
}

std::string GaussianMixture::Describe() const {
  return std::format("em-gmm (components={}, initializer: {}, metric for initialization: {})",
                     options_.component_count, initializer_->Name(), metric_->Name());
}

Matrix GaussianMixture::AssignHardResponsibilities(const Matrix& data,
                                                   const Matrix& centers) const {
  Matrix responsibilities(data.RowCount(), centers.RowCount());
  for (std::size_t object = 0; object < data.RowCount(); ++object) {
    responsibilities(object, NearestCenter(data.Row(object), centers, *metric_)) = 1.0;
    PinFixedObject(responsibilities, object);
  }
  return responsibilities;
}

std::vector<GaussianMixture::Component> GaussianMixture::MaximizeParameters(
    const Matrix& data, const Matrix& responsibilities) const {
  const std::size_t dimension = data.ColumnCount();
  std::vector<Component> components;
  components.reserve(responsibilities.ColumnCount());

  for (std::size_t component = 0; component < responsibilities.ColumnCount(); ++component) {
    double responsibility_sum = 0.0;
    std::vector<double> mean(dimension, 0.0);
    for (std::size_t object = 0; object < data.RowCount(); ++object) {
      const double responsibility = responsibilities(object, component);
      responsibility_sum += responsibility;
      for (std::size_t feature = 0; feature < dimension; ++feature) {
        mean[feature] += responsibility * data(object, feature);
      }
    }
    responsibility_sum = std::max(responsibility_sum, kMinimumResponsibilitySum);
    for (double& coordinate : mean) {
      coordinate /= responsibility_sum;
    }

    Matrix covariance(dimension, dimension);
    std::vector<double> deviation(dimension);
    for (std::size_t object = 0; object < data.RowCount(); ++object) {
      const double responsibility = responsibilities(object, component);
      for (std::size_t feature = 0; feature < dimension; ++feature) {
        deviation[feature] = data(object, feature) - mean[feature];
      }
      for (std::size_t first = 0; first < dimension; ++first) {
        for (std::size_t second = 0; second < dimension; ++second) {
          covariance(first, second) += responsibility * deviation[first] * deviation[second];
        }
      }
    }
    for (std::size_t first = 0; first < dimension; ++first) {
      for (std::size_t second = 0; second < dimension; ++second) {
        covariance(first, second) /= responsibility_sum;
      }
    }

    components.push_back(Component{
        .weight = responsibility_sum / static_cast<double>(data.RowCount()),
        .mean = std::move(mean),
        .covariance_factor = FactorizeWithRegularization(std::move(covariance))});
  }
  return components;
}

double GaussianMixture::ExpectResponsibilities(const Matrix& data,
                                               const std::vector<Component>& components,
                                               Matrix& responsibilities) const {
  const std::size_t dimension = data.ColumnCount();
  const double normalization = static_cast<double>(dimension) * std::log(2.0 * std::numbers::pi);
  std::vector<double> difference(dimension);
  std::vector<double> log_weighted_densities(components.size());
  double total_log_likelihood = 0.0;

  for (std::size_t object = 0; object < data.RowCount(); ++object) {
    for (std::size_t component = 0; component < components.size(); ++component) {
      const Component& current = components[component];
      for (std::size_t feature = 0; feature < dimension; ++feature) {
        difference[feature] = data(object, feature) - current.mean[feature];
      }
      current.covariance_factor.SolveLowerInPlace(difference);
      double squared_mahalanobis = 0.0;
      for (const double value : difference) {
        squared_mahalanobis += value * value;
      }
      log_weighted_densities[component] =
          std::log(current.weight) -
          0.5 * (normalization + current.covariance_factor.LogDeterminant() + squared_mahalanobis);
    }

    const double largest = *std::ranges::max_element(log_weighted_densities);
    double exponent_sum = 0.0;
    for (const double value : log_weighted_densities) {
      exponent_sum += std::exp(value - largest);
    }
    const double log_normalizer = largest + std::log(exponent_sum);
    for (std::size_t component = 0; component < components.size(); ++component) {
      responsibilities(object, component) =
          std::exp(log_weighted_densities[component] - log_normalizer);
    }
    PinFixedObject(responsibilities, object);
    total_log_likelihood += log_normalizer;
  }
  return total_log_likelihood / static_cast<double>(data.RowCount());
}

void GaussianMixture::PinFixedObject(Matrix& responsibilities, std::size_t object) const {
  if (!IsFixed(fixed_assignments_, object)) {
    return;
  }
  std::ranges::fill(responsibilities.MutableRow(object), 0.0);
  responsibilities(object, static_cast<std::size_t>(*fixed_assignments_[object])) = 1.0;
}

CholeskyFactor GaussianMixture::FactorizeWithRegularization(Matrix covariance) const {
  double regularization = options_.covariance_regularization;
  for (int attempt = 0; attempt < kRegularizationAttempts; ++attempt) {
    Matrix regularized = covariance;
    AddToDiagonal(regularized, regularization);
    if (auto factor = CholeskyFactor::Compute(regularized)) {
      return *factor;
    }
    regularization *= kRegularizationGrowth;
  }
  throw std::runtime_error("gmm component covariance is singular even after regularization");
}

std::vector<int> GaussianMixture::MostLikelyComponents(const Matrix& responsibilities) {
  std::vector<int> labels(responsibilities.RowCount());
  for (std::size_t object = 0; object < responsibilities.RowCount(); ++object) {
    const Point row = responsibilities.Row(object);
    labels[object] = static_cast<int>(std::ranges::max_element(row) - row.begin());
  }
  return labels;
}

Matrix GaussianMixture::MeansOf(const std::vector<Component>& components) {
  Matrix means;
  for (const Component& component : components) {
    means.AppendRow(component.mean);
  }
  return means;
}

}
