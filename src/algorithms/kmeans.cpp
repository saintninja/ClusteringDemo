#include "clustering/algorithms/kmeans.hpp"

#include <format>
#include <stdexcept>
#include <utility>

#include "clustering/algorithms/nearest_center.hpp"
#include "clustering/core/validation.hpp"

namespace clustering {

KMeans::KMeans(KMeansOptions options, std::unique_ptr<CentroidInitializer> initializer,
               MetricPtr metric, FixedAssignments fixed_assignments)
    : options_(options),
      initializer_(std::move(initializer)),
      metric_(std::move(metric)),
      fixed_assignments_(std::move(fixed_assignments)) {
  if (options_.cluster_count == 0 || options_.max_iterations == 0) {
    throw std::invalid_argument("k-means needs a positive cluster count and iteration limit");
  }
  if (!initializer_ || !metric_) {
    throw std::invalid_argument("k-means needs an initializer and a distance metric");
  }
}

ClusteringResult KMeans::Fit(const Matrix& data) {
  RequireClusterCount(data, options_.cluster_count);
  RequireValidFixedAssignments(fixed_assignments_, data.RowCount(), options_.cluster_count);

  ClusteringResult result;
  result.centers = initializer_->Initialize(data, options_.cluster_count, *metric_);
  result.labels = InitialLabels(data.RowCount());

  bool labels_changed = true;
  while (labels_changed && result.iterations < options_.max_iterations) {
    ++result.iterations;
    labels_changed = AssignObjectsToNearestCenters(data, result.centers, result.labels);
    MoveCentersToClusterMeans(data, result.labels, result.centers);
  }
  result.converged = !labels_changed;
  return result;
}

std::string KMeans::Describe() const {
  return std::format("k-means (k={}, initializer: {}, metric: {})", options_.cluster_count,
                     initializer_->Name(), metric_->Name());
}

std::vector<int> KMeans::InitialLabels(std::size_t object_count) const {
  std::vector<int> labels(object_count, kUnassignedLabel);
  for (std::size_t object = 0; object < object_count; ++object) {
    if (IsFixed(fixed_assignments_, object)) {
      labels[object] = *fixed_assignments_[object];
    }
  }
  return labels;
}

bool KMeans::AssignObjectsToNearestCenters(const Matrix& data, const Matrix& centers,
                                           std::vector<int>& labels) const {
  bool any_label_changed = false;
  for (std::size_t object = 0; object < data.RowCount(); ++object) {
    if (IsFixed(fixed_assignments_, object)) {
      continue;
    }
    const auto nearest = static_cast<int>(NearestCenter(data.Row(object), centers, *metric_));
    if (labels[object] != nearest) {
      labels[object] = nearest;
      any_label_changed = true;
    }
  }
  return any_label_changed;
}

void KMeans::MoveCentersToClusterMeans(const Matrix& data, const std::vector<int>& labels,
                                       Matrix& centers) {
  Matrix sums(centers.RowCount(), centers.ColumnCount());
  std::vector<std::size_t> cluster_sizes(centers.RowCount(), 0);
  for (std::size_t object = 0; object < data.RowCount(); ++object) {
    const auto cluster = static_cast<std::size_t>(labels[object]);
    for (std::size_t feature = 0; feature < data.ColumnCount(); ++feature) {
      sums(cluster, feature) += data(object, feature);
    }
    ++cluster_sizes[cluster];
  }
  for (std::size_t cluster = 0; cluster < centers.RowCount(); ++cluster) {
    if (cluster_sizes[cluster] == 0) {
      continue;
    }
    for (std::size_t feature = 0; feature < centers.ColumnCount(); ++feature) {
      centers(cluster, feature) = sums(cluster, feature) / static_cast<double>(cluster_sizes[cluster]);
    }
  }
}

}
