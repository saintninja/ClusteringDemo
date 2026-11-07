// ClusteringDemo - pure C++ library for demo clustering algos
// Copyright (C) 2026 Stanislav Furmavnin (GitHub: saintninja)
// SPDX-License-Identifier: LicenseRef-EIL-1.0
//
// This program is licensed under the Educational Integrity License (EIL) v1.0.
// You may use, modify, and distribute this file strictly for Permitted Uses
// (commercial or personal non-evaluated projects).
//
// ANY USE FOR ACADEMIC WORK OR SUBMISSION FOR CREDIT IS STRICTLY PROHIBITED.
// Any modification or redistribution must retain this license under Copyleft terms.
//
// See the full LICENSE file in the repository root for terms and conditions.
#pragma once

#include <memory>
#include <vector>

#include "clustering/algorithms/clusterer.hpp"
#include "clustering/core/cholesky_factor.hpp"
#include "clustering/core/fixed_assignments.hpp"
#include "clustering/distance/distance_metric.hpp"
#include "clustering/initialization/centroid_initializer.hpp"

namespace clustering {

struct GaussianMixtureOptions {
  std::size_t component_count = 0;
  std::size_t max_iterations = 200;
  double tolerance = 1e-4;
  double covariance_regularization = 1e-6;
};

class GaussianMixture final : public Clusterer {
 public:
  GaussianMixture(GaussianMixtureOptions options, std::unique_ptr<CentroidInitializer> initializer,
                  MetricPtr metric, FixedAssignments fixed_assignments = {});

  [[nodiscard]] ClusteringResult Fit(const Matrix& data) override;
  [[nodiscard]] std::string Describe() const override;

 private:
  struct Component {
    double weight;
    std::vector<double> mean;
    CholeskyFactor covariance_factor;
  };

  [[nodiscard]] Matrix AssignHardResponsibilities(const Matrix& data, const Matrix& centers) const;
  [[nodiscard]] std::vector<Component> MaximizeParameters(const Matrix& data,
                                                          const Matrix& responsibilities) const;
  [[nodiscard]] double ExpectResponsibilities(const Matrix& data,
                                              const std::vector<Component>& components,
                                              Matrix& responsibilities) const;
  void PinFixedObject(Matrix& responsibilities, std::size_t object) const;
  [[nodiscard]] CholeskyFactor FactorizeWithRegularization(Matrix covariance) const;
  [[nodiscard]] static std::vector<int> MostLikelyComponents(const Matrix& responsibilities);
  [[nodiscard]] static Matrix MeansOf(const std::vector<Component>& components);

  GaussianMixtureOptions options_;
  std::unique_ptr<CentroidInitializer> initializer_;
  MetricPtr metric_;
  FixedAssignments fixed_assignments_;
};

}
