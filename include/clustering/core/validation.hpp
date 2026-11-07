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

#include <stdexcept>

#include "clustering/core/matrix.hpp"

namespace clustering {

inline void RequireNonEmpty(const Matrix& data) {
  if (data.IsEmpty() || data.ColumnCount() == 0) {
    throw std::invalid_argument("dataset must contain at least one object and one feature");
  }
}

inline void RequireClusterCount(const Matrix& data, std::size_t cluster_count) {
  RequireNonEmpty(data);
  if (cluster_count == 0 || cluster_count > data.RowCount()) {
    throw std::invalid_argument("cluster count must be between 1 and the number of objects");
  }
}

}
