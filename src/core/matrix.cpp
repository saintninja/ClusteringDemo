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
#include "clustering/core/matrix.hpp"

#include <stdexcept>

namespace clustering {

void Matrix::AppendRow(Point row) {
  if (row_count_ == 0) {
    column_count_ = row.size();
  } else if (row.size() != column_count_) {
    throw std::invalid_argument("appended row width differs from matrix width");
  }
  values_.insert(values_.end(), row.begin(), row.end());
  ++row_count_;
}

}
