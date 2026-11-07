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
