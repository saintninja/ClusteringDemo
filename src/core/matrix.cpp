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
