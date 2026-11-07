#pragma once

#include "clustering/core/matrix.hpp"

namespace clustering {

[[nodiscard]] Matrix Standardize(const Matrix& data);

}
