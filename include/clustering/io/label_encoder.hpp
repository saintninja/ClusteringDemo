#pragma once

#include <span>
#include <string>
#include <vector>

namespace clustering {

[[nodiscard]] std::vector<int> EncodeLabels(std::span<const std::string> label_texts);

}
