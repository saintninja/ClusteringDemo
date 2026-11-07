#include "clustering/io/label_encoder.hpp"

#include <map>

namespace clustering {

std::vector<int> EncodeLabels(std::span<const std::string> label_texts) {
  std::map<std::string, int> code_of_text;
  for (const std::string& text : label_texts) {
    code_of_text.try_emplace(text, 0);
  }
  int next_code = 0;
  for (auto& [text, code] : code_of_text) {
    code = next_code++;
  }
  std::vector<int> codes;
  codes.reserve(label_texts.size());
  for (const std::string& text : label_texts) {
    codes.push_back(code_of_text.at(text));
  }
  return codes;
}

}
