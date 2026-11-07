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
