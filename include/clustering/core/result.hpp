#pragma once

#include <expected>
#include <string>
#include <utility>

namespace clustering {

struct Error {
  std::string message;
};

template <typename Value>
using Result = std::expected<Value, Error>;

[[nodiscard]] inline std::unexpected<Error> Failure(std::string message) {
  return std::unexpected<Error>(Error{std::move(message)});
}

}
