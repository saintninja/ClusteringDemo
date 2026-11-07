#pragma once

#include <cmath>
#include <exception>
#include <format>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "clustering/core/result.hpp"

namespace clustering::tests {

struct TestFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline void Expect(bool condition, const std::string& description) {
  if (!condition) {
    throw TestFailure(description);
  }
}

inline void ExpectNear(double actual, double expected, double tolerance,
                       const std::string& description) {
  if (!(std::abs(actual - expected) <= tolerance)) {
    throw TestFailure(std::format("{}: expected {} but got {}", description, expected, actual));
  }
}

template <typename Value>
Value ValueOrFail(Result<Value> result, const std::string& description) {
  if (!result) {
    throw TestFailure(description + ": " + result.error().message);
  }
  return std::move(*result);
}

template <typename Action>
void ExpectThrows(Action&& action, const std::string& description) {
  try {
    std::forward<Action>(action)();
  } catch (const std::exception&) {
    return;
  }
  throw TestFailure(description + ": expected an exception");
}

class TestSuite {
 public:
  void Add(std::string name, std::function<void()> body) {
    cases_.push_back({std::move(name), std::move(body)});
  }

  [[nodiscard]] int Run() const {
    int failed = 0;
    for (const auto& test : cases_) {
      try {
        test.body();
        std::cout << "[ PASS ] " << test.name << '\n';
      } catch (const std::exception& error) {
        ++failed;
        std::cout << "[ FAIL ] " << test.name << ": " << error.what() << '\n';
      }
    }
    std::cout << cases_.size() - static_cast<std::size_t>(failed) << '/' << cases_.size()
              << " tests passed\n";
    return failed == 0 ? 0 : 1;
  }

 private:
  struct Case {
    std::string name;
    std::function<void()> body;
  };
  std::vector<Case> cases_;
};

void AddLinearAlgebraTests(TestSuite& suite);
void AddDistanceTests(TestSuite& suite);
void AddQualityMetricTests(TestSuite& suite);
void AddInitializerTests(TestSuite& suite);
void AddAlgorithmTests(TestSuite& suite);
void AddInputOutputTests(TestSuite& suite);
void AddPipelineTests(TestSuite& suite);

}
