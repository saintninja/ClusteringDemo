#include <algorithm>
#include <iostream>
#include <string_view>
#include <vector>

#include "clustering/app/command_line.hpp"
#include "clustering/app/pipeline.hpp"
#include "clustering/app/report_formatter.hpp"

int main(int argument_count, char** argument_values) {
  const std::vector<std::string_view> arguments(argument_values + 1,
                                                argument_values + argument_count);
  const bool help_requested =
      arguments.empty() || std::ranges::find(arguments, "--help") != arguments.end();
  if (help_requested) {
    std::cout << clustering::UsageText();
    return 0;
  }

  const auto configuration = clustering::ParseCommandLine(arguments);
  if (!configuration) {
    std::cerr << "Error: " << configuration.error().message << "\n\n" << clustering::UsageText();
    return 2;
  }

  const auto outcome = clustering::RunPipeline(*configuration);
  if (!outcome) {
    std::cerr << "Error: " << outcome.error().message << '\n';
    return 1;
  }

  std::cout << clustering::FormatOutcome(*outcome) << "Result written to "
            << configuration->output_path.string() << '\n';
  return 0;
}
