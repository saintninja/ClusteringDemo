#include "clustering/initialization/initializer_factory.hpp"

#include <stdexcept>
#include <utility>

#include "clustering/initialization/forgy_initializer.hpp"
#include "clustering/initialization/kmeans_plus_plus_initializer.hpp"
#include "clustering/initialization/seeded_initializer.hpp"

namespace clustering {

std::optional<InitializerKind> ParseInitializerKind(std::string_view name) {
  if (name == "forgy") return InitializerKind::kForgy;
  if (name == "kmeans++") return InitializerKind::kKMeansPlusPlus;
  if (name == "seeded" || name == "ssl") return InitializerKind::kSeeded;
  return std::nullopt;
}

std::unique_ptr<CentroidInitializer> MakeInitializer(const InitializerSettings& settings) {
  switch (settings.kind) {
    case InitializerKind::kForgy:
      return std::make_unique<ForgyInitializer>(settings.seed);
    case InitializerKind::kKMeansPlusPlus:
      return std::make_unique<KMeansPlusPlusInitializer>(settings.seed);
    case InitializerKind::kSeeded:
      if (!settings.training) {
        throw std::invalid_argument("seeded initialization requires a training set");
      }
      return std::make_unique<SeededInitializer>(*settings.training, settings.seed);
  }
  std::unreachable();
}

}
