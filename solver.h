#pragma once

#include <attributes.h>

#include <vector>
#include <array>

namespace setsolver {

  using Match = std::array<size_t, 3>;
  using Matches = std::vector<Match>;

  Matches findMatches(const FeatureSet& featureSet);

}
