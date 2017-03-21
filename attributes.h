#pragma once

#include "opencv2/core.hpp"

#include <cards.h>
#include <color.h>
#include <symbol.h>
#include <shading.h>

#include <vector>

namespace setsolver {  

  struct CardFeatures {
    Color color;
    Symbol symbol;
    Shading shading;
  };
  std::ostream& operator<< (std::ostream& stream, const CardFeatures& cardFeatures);

  using FeatureSet = std::vector<CardFeatures>;

  FeatureSet getCardFeatures(const cv::Mat& frame, const Cards& cards);

}
