#pragma once

#include <opencv2/core.hpp>

#include <ostream>

namespace cv {
  class Mat;
}

namespace setsolver {
  
  enum class Symbol {
    DIAMOND,
    SQUIGGLE,
    OVAL,
  };
  std::ostream& operator<< (std::ostream& stream, const Symbol& symbol);

  Symbol computeSymbol(const cv::Mat& card, const std::vector<cv::Point>& contour);

}
