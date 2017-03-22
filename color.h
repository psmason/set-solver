#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <ostream>

namespace cv {
  class Mat;
}

namespace setsolver {
  
  enum class Color {
    RED,
    GREEN,
    PURPLE,
  };

  Color computeColor(const cv::Mat& card,
                     const cv::Mat& mask,
                     const std::vector<cv::Point>& contour);

}
