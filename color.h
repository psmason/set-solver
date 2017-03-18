#pragma once

namespace cv {
  class Mat;
}

namespace setsolver {
  
  enum class Color {
    RED,
    GREEN,
    PURPLE,
  };

  Color computeColor(const cv::Mat& card, const cv::Mat& mask);

}
