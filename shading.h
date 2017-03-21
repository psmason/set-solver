#pragma once

#include <ostream>

namespace cv {
  class Mat;
}

namespace setsolver {

  enum class Shading {
    SOLID,
    STRIPED,
    OPEN,
  };
  std::ostream& operator<< (std::ostream& stream, const Shading& shading);

  Shading computeShading(const cv::Mat& card, const cv::Mat& mask);

}
