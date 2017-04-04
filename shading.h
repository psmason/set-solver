#pragma once

#include <opencv2/core.hpp>
#include <vector>

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
  Shading parseShading(const std::string& s);
  
  Shading computeShading(const cv::Mat& card,
                         const std::vector<cv::Point>& contour);

}
