#pragma once

#include <opencv2/core.hpp>

namespace setsolver {
  
  cv::Vec3b getBackgroundColor(const cv::Mat& card,
                               const std::vector<cv::Point>& contour);

}
