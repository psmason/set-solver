#pragma once

#include <types.h>

#include <opencv2/core.hpp>

namespace setsolver {
  
  cv::Vec3b getBackgroundColor(const cv::Mat& card,
                               const std::vector<cv::Point>& contour);

  cv::Mat correctCard(const cv::Mat& frame, const Contour& cardContour);

}
