#pragma once

/*
  This is largely a wrapper around the squares.cpp opencv example.
*/

#include "opencv2/core.hpp"
#include <vector>

namespace extractcards {  

  using Card = std::vector<cv::Point>;
  using Cards = std::vector<Card>;

  // finds all cards in the image
  Cards find(const cv::Mat& image);

}
