#pragma once
 
#include "opencv2/core.hpp"
#include <vector>

namespace setsolver {  

  using Card = std::vector<cv::Point>;
  using Cards = std::vector<Card>;

  // finds all cards in the image
  Cards findCards(const cv::Mat& image);

}
