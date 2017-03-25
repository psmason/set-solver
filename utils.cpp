#include <utils.h>

#include <types.h>

#include <opencv2/imgproc.hpp>

#include <vector>

namespace setsolver {

  namespace {
    using namespace cv;
  }
  
  Vec3b getBackgroundColor(const Mat& card,
                             const Contour& contour) {
      Contours contours;
      contours.push_back(contour);
      Mat contourMask(card.size(), CV_8U);
      contourMask = 0;

      // draw contours so that we exclude all colors
      drawContours(contourMask, contours, 0, 255, 16);       // includes border colors and exterior
      drawContours(contourMask, contours, 0, 0, 8);          // excludes border colors
      drawContours(contourMask, contours, 0, 0, CV_FILLED);  // excludes interior
      
      const auto background = mean(card, contourMask);
      return Vec3b(background[0],
                   background[1],
                   background[2]);
  }
  
}
