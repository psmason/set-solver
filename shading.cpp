#include <shading.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

namespace setsolver {

  namespace {
    using namespace cv;
    using namespace std;

    using Contour = vector<Point>;
    using Contours = vector<Contour>;
    
    string shadingToString(const Shading color) {
      switch (color) {
      case Shading::SOLID:
        return "SOLID";
      case Shading::STRIPED:
        return "STRIPED";
      case Shading::OPEN:
        return "OPEN";
      default:
        throw std::runtime_error("unknown color");
      }
    }

    Shading distanceToShading(const double distance) {
      if (distance < 10.0) {
        return Shading::OPEN;
      }
      else if (distance > 100.0) {
        return Shading::SOLID;
      }
      else {
        return Shading::STRIPED;
      }
    }

    Vec3b getBackgroundColor(const Mat& card,
                             const Contour& contour) {
      Contours contours;
      contours.push_back(contour);
      Mat contourMask(card.size(), CV_8U);
      contourMask = 0;

      // draw contours so that we are confidence we exclude all colors
      drawContours(contourMask, contours, 0, 255, 16);       // includes border colors and exterior
      drawContours(contourMask, contours, 0, 0, 8);          // excludes border colors
      drawContours(contourMask, contours, 0, 0, CV_FILLED);  // excludes interior
      
      const auto background = mean(card, contourMask);
      return Vec3b(background[0],
                   background[1],
                   background[2]);
    }
  }

  std::ostream& operator<< (std::ostream& stream, const Shading& shading)
  {
    stream << shadingToString(shading);
    return stream;
  }

  Shading computeShading(const Mat& card,
                         const std::vector<cv::Point>& contour)  {
    Contours contours;
    contours.push_back(contour);
    Mat contourMask(card.size(), CV_8U);
    contourMask = 0;
    drawContours(contourMask, contours, 0, 255, CV_FILLED);
    drawContours(contourMask, contours, 0, 0, 16);

    Mat interiorCopy;
    card.copyTo(interiorCopy, contourMask);

    const auto backgroundColor = getBackgroundColor(card, contour);
    Mat maskedDistance(card.size(),  DataType<float>::type);
    maskedDistance = 0.0;
    for(int j = 0; j < maskedDistance.rows; j++) {
      for(int i = 0; i < maskedDistance.cols; i++) {
        auto v = interiorCopy.at<Vec3b>(j, i);
        maskedDistance.at<float>(j, i) = norm(backgroundColor, v);
      }
    }
    
    const auto backgroundDistance = mean(maskedDistance, contourMask)[0];
    return distanceToShading(backgroundDistance);
  }   
}
