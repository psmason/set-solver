#include <shading.h>

#include <utils.h>
#include <types.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

namespace setsolver {

  namespace {
    using namespace cv;
    using namespace std;
    
    string shadingToString(const Shading color) {
      switch (color) {
      case Shading::SOLID:
        return "SOLID";
      case Shading::STRIPED:
        return "STRIPED";
      case Shading::OPEN:
        return "OPEN";
      default:
        throw runtime_error("unknown color");
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

  }

  ostream& operator<< (ostream& stream, const Shading& shading)
  {
    stream << shadingToString(shading);
    return stream;
  }

  Shading computeShading(const Mat& card,
                         const vector<Point>& contour)  {
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
