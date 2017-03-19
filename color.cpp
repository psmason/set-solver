#include <color.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>

namespace setsolver {

  namespace {
    using namespace cv;
    
    Color getColorFromHue(const double hue) {
      // hue is in the range of 0 to 179.
      // let's give each color a 60 degree area.
      if (hue < 40.0 || hue > 160) {
        return Color::RED;
      } else if (hue > 100) {
        return Color::PURPLE;
      } else {
        return Color::GREEN;
      }  
    }
  }
    
  Color computeColor(const Mat& card, const Mat& mask)
  {
    using namespace cv;

    Mat blurred;
    GaussianBlur(card, blurred, Size(21, 21), 0, 0);

    Mat hsv;
    cvtColor(blurred, hsv, CV_BGR2HSV);
    std::vector<Mat> channels;
    split(hsv, channels);

    Mat saturationMask = mask & (channels[1] > 10);
    std::cout << "contour mask:" << mean(hsv, mask) << std::endl;
    std::cout << "saturation mask:" << mean(hsv, saturationMask) << std::endl;

    Mat saturationCopy;
    card.copyTo(saturationCopy, saturationMask);
    imshow("saturation mask", saturationCopy);

    return Color::GREEN;
  }
}
