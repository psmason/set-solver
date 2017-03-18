#include <color.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

namespace setsolver {

  namespace {
    using namespace cv;
    
    Color getColorFromHue(const double hue) {
      if (hue < 45.0) {
        return Color::RED;
      } else if (hue > 90) {
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
    GaussianBlur(card, blurred, Size(13, 13), 0, 0);

    Mat hsv;
    cvtColor(blurred, hsv, CV_BGR2HSV);

    imshow("test features", blurred);
    imshow("test mask", mask);
    std::cout << mean(hsv, mask)[0] << std::endl;
    waitKey();
    

    return getColorFromHue(mean(hsv, mask)[0]);
  }
}
