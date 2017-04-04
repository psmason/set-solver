#include <color.h>

#include <types.h>
#include <utils.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>
#include <array>

namespace setsolver {

  namespace {
    using namespace std;
    using namespace cv;
    
    Color getColorFromDensities(const array<float, 3>& densities) {
      // assumes red, green, purple values, respectively in the array.
      const auto red = densities[0];
      const auto green = densities[1];
      const auto purple = densities[2];

      if (red > green) {
        if (red > purple) {
          return Color::RED;
        }
        else {          
          return Color::PURPLE;
        }
      }
      else {
        if (green > purple) {
          return Color::GREEN;
        }
        else {
          return Color::PURPLE;
        }
      }
    }
    
    Mat filterBackgroundColor(const Mat& card,
                              const Mat& mask,
                              const Vec3b& backgroundColor) {
      Mat filtered;
      card.copyTo(filtered, mask);
      for(int j = 0; j < filtered.rows; j++) {
        for(int i = 0; i < filtered.cols; i++) {
          auto v = filtered.at<Vec3b>(j, i);
          auto distance = norm(backgroundColor, v);
          if (distance < 50.0) {
            filtered.at<Vec3b>(j, i) = Vec3b(0, 0, 0);
          }
        }
      }
      return filtered;
    }

    array<float, 3> computeRGPDensities(const Mat& filtered,
                                        const Mat& mask) {
      // return the red/green/purple intensities from a histogram
      Mat hsv;
      cvtColor(filtered, hsv, CV_BGR2HSV);

      vector<Mat> channels;
      split(hsv, channels);

      /// Establish the number of bins
      int histSize = 256;

      /// Set the ranges ( for B,G,R) )
      float range[] = {0, 256};
      const float* histRange = {range};
      bool uniform = true;
      bool accumulate = false;

      Mat h_hist;

      /// Compute the histograms:
      calcHist(&channels[0],
               1,
               0,
               mask,
               h_hist,
               1,
               &histSize,
               &histRange,
               uniform,
               accumulate);

      /// Draw for each channel
      array<float, 3> densities = {0.0, 0.0, 0.0}; 
      for(int i = 0; i<histSize; i++) {
        if (0 == i) {
          // 0 probably represents a mask rather than some red value
          continue;
        }

        const auto v = h_hist.at<float>(i);
        if (i<30 || i>150) {
          // RED
          densities[0] += v;
        }    
        else if (i >= 30 && i < 90) {
          // GREEN
          densities[1] += v;
        }
        else {
          // PURPLE
          densities[2] += v;
        }
      }

      return densities;
    }

    string colorToString(const Color& color) {
      switch (color) {
      case Color::RED:
        return "RED";
      case Color::PURPLE:
        return "PURPLE";
      case Color::GREEN:
        return "GREEN";
      default:
        throw runtime_error("unknown color");
      }
    }
                             
  }

  Color parseColor(const std::string& s) {
    if ("RED" == s) {
      return Color::RED;
    }
    else if ("PURPLE" == s) {
      return Color::PURPLE;
    }
    else if ("GREEN" == s) {
      return Color::GREEN;
    }
    assert(!"failed to parse color");
  }

  ostream& operator<< (ostream& stream, const Color& color)
  {
    stream << colorToString(color);
    return stream;
  }
    
  Color computeColor(const Mat& card,
                     const Mat& mask,
                     const vector<Point>& contour)
  {
    const auto backgroundColor = getBackgroundColor(card, contour);
    auto filtered = filterBackgroundColor(card, mask, backgroundColor);
    const auto densities = computeRGPDensities(filtered, mask);
    return getColorFromDensities(densities);
  }
}
