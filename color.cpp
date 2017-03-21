#include <color.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>
#include <array>

namespace setsolver {

  namespace {
    using namespace std;
    using namespace cv;

    using Contour = vector<Point>;
    using Contours = vector<Contour>;
    
    Color getColorFromDensities(const std::array<float, 3>& densities) {
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

    Vec3b getBackgroundColor(const Mat& card,
                             const Contour& contour) {
      Contours contours;
      contours.push_back(contour);
      Mat contourMask(card.size(), CV_8U);
      contourMask = 0;
      drawContours(contourMask, contours, 0, 255, 8);

      const auto background = mean(card, contourMask);
      return Vec3b(background[0],
                   background[1],
                   background[2]);
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

    std::array<float, 3> computeRGPDensities(const Mat& filtered,
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
                             
  }
    
  Color computeColor(const Mat& card,
                     const Mat& mask,
                     const std::vector<cv::Point>& contour)
  {
    using namespace cv;

    Mat img;
    card.copyTo(img, mask);
    imshow("masked", img);
    
    const auto backgroundColor = getBackgroundColor(card, contour);
    auto filtered = filterBackgroundColor(card, mask, backgroundColor);
    imshow("filtered", filtered);

    const auto densities = computeRGPDensities(filtered, mask);
    return getColorFromDensities(densities);
  }
}
