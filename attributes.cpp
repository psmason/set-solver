#include <attributes.h>

#include <color.h>
#include <symbol.h>
#include <types.h>
#include <utils.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <array>
#include <vector>
#include <algorithm>

namespace setsolver {  
  namespace {
    using namespace cv;
    using namespace std;
    
    Mat computeFeatureMask(const Mat& card,
                           const Contours& contours) {
      Mat mask(card.size(), CV_8U);
      mask = 0;
      drawContours(mask, contours, 0, 255, CV_FILLED);
  
      return mask;
    }

    Contours computeFeatureContours(const Mat& card) {
      Mat gray;
      cvtColor(card, gray, CV_BGR2GRAY);

      Mat blurred;
      GaussianBlur(gray, blurred, Size(9, 9), 0, 0);
  
      Mat _img;
      double otsu_thresh_val = threshold(blurred,
                                             _img,
                                             0,
                                             255,
                                             CV_THRESH_BINARY | CV_THRESH_OTSU);
  
      Mat canny;
      Canny(blurred, canny, otsu_thresh_val*0.25, otsu_thresh_val*0.5);

      // dilate canny output to remove potential
      // holes between edge segments
      dilate(canny, canny, Mat(), Point(-1,-1));

      Contours contours;
      findContours(canny,
                   contours,
                   CV_RETR_EXTERNAL,
                   CV_CHAIN_APPROX_SIMPLE);

      // filtering out contours resembling set shapes
      auto itr = std::partition(contours.begin(), contours.end(),
                                [](const Contour& contour) {
                                  // http://docs.opencv.org/3.2.0/d1/d32/tutorial_py_contour_properties.html
                                  const auto area = contourArea(contour);
                                  if (area < 1500.0) {
                                    return false;
                                  }
                                  
                                  Contour hull;
                                  convexHull(Mat(contour), hull);
                                  const auto solidity = area/contourArea(hull);
                                  return solidity > 0.8;
                                });
      contours.erase(itr, contours.end());
      
      return contours;
    }
    
  }

  std::ostream& operator<< (std::ostream& stream, const CardFeatures& cardFeatures) {
    stream << "color=" << cardFeatures.color
           << "\tsymbol=" << cardFeatures.symbol
           << "\tshading=" << cardFeatures.shading
           << "\tnumber=" << cardFeatures.number;
    return stream;
  }

  FeatureSet getCardFeatures(const Mat& frame,
                             const Cards& cards,
                             const bool debug) {

    /*
    // black out all pixels not within the cards
    auto copy = frame.clone();
    copy = 0;
    for (size_t i=0; i<cards.size(); ++i) {
      drawContours(copy, cards, i, CV_RGB(255, 255, 255), CV_FILLED);
    }
    
    Mat maskedFrame;
    bitwise_and(frame, copy, maskedFrame);
    */
    
    FeatureSet featureSet;
    for (const auto& cardContour: cards) {

      const auto corrected = correctCard(frame, cardContour);
      if (debug) {
        imshow("find attributes - corrected card", corrected);
      }
      auto contours = computeFeatureContours(corrected);

      if (0 == contours.size() || 3 < contours.size()) {
        // invalid read on the webcame frame.
        // hopefully next iteration works.
        featureSet.clear();
        return featureSet;
      }

      const auto featureMask = computeFeatureMask(corrected, contours);
      const auto color = computeColor(corrected, featureMask, contours.front());
      const auto symbol = computeSymbol(contours.front());
      const auto shading = computeShading(corrected, contours.front());
      featureSet.push_back(CardFeatures{color, symbol, shading, contours.size()});
    }
  
    return featureSet;
  }
  
}
