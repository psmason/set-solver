#include <attributes.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <color.h>
#include <symbol.h>

#include <array>
#include <vector>
#include <iostream>

namespace setsolver {  
  namespace {
    using namespace cv;
    using namespace std;
    using Contour = vector<Point>;
    using Contours = vector<Contour>;
    
    Mat computeFeatureMask(const Mat& card,
                           const Contours& contours) {
      using namespace cv;
      Mat mask(card.size(), CV_8U);
      mask = 0;
      drawContours(mask, contours, 0, 255, CV_FILLED);
  
      return mask;
    }

    Contours computeFeatureContours(const Mat& card) {
      using namespace cv;
      Mat gray;
      cvtColor(card, gray, CV_BGR2GRAY);

      Mat blurred;
      GaussianBlur(gray, blurred, Size(9, 9), 0, 0);
  
      Mat _img;
      double otsu_thresh_val = cv::threshold(blurred,
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

  void orderPoints(std::array<Point2f, 4>& points) {
    // assumes points are (column, row) coordinates.
    // order clockwise from bottom left corner.

    // sort by column
    std::sort(points.begin(), points.end(),
              [](const Point2f& lhs, const Point2f& rhs) {
                return lhs.x < rhs.x;
              });

    // sort by row
    std::sort(points.begin(), points.begin()+2,
              [](const Point2f& lhs, const Point2f& rhs) {
                return lhs.y > rhs.y;
              });
    std::sort(points.begin()+2, points.end(),
              [](const Point2f& lhs, const Point2f& rhs) {
                return lhs.y < rhs.y;
              });
  }

  Mat correctCard(const Mat& maskedFrame, const Contour& cardContour) {
    // correcting for rotations
    const auto rect = minAreaRect(cardContour);
      
    // normalize the card accounting for rotation
    std::array<Point2f, 4> warpedPts;
    rect.points(warpedPts.begin());      
    orderPoints(warpedPts);

    std::array<Point2f, 4> correctedPts;
    correctedPts[0] = cv::Point2f(0, 150);
    correctedPts[1] = cv::Point2f(0, 0);
    correctedPts[2] = cv::Point2f(250, 0);
    correctedPts[3] = cv::Point2f(250, 150);

    cv::Mat transform = getPerspectiveTransform(warpedPts.begin(),     
                                                correctedPts.begin());
    cv::Mat warpedImg;
    cv::warpPerspective(maskedFrame, warpedImg, transform, Size(250,150));

    return warpedImg;
  }

  FeatureSet getCardFeatures(const Mat& frame, const Cards& cards) {

    // black out all pixels not within the cards
    auto copy = frame.clone();
    copy = 0;
    for (int i=0; i<cards.size(); ++i) {
      drawContours(copy, cards, i, CV_RGB(255, 255, 255), CV_FILLED);
    }
    Mat maskedFrame;
    bitwise_and(frame, copy, maskedFrame);

    imshow("masked", maskedFrame);
    waitKey(50);
    
    FeatureSet featureSet;
    for (const auto& cardContour: cards) {

      const auto corrected = correctCard(maskedFrame, cardContour);
      imshow("corrected card", corrected);
      auto contours = computeFeatureContours(corrected);

      const auto featureMask = computeFeatureMask(corrected, contours);
      const auto color = computeColor(corrected, featureMask, contours.front());
      const auto symbol = computeSymbol(corrected, contours.front());
      const auto shading = computeShading(corrected, featureMask, contours.front());

      std::cout << CardFeatures{color, symbol, shading, contours.size()} << std::endl;
      waitKey();
      
      featureSet.push_back(CardFeatures{color, symbol, shading, contours.size()});
    }
  
    return featureSet;
  }
  
}
