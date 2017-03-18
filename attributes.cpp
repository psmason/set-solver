#include <attributes.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <color.h>

#include <vector>
#include <iostream>

namespace setsolver {  
  namespace {
    using namespace cv;
    using namespace std;
    using Contours = vector<vector<Point>>;
    
    Mat computeFeatureMask(const Mat& card,
                           const vector<vector<Point>>& contours) {
      using namespace cv;
      Mat mask(card.size(), CV_8U);
      mask = 0;
      drawContours(mask, contours, 0, 255,-1);
  
      return mask;
    }

    Contours computeCardContours(const Mat& card) {
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

      return contours;
    }

    std::string colorToString(const Color color) {
      switch (color) {
      case Color::RED:
        return "RED";
      case Color::PURPLE:
        return "PURPLE";
      case Color::GREEN:
        return "GREEN";
      }
    }
  }

  std::ostream& operator<< (std::ostream& stream, const CardFeatures& cardFeatures) {
    stream << "color=" << colorToString(cardFeatures.color);
    return stream;
  }

  FeatureSet getCardFeatures(const Mat& frame, const Cards& cards) {

    // black out all pixels not within the cards
    auto copy = frame.clone();
    copy = 0;

    for (int i=0; i<cards.size(); ++i) {
      drawContours(copy, cards, i, CV_RGB(255, 255, 255), CV_FILLED);
    }

    Mat masked;
    bitwise_and(frame, copy, masked);
    //auto masked = frame & copy;

    imshow("masked", masked);
    //imshow("card region", masked);
    waitKey(50);
    
    FeatureSet featureSet;
    for (const auto& cardContour: cards) {

      const auto rect = minAreaRect(cardContour);
      /*
       cv::Mat rot_mat = cv::getRotationMatrix2D(rect.center, rect.angle, 1);

       cv::Mat rotated;
       auto img = masked.clone();
       cv::warpAffine(img, rotated, rot_mat, img.size(), cv::INTER_CUBIC);

       //imshow("bounded", masked(rect.boundingRect()));
       imshow("transformed", rotated);
       waitKey();
      */
      
      // normalize the card accounting for rotation
      cv::Point2f pts[4];
      rect.points(pts);

      /*
      std::cout << "rect points" << std::endl;
      std::cout << pts[0] << std::endl;
      std::cout << pts[1] << std::endl;
      std::cout << pts[2] << std::endl;
      std::cout << pts[3] << std::endl;
      */
      
      cv::Point2f warpedPts[4];
      warpedPts[0] = cv::Point2f(0,150);
      warpedPts[1] = cv::Point2f(0, 0);
      warpedPts[2] = cv::Point2f(250, 0);
      warpedPts[3] = cv::Point2f(250, 150);

      cv::Mat transform = getPerspectiveTransform(pts,     
                                                  warpedPts);
      cv::Mat warpedImg;
      cv::warpPerspective(masked, warpedImg, transform, Size(250,150));

      imshow("bounded", masked(rect.boundingRect()));
      imshow("transformed", warpedImg);
      waitKey();


      // cv::Size size = rect.size();
      // Mat dst(size, CV_8U);
      
      /*
      Mat card = masked(rect.boundingRect());

      cv::Point2f ptCp(card.cols*0.5, card.rows*0.5);
      cv::Mat M = cv::getRotationMatrix2D(ptCp, rect.angle, 1.0);

      Mat dst;
      double w = rect.boundingRect().size().width;
      double h = rect.boundingRect().size().height;
      double d = sqrt(w*w + h*h);
      cv::warpAffine(card, dst, M, Size(d,d), cv::INTER_CUBIC);

      imshow("card rect", card);
      imshow("card rotated", dst);
      waitKey();
      */

      /*
      Mat card = masked(boundingRect(cardContour));

      imshow("card", card);
      waitKey(); 
      */

      /*
      const auto contours = computeCardContours(card);
      const auto mask = computeFeatureMask(card, contours);     
      const auto color = computeColor(card, mask);
      */
      // featureSet.push_back(CardFeatures{color});
    }
  
    return featureSet;
  }
  
}
