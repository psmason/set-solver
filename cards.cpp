#include <cards.h>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <set>

namespace {
  using namespace cv;
  using namespace std;
  
  double OTSU_MULTIPLIER = 0.35;

  // helper function:
  // finds a cosine of angle between vectors
  // from pt0->pt1 and from pt0->pt2
  double angle(const Point pt1, const Point pt2, const Point pt0) {
    const auto dx1 = pt1.x - pt0.x;
    const auto dy1 = pt1.y - pt0.y;
    const auto dx2 = pt2.x - pt0.x;
    const auto dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
  }
    
} // close anonymous

namespace setsolver {

  Cards findCards(const Mat& image) {


    Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);

    Mat _img;
    double otsu_thresh_val = threshold(gray,
                                           _img,
                                           0,
                                           255,
                                           CV_THRESH_BINARY | CV_THRESH_OTSU);
    
    Mat canny;
    const auto lowerThreshold = otsu_thresh_val*OTSU_MULTIPLIER;
    Canny(gray, canny, lowerThreshold, 2.0*lowerThreshold);

    // dilate canny output to remove potential
    // holes between edge segments
    dilate(canny, canny, Mat(), Point(-1,-1));

    imshow("canny", canny);

    vector<vector<Point>> imgContours;
    findContours(canny, imgContours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    vector<vector<Point>> cardContours;
    for(size_t i = 0; i < imgContours.size(); i++) {
      // approximate contour with accuracy proportional
      // to the contour perimeter

      vector<Point> approx;
      approxPolyDP(Mat(imgContours[i]), approx, arcLength(Mat(imgContours[i]), true)*0.02, true);

      // square contours should have 4 vertices after approximation
      // relatively large area (to filter out noisy contours)
      // and be convex.
      // Note: absolute value of an area is used because
      // area may be positive or negative - in accordance with the
      // contour orientation
      if(approx.size() == 4 &&
          fabs(contourArea(Mat(approx))) > 1000 &&
          isContourConvex(Mat(approx))) {
        double maxCosine = 0;

        for( int j = 2; j < 5; j++ ) {
          // find the maximum cosine of the angle between joint edges
          double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
          maxCosine = MAX(maxCosine, cosine);
        }

        // if cosines of all angles are small
        // (all angles are ~90 degree) then write quandrange
        // vertices to resultant sequence
        if (maxCosine >= 0.3) {
          continue;
        }

        // it's tempting to checking width/height ratios of the contours
        // https://en.wikipedia.org/wiki/Standard_52-card_deck
        //
        // however you need to allow for perspective changes.
        // https://en.wikipedia.org/wiki/Keystone_effect
        // and i don't believe you can back out the true dimension
        // since the camera will have no depth information.        
        //
        // so no more tests...
        cardContours.push_back(approx);
      }
    }
    return cardContours;  
  }
}
