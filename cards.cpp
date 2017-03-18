#include <cards.h>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <set>

namespace {

  int CANNY_THRESHOLD = 50;
  int THRESHOLDS_TO_CHECK = 11;
  double OTSU_MULTIPLIER = 0.35;

  // helper function:
  // finds a cosine of angle between vectors
  // from pt0->pt1 and from pt0->pt2
  double angle(const cv::Point pt1, const cv::Point pt2, const cv::Point pt0) {
    const auto dx1 = pt1.x - pt0.x;
    const auto dy1 = pt1.y - pt0.y;
    const auto dx2 = pt2.x - pt0.x;
    const auto dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
  }
    
} // close anonymous

namespace extractcards {

  Cards find(const cv::Mat& image) {
    using namespace cv;
    using namespace std;

    Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);

    Mat _img;
    double otsu_thresh_val = cv::threshold(gray,
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
    waitKey(50);

    vector<vector<Point>> imgContours;
    vector<Vec4i> hierarchy;
    findContours(canny, imgContours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
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
    
    /*

    vector<vector<Point> > candidates;
    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());

    imshow("pyr", timg);
    waitKey();
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // find squares in every color plane of the image
    for(int c = 0; c < 1; c++) {
      int ch[] = {c, 0};
      mixChannels(&timg, 1, &gray0, 1, ch, 1);

      // try several threshold levels
      for(int l = 0; l < THRESHOLDS_TO_CHECK; l++) {
        // hack: use Canny instead of zero threshold level.
        // Canny helps to catch squares with gradient shading
        if(l == 0) {
          // apply Canny. Take the upper threshold from slider
          // and set the lower to 0 (which forces edges merging)
          Canny(gray0, gray, 0, 30, 3);
          // dilate canny output to remove potential
          // holes between edge segments
          dilate(gray, gray, Mat(), Point(-1,-1));
        }
        else {
          // apply threshold if l!=0:
          //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
          gray = gray0 >= (l+1)*255/THRESHOLDS_TO_CHECK;
        }

        imshow("pyr", gray);
        waitKey();

        // find contours and store them all as a list
        findContours(gray, contours, hierarchy, CV_RETR_CCOMP, CHAIN_APPROX_SIMPLE);

        /// Draw contours
        Mat drawing = gray.clone(); 
        for( int i = 0; i< contours.size(); i++ ) {
          Scalar color(124, 252, 0);
          drawContours(drawing, contours, i, color, 2, 8);
        }

        imshow("pyr", drawing);
        waitKey();
        
        // remove contours which have a parent
        for (int i=contours.size(); i>-1; --i) {
          if (hierarchy[i][3] == -1 ) {
            contours.erase(contours.begin()+i);
          }
        }

        vector<Point> approx;
        // test each contour
        for(size_t i = 0; i < contours.size(); i++) {
          // approximate contour with accuracy proportional
          // to the contour perimeter

          if (contours[i].empty()) {
            continue;
          }
          
          std::cout <<  contours[i].size() << std::endl;
          approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

          // square contours should have 4 vertices after approximation
          // relatively large area (to filter out noisy contours)
          // and be convex.
          // Note: absolute value of an area is used because
          // area may be positive or negative - in accordance with the
          // contour orientation
          if( approx.size() == 4 &&
              fabs(contourArea(Mat(approx))) > 1000 &&
              isContourConvex(Mat(approx)) ) {
            double maxCosine = 0;

            for( int j = 2; j < 5; j++ ) {
              // find the maximum cosine of the angle between joint edges
              double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
              maxCosine = MAX(maxCosine, cosine);
            }

            // if cosines of all angles are small
            // (all angles are ~90 degree) then write quandrange
            // vertices to resultant sequence
            if( maxCosine < 0.3 ) {
              candidates.push_back(approx);
            }
          }
        }
      }
    }

    // removing contours which are contained in another contour
    std::set<int> removals;
    for (int i=0; i<candidates.size(); ++i) {
      for (int j=0; j<candidates.size(); ++j) {
        if (j == i) {
          continue;
        }
        if (pointPolygonTest(candidates[i], candidates[j][0], false) >= 0) {
          removals.insert(j);
        }
      }
    }

    for (auto itr=removals.rbegin(); itr!=removals.rend(); ++itr) {
      candidates.erase(candidates.begin() + *itr);
    }

    return candidates;
    */
  }
}
