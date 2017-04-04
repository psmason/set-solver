#include <utils.h>

#include <types.h>

#include <opencv2/imgproc.hpp>

#include <array>
#include <vector>

namespace setsolver {

  namespace {
    using namespace cv;

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

      // if ordered point height is greater than width,
      // a rotation will put the card into a normalized landspace shape.
      const auto height = points[0].y - points[1].y;
      const auto width  = points[2].x - points[1].x;
      if (height > width) {
        std::rotate(points.begin(), points.begin()+1, points.end());
      }
    }
    
  }
  
  Vec3b getBackgroundColor(const Mat& card,
                             const Contour& contour) {
      Contours contours;
      contours.push_back(contour);
      Mat contourMask(card.size(), CV_8U);
      contourMask = 0;

      // draw contours so that we exclude all colors
      drawContours(contourMask, contours, 0, 255, 16);       // includes border colors and exterior
      drawContours(contourMask, contours, 0, 0, 8);          // excludes border colors
      drawContours(contourMask, contours, 0, 0, CV_FILLED);  // excludes interior
      
      const auto background = mean(card, contourMask);
      return Vec3b(background[0],
                   background[1],
                   background[2]);
  }

  cv::Mat correctCard(const Mat& frame, const Contour& cardContour) {
    // correcting for rotations
    const auto rect = minAreaRect(cardContour);
      
    // normalize the card accounting for rotation
    std::array<Point2f, 4> warpedPts;
    rect.points(warpedPts.begin());      
    orderPoints(warpedPts);

    std::array<Point2f, 4> correctedPts;
    correctedPts[0] = Point2f(0, 150);
    correctedPts[1] = Point2f(0, 0);
    correctedPts[2] = Point2f(250, 0);
    correctedPts[3] = Point2f(250, 150);

    Mat transform = getPerspectiveTransform(warpedPts.begin(),     
                                                correctedPts.begin());
    Mat warpedImg;
    warpPerspective(frame, warpedImg, transform, Size(250,150));

    return warpedImg;
  }
  
}
