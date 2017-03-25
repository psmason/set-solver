#include <symbol.h>

#include <opencv2/imgproc.hpp>

#include <vector>

namespace setsolver {

  namespace {
    using namespace cv;
    using namespace std;
    
    std::string symbolToString(const Symbol symbol) {
      switch (symbol) {
      case Symbol::DIAMOND:
        return "DIAMOND";
      case Symbol::SQUIGGLE:
        return "SQUIGGLE";
      case Symbol::OVAL:
        return "OVAL";
      }
    }

  }

  std::ostream& operator<< (std::ostream& stream, const Symbol& symbol)
  {
    stream << symbolToString(symbol);
    return stream;
  }

  Symbol computeSymbol(const cv::Mat& card, const vector<cv::Point>& contour)
  {
    vector<Point> approx;
    approxPolyDP(Mat(contour), approx, arcLength(Mat(contour), true)*0.02, true);

    if (!isContourConvex(approx)) {
      return Symbol::SQUIGGLE;
    }

    // identify ratio of contour area to min bounding rectangle area.
    // oval is assumed to have a higher ratio.
    const auto rec = minAreaRect(approx);
    const auto area = contourArea(approx, false);
    const auto ratio = area / rec.size.area();
  
    if (ratio > 0.8) {
      return Symbol::OVAL;    
    }

    return Symbol::DIAMOND;
  }
  
}
