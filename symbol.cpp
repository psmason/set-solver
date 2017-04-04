#include <symbol.h>

#include <opencv2/imgproc.hpp>

#include <vector>

namespace setsolver {

  namespace {
    using namespace cv;
    using namespace std;
    
    string symbolToString(const Symbol symbol) {
      switch (symbol) {
      case Symbol::DIAMOND:
        return "DIAMOND";
      case Symbol::SQUIGGLE:
        return "SQUIGGLE";
      case Symbol::OVAL:
        return "OVAL";
      default:
        throw runtime_error("unknown symbol");
      }
    }

  }

  ostream& operator<< (ostream& stream, const Symbol& symbol)
  {
    stream << symbolToString(symbol);
    return stream;
  }

  Symbol parseSymbol(const std::string& s)
  {
    if ("DIAMOND" == s) {
      return Symbol::DIAMOND;
    }
    else if ("SQUIGGLE" == s) {
      return Symbol::SQUIGGLE;
    }
    else if ("OVAL" == s) {
      return Symbol::OVAL;
    }
    assert(!"failed to parse symbol");
  }

  Symbol computeSymbol(const vector<Point>& contour)
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
