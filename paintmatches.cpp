#include <paintmatches.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace setsolver {

  namespace {
    using namespace cv;
    
    std::array<Scalar, 6> COLOR_WHEEL = {
      Scalar(0, 0, 255),   // red
      Scalar(0, 255, 255), // yellow
      Scalar(0, 255, 0),   // green
      Scalar(255, 255, 0), // cyan
      Scalar(255, 0, 0),   // blue
      Scalar(255, 0, 255), // magenta
    };
    const int COLOR_WHEEL_SIZE = COLOR_WHEEL.size();

    const int CIRCLE_WIDTH = 5;
    const double OVERLAY_ALPHA = 0.3;

  } // close anonymous

  void paintMatches(cv::Mat& canvas,
                    const Matches& matches,
                    const Cards& cards)
  {    
    for (size_t i=0; i<matches.size(); ++i) {
      const auto& match = matches[i];
      assert(3 == match.size());       
      for (size_t j=0; j<match.size(); ++j) {
        Mat overlay;
        canvas.copyTo(overlay);
        const auto card = cards[match[j]];
        const auto center = minAreaRect(card).center;
        circle(overlay,
               center,
               (i+1)*CIRCLE_WIDTH,
               COLOR_WHEEL[i%COLOR_WHEEL_SIZE],
               CIRCLE_WIDTH-1);
        addWeighted(overlay,
                    OVERLAY_ALPHA,
                    canvas,
                    1-OVERLAY_ALPHA,
                    0,
                    canvas);
      }
    }
  }
  
}
