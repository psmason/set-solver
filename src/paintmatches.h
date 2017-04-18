#pragma once

#include <solver.h>
#include <cards.h>

namespace cv {
  class Mat;
}

namespace setsolver {

  void paintMatches(cv::Mat& canvas,
                    const Matches& matches,
                    const Cards& cards);
  
}
