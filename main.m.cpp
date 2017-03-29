#include <cards.h>
#include <attributes.h>
#include <solver.h>
#include <paintmatches.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <algorithm>

namespace {
  using namespace cv;  
  cv::Scalar CARD_HIGHLIGHT(124, 252, 0);

  void processForDebug(Mat& drawing,
                       setsolver::Cards& cards,
                       const Mat& frame) {
    const Point2f frameCenter(drawing.cols/2.0, drawing.rows/2.0);

    circle(drawing,
           frameCenter,
           4,
           Scalar(0, 0, 255),
           4);

    // sorting by distance from frame center.
    std::sort(cards.begin(), cards.end(),
              [&drawing, &frameCenter](const setsolver::Card& lhs, const setsolver::Card& rhs) {
                // not exactly efficient, but good enough.
                const auto lCenter = minAreaRect(lhs).center;
                const auto rCenter = minAreaRect(rhs).center;
                return norm(frameCenter - lCenter) < norm(frameCenter - rCenter);
              });

    if (!cards.empty()) {
      cards.erase(cards.begin()+1, cards.end());
      drawContours(drawing,
                   cards,
                   0,
                   CARD_HIGHLIGHT,
                   1);
      const auto features = setsolver::getCardFeatures(frame, cards, true);
      if (!features.empty()) {
        std::cout << "DEBUG: "
                  << features.front()
                  << std::endl;
      }
    }
  }
                       
  
}

int main(int argc, char* argv[])
{
  CommandLineParser parser(argc, argv,
                           "{help h usage ? |       | print this message   }"
                           "{d debug        | false | Set to debug for single card analysis."
                                                    " Features of the card closest to the center"
                                                    " will be displayed.}");
  parser.about("A C++ task attempting to solve the card game Set.");

  if (parser.has("help")) {
    parser.printMessage();
    exit(0);
  }

  if (!parser.check()) {
    parser.printErrors();
    exit(1);
  }
  const bool debug = parser.get<bool>("debug");
  
  VideoCapture cap(0); 
  if(!cap.isOpened()) {
    std::cout << "failed to open camera" << std::endl;
    exit(1);
  }

  while (true) {
    Mat frame;
    cap >> frame; // get a new frame from camera
    Mat drawing = frame.clone();
    
    auto cards = setsolver::findCards(frame, debug);

    if (debug) {
      processForDebug(drawing, cards, frame);
    }
    else {    
      if (cards.size() && 0 == cards.size() % 3) {
        /// Draw contours
        for(size_t i = 0; i< cards.size(); i++) {
          drawContours(drawing,
                       cards,
                       i,
                       CARD_HIGHLIGHT,
                       1);
        }
      
        const auto featureSet = setsolver::getCardFeatures(frame, cards);
        std::cout << "features (count="
                  << featureSet.size()
                  << ")"
                  << std::endl;
        for (const auto& feature: featureSet) {
          std::cout << feature << std::endl;
        }
        const auto matches = setsolver::findMatches(featureSet);
        setsolver::paintMatches(drawing, matches, cards);                
      }
    }
    
    imshow("result", drawing);
    waitKey(50);
  }
}
