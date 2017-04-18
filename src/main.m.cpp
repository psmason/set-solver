#include <cards.h>
#include <attributes.h>
#include <solver.h>
#include <paintmatches.h>
#include <utils.h>
#include <debug.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <memory>
#include <iostream>
#include <algorithm>

namespace {  
  using namespace cv;
  using namespace setsolver;
  const cv::Scalar CARD_HIGHLIGHT(124, 252, 0);
}

int main(int argc, char* argv[])
{
  CommandLineParser parser(argc, argv,
                           "{help h usage ? |       | print this message   }"
                           "{d debug        | false | Set to debug for single card analysis."
                                                    " Features of the card closest to the center"
                                                    " will be displayed.}"
                           "{c color        |       | known color for debug training dump}"
                           "{s symbol       |       | known symbol for debug training dump}"
                           "{t shading      |       | known shading for debug training dump}"
                           "{n number       |       | known number for debug training dump}"
                           );
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

  auto knownFeatures = setsolver::getKnownFeatures(parser);

  while (true) {
    Mat frame;
    cap >> frame; // get a new frame from camera
    Mat drawing = frame.clone();
    
    auto cards = findCards(frame, debug);

    if (debug) {
      debugCards(drawing, cards, frame, knownFeatures);
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
      
        const auto featureSet = getCardFeatures(frame, cards);
        std::cout << "features (count="
                  << featureSet.size()
                  << ")"
                  << std::endl;
        for (const auto& feature: featureSet) {
          std::cout << feature << std::endl;
        }
        const auto matches = findMatches(featureSet);
        paintMatches(drawing, matches, cards);                
      }
    }
    
    imshow("result", drawing);
    waitKey(50);
  }
}
