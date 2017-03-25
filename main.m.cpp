#include <cards.h>
#include <attributes.h>
#include <solver.h>
#include <paintmatches.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

namespace {
  cv::Scalar CARD_HIGHLIGHT(124, 252, 0);
}

int main()
{
  using namespace cv;
  
  VideoCapture cap(0); 
  if(!cap.isOpened()) { 
    exit(1);
  }

  while (true) {
    Mat frame;
    cap >> frame; // get a new frame from camera

    const auto cards = setsolver::findCards(frame);
    std::cout << "cards found: " << cards.size() << std::endl;
    Mat drawing = frame.clone(); 
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
      std::cout << "features" << std::endl;
      for (const auto& feature: featureSet) {
        std::cout << feature << std::endl;
      }
      const auto matches = setsolver::findMatches(featureSet);
      setsolver::paintMatches(drawing, matches, cards);                
    }
    imshow("cards", drawing);
    waitKey(50);
  }

  return 0;
}
