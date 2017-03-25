// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image

#include <cards.h>
#include <attributes.h>
#include <solver.h>
#include <paintmatches.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

int main(int argc, char* argv[])
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
    if (cards.size() && 0 == cards.size() % 3) {
      /// Draw contours
      Mat drawing = frame.clone(); 
      for( int i = 0; i< cards.size(); i++ ) {
        drawContours(drawing,
                     cards,
                     i,
                     Scalar(124, 252, 0),
                     1);
      }
      
      const auto featureSet = setsolver::getCardFeatures(frame, cards);
      std::cout << "features" << std::endl;
      for (const auto& feature: featureSet) {
        std::cout << feature << std::endl;
      }
      const auto matches = setsolver::findMatches(featureSet);
      setsolver::paintMatches(drawing, matches, cards);
                 
      imshow("cards", drawing);
    }
    waitKey(50);
  }

  return 0;
}
