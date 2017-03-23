// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image

#include <cards.h>
#include <attributes.h>
#include <solver.h>

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

    const auto cards = setsolver::find(frame);
    if (cards.size() && 0 == cards.size() % 3) {
      /// Draw contours
      Mat drawing = frame.clone(); 
      for( int i = 0; i< cards.size(); i++ ) {
        drawContours(drawing,
                     cards,
                     i,
                     Scalar(124, 252, 0),
                     2,
                     8);
      }

      const auto featureSet = setsolver::getCardFeatures(frame, cards);
      const auto matches = setsolver::findMatches(featureSet);

      std::array<Scalar, 6> colorWheel;
      colorWheel[0] = Scalar(0, 0, 255);   // red
      colorWheel[1] = Scalar(0, 255, 255); // yellow
      colorWheel[2] = Scalar(0, 255, 0);   // green
      colorWheel[3] = Scalar(255, 255, 0); // cyan
      colorWheel[4] = Scalar(255, 0, 0);   // blue
      colorWheel[5] = Scalar(255, 0, 255); // magenta
                 
      for (int i=0; i<matches.size(); ++i) {
        const auto& match = matches[i];
        assert(3 == match.size());

        std::cout << "match found! ";
        for (const auto& index: match) {
          std::cout << " " << index;
        }
        std::cout << std::endl;
        
        for (int j=0; j<match.size(); ++j) {
          Mat overlay;
          drawing.copyTo(overlay);
          const auto card = cards[match[j]];
          const auto center = minAreaRect(card).center;
          circle(overlay, center, (i+1)*5, colorWheel[i%6], 4);
          addWeighted(overlay, 0.3, drawing, 0.7, 0, drawing);
        }
      }
      imshow("cards", drawing);
      waitKey(50);
    }
  }

  return 0;
}
