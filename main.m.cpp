#include <cards.h>
#include <attributes.h>
#include <solver.h>
#include <paintmatches.h>
#include <utils.h>
#include <client.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <memory>
#include <iostream>
#include <algorithm>

#include <random>

namespace {  
  using namespace cv;
  using namespace setsolver;
  cv::Scalar CARD_HIGHLIGHT(124, 252, 0);

  std::random_device rd;
  std::mt19937_64 rng(rd());

  void processForDebug(Mat& drawing,
                       Cards& cards,
                       const Mat& frame,
                       std::shared_ptr<CardFeatures> knownFeatures) {
    const Point2f frameCenter(drawing.cols/2.0, drawing.rows/2.0);

    circle(drawing,
           frameCenter,
           4,
           Scalar(0, 0, 255),
           4);

    // sorting by distance from frame center.
    std::sort(cards.begin(), cards.end(),
              [&drawing, &frameCenter](const Card& lhs, const Card& rhs) {
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

      const auto predictedColor = tfColor(correctCard(frame, cards.front()));
      std::cout << "DEBUG: predicted color = " << predictedColor << std::endl;
      if (knownFeatures &&
          knownFeatures->color != predictedColor) {
        std::cout << "DEBUG: bad prediction.  Writing new file!"
                  << std::endl;
        std::cout << "DEBUG: known=" << *knownFeatures
                  << std::endl;

        const auto corrected = correctCard(frame, cards.front());
        std::ostringstream path;
        path << "./tf/training/"
             << knownFeatures->color
             << "-" << knownFeatures->symbol
             << "-" << knownFeatures->shading
             << "-" << knownFeatures->number
             << "-" << std::hex << rng()
             << ".JPG";
        assert(imwrite(path.str(), corrected));
      }
      //const auto features = getCardFeatures(frame, cards, true);
      // if (!features.empty()) {
      // std::cout << "DEBUG: "
      //           << features.front()
      //           << std::endl;
      // }

      // if (knownFeatures) {
      //   const auto corrected = correctCard(frame, cards.front());
      //   std::ostringstream path;
      //   path << "./tf/training/"
      //        << knownFeatures->color
      //        << "-" << knownFeatures->symbol
      //        << "-" << knownFeatures->shading
      //        << "-" << knownFeatures->number
      //        << "-" << std::hex << rng()
      //        << ".JPG";
      //   assert(imwrite(path.str(), corrected));
      // }
    }
  }                      
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

  std::shared_ptr<CardFeatures> knownFeatures;
  if (parser.has("color")
      && parser.has("symbol")
      && parser.has("shading")
      && parser.has("number")) {
    const auto numberValue = parser.get<size_t>("number");
    assert(1 <= numberValue
           && numberValue <= 3
           && "failed to parse a valid number");
    knownFeatures = std::make_shared<CardFeatures>(CardFeatures{
        parseColor(parser.get<std::string>("color")),
        parseSymbol(parser.get<std::string>("symbol")),
        parseShading(parser.get<std::string>("shading")),
        numberValue}
    );
  }

  while (true) {
    Mat frame;
    cap >> frame; // get a new frame from camera
    Mat drawing = frame.clone();
    
    auto cards = findCards(frame, debug);

    if (debug) {
      processForDebug(drawing, cards, frame, knownFeatures);
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
