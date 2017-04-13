#include <debug.h>

#include <utils.h>
#include <client.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <random>

namespace setsolver {

  namespace {
    using namespace cv;
    const cv::Scalar CARD_HIGHLIGHT(124, 252, 0);

    std::random_device rd;
    std::mt19937_64 rng(rd());    
  }

  std::shared_ptr<CardFeatures> getKnownFeatures(const cv::CommandLineParser& parser)
  {
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
    return knownFeatures;
  }

  void debugCards(cv::Mat& drawing,
                  Cards& cards,
                  const cv::Mat& frame,
                  std::shared_ptr<CardFeatures> knownFeatures)
  {
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
      
      const auto features = getCardFeatures(frame, cards, true);
      if (!features.empty()) {
        std::cout << "DEBUG (non tf):"
                  << features.front()
                  << std::endl;
      }
    }
  }                      

}
