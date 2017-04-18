#pragma once

#include <attributes.h>

#include <memory>

namespace cv {
  class CommandLineParser;
  class Mat;
}

namespace setsolver {

  std::shared_ptr<CardFeatures> getKnownFeatures(const cv::CommandLineParser& parser);

  void debugCards(cv::Mat& drawing,
                  Cards& cards,
                  const cv::Mat& frame,
                  std::shared_ptr<CardFeatures> knownFeatures);

}
