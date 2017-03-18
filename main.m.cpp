// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image

#include <cards.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <math.h>
#include <string.h>

#include <map>
#include <set>

using namespace cv;
using namespace std;

enum class Symbol {
  DIAMOND,
  SQUIGGLE,
  OVAL,
};

using Contour = vector<Point>;

static void help()
{
  cout <<
    "\nA program using pyramid scaling, Canny, contours, contour simpification and\n"
    "memory storage (it's got it all folks) to find\n"
    "squares in a list of images pic1-6.png\n"
    "Returns sequence of squares detected on the image.\n"
    "the sequence is stored in the specified memory storage\n"
    "Call:\n"
    "./squares [file_name (optional)]\n"
    "Using OpenCV version " << CV_VERSION << "\n" << endl;
}

const char* wndname = "Square Detection Demo";

// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
  for( size_t i = 0; i < squares.size(); i++ )
    {
      const Point* p = &squares[i][0];
      int n = (int)squares[i].size();
      polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }

  imshow(wndname, image);
}

enum class Color {
  RED,
  GREEN,
  PURPLE,
};

Color getColorFromHue(const double hue) {
  if (hue < 45.0) {
    return Color::RED;
  } else if (hue > 90) {
    return Color::PURPLE;
  } else {
    return Color::GREEN;
  }  
}

std::string ColorToString(const Color color) {
  switch (color) {
  case Color::RED:
    return "RED";
  case Color::PURPLE:
    return "PURPLE";
  case Color::GREEN:
    return "GREEN";
  }
}

Color computeColor(const Mat& card, const Mat& mask) {
  Mat blurred;
  GaussianBlur(card, blurred, Size(13, 13), 0, 0);

  Mat hsv;
  cvtColor(blurred, hsv, CV_BGR2HSV);

  std::cout << "object mean: " << mean(hsv, mask) << " for type " << card.type() << std::endl;
  const auto color = getColorFromHue(mean(hsv, mask)[0]);
  return color;
}

enum class Shading {
  SOLID,
  STRIPED,
  OPEN,
};

std::string ShadingToString(const Shading color) {
  switch (color) {
  case Shading::SOLID:
    return "SOLID";
  case Shading::STRIPED:
    return "STRIPED";
  case Shading::OPEN:
    return "OPEN";
  }
}

Shading computeShading(const Mat& card, const Mat& mask)  {
  // run a threshol to make the image black/white.
  // let the fraction of white pixels determing shading.
  Mat gray;
  cvtColor(card, gray, COLOR_BGR2GRAY);
  
  Mat binary;
  cv::threshold(gray, binary, 185.0, 255.0, THRESH_BINARY_INV);

  // std::cout << "showing threshold" << std::endl;
  // imshow(wndname, binary);
  // waitKey();

  // std::cout << "showing shading mask" << std::endl;
  // imshow(wndname, mask);
  // waitKey();
  
  // bitwise_and(binary, mask, binary);
  // imshow(wndname, binary);
  // waitKey();

  std::cout << "shading mean: " << mean(binary, mask) << std::endl;
  const auto meanShading = mean(binary, mask)[0];
  if (meanShading < 100) {
    return Shading::OPEN;
  }
  else if (meanShading > 190) {
    return Shading::SOLID;
  }
  else {
    return Shading::STRIPED;
  }
}

std::string symbolToString(const Symbol symbol) {
  switch (symbol) {
  case Symbol::DIAMOND:
    return "DIAMOND";
  case Symbol::SQUIGGLE:
    return "SQUIGGLE";
  case Symbol::OVAL:
    return "OVAL";
  }
}

void drawContour(const Mat& card,
                 const vector<Point>& contour) {
  vector<vector<Point>> contours;
  contours.push_back(contour);

  Mat mask(card.size(), CV_8U);
  mask = 0;
  drawContours(mask, contours, 0, 255, 10);
  
  // imshow("symbol contour", mask);
  // waitKey();
}

Symbol computeSymbol(const Mat& card,
                     const vector<Point>& contour) {
  vector<Point> approx;
  approxPolyDP(Mat(contour), approx, arcLength(Mat(contour), true)*0.02, true);

  if (!isContourConvex(approx)) {
    return Symbol::SQUIGGLE;
  }

  // identify ratio of contour area to min bounding rectangle area.
  // oval is assumed to have a higher ratio.
  const auto rec = minAreaRect(approx);
  const auto area = contourArea(approx, false);
  const auto ratio = area / rec.size.area();

  std::cout << "rectangle: " << rec.size << std::endl;
  std::cout << "area: " << area << std::endl;
  std::cout << "ratio: " << area / rec.size.area() << std::endl;
  
  if (ratio > 0.8) {
    return Symbol::OVAL;    
  }

  return Symbol::DIAMOND;
}


vector<vector<Point>> computeCardContours(const Mat& card) {
  Mat gray;
  cvtColor(card, gray, CV_BGR2GRAY);

  Mat blurred;
  GaussianBlur(gray, blurred, Size(9, 9), 0, 0);

  // imshow(wndname, blurred);
  // waitKey();
  
  Mat _img;
  double otsu_thresh_val = cv::threshold(blurred,
                                         _img,
                                         0,
                                         255,
                                         CV_THRESH_BINARY | CV_THRESH_OTSU);
  
  Mat canny;
  Canny(blurred, canny, otsu_thresh_val*0.25, otsu_thresh_val*0.5);

  // dilate canny output to remove potential
  // holes between edge segments
  dilate(canny, canny, Mat(), Point(-1,-1));

  // imshow(wndname, canny);
  // waitKey();

  vector<vector<Point>> contours;
  findContours(canny,
               contours,
               CV_RETR_EXTERNAL,
               CV_CHAIN_APPROX_SIMPLE);

  /// Draw contours
  Mat drawing = card.clone(); 
  for( int i = 0; i< contours.size(); i++ ) {
    Scalar color(124, 252, 0);
    drawContours(drawing, contours, i, color, 2, 8);
  }

  // imshow(wndname, drawing);
  // waitKey();

  return contours;
}

Mat computeFeatureMask(const Mat& card,
                       const vector<vector<Point>>& contours) {
  Mat mask(card.size(), CV_8U);
  mask = 0;
  drawContours(mask, contours, 0, 255,-1);

  // imshow("feature mask", mask);
  // waitKey();
  
  return mask;
}

void getCardFeatures(const Mat& card)
{
  const auto contours = computeCardContours(card);
  std::cout << "object count: " << contours.size() << std::endl;

  const auto featureMask = computeFeatureMask(card, contours);

  const auto color = computeColor(card, featureMask);
  std::cout << "color: " << ColorToString(color) << std::endl;
  
  const auto shading = computeShading(card, featureMask);
  std::cout << "shading: " << ShadingToString(shading) << std::endl;

  const auto symbol = computeSymbol(card, contours.front());
  std::cout << "symbol: " << symbolToString(symbol) << std::endl;

  imshow("card", card);
  std::cout << "SUMMARY :: number=" << contours.size()
            << " color=" << ColorToString(color)
            << " shading=" << ShadingToString(shading)
            << " symbol=" << symbolToString(symbol)
            << std::endl;
  waitKey();
}

extractcards::Cards locateSquares(const Mat& image) {
  Mat blurred;
  GaussianBlur(image, blurred, Size(17, 17), 0, 0);
  return extractcards::find(blurred);  
}

void Otsu_Demo( int, void* )
{

}

int main(int argc, char** argv)
{
 
  static const char* names[] = { "./set.JPG", 0 };
  help();

  if( argc != 1) {
    std::cout << "expecting one arguments" << std::endl;
    exit(1);
  }

  VideoCapture cap(0); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
    exit(1);

  void Otsu_Demo( int, void* );
  int multiplier = 1;
  int maxMultiplier = 20;

  namedWindow("canny", 1);
  createTrackbar("canny",
                 "canny", &multiplier,
                  maxMultiplier, Otsu_Demo);

  Otsu_Demo(multiplier, 0);

  while (true) {
    Mat frame;
    cap >> frame; // get a new frame from camera

    const auto cards = extractcards::find(frame);

    if (0 == cards.size() % 3) {
      /// Draw contours
      Mat drawing = frame.clone(); 
      for( int i = 0; i< cards.size(); i++ ) {
        Scalar color(124, 252, 0);
        drawContours(drawing, cards, i, color, 2, 8);
      }

      imshow("contours", drawing);
      waitKey(50);
    }
    

    //const auto squares = locateSquares(frame);
    /*
    if (0 == squares.size() % 3) {
      auto copy = frame.clone();
      drawSquares(copy, squares);
      waitKey();
    }
    */
  }

  /*
  for(int i = 0; names[i] != 0; i++) {

    Mat image = imread(names[i], 1);
      if( image.empty() )
        {
          cout << "Couldn't load " << names[i] << endl;
          continue;
        }

      // imshow(wndname, image);
      // waitKey();

      const auto squares = locateSquares(image);
      if (0 != squares.size() % 3) {
        cout << "unexpected number of cards detected: " << squares.size()
             << endl;
        exit(1);
      }
      for (const auto& square: squares) {
        Mat card = image(boundingRect(square));

        imshow(wndname, card);
        waitKey();

        getCardFeatures(card);
      }
    }
*/

  return 0;
}
