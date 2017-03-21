// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image

#include <cards.h>
#include <attributes.h>

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

/*
void getCardFeatures(const Mat& card)
{
  const auto contours = computeCardContours(card);
  std::cout << "object count: " << contours.size()x << std::endl;

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
*/

void Otsu_Demo( int, void* )
{}

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

    const auto cards = setsolver::find(frame);

    if (cards.size() && 0 == cards.size() % 3) {
      imwrite("./frame.JPG", frame);
      
      
      /// Draw contours
      Mat drawing = frame.clone(); 
      for( int i = 0; i< cards.size(); i++ ) {
        Scalar color(124, 252, 0);
        drawContours(drawing, cards, i, color, 2, 8);
      }

      imshow("contours", drawing);
      waitKey(50);

      const auto featureSet = setsolver::getCardFeatures(frame, cards);
      // for (const auto& features: featureSet) {        
      //   std::cout << features << std::endl;
      // }
    }
  }

  return 0;
}
