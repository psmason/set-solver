#include <shading.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

namespace setsolver {

  namespace {
    using namespace cv;
    using namespace std;
    
    string shadingToString(const Shading color) {
      switch (color) {
      case Shading::SOLID:
        return "SOLID";
      case Shading::STRIPED:
        return "STRIPED";
      case Shading::OPEN:
        return "OPEN";
      }
    }

    Mat testSrc;
    Mat testDst;
    int threshold_value = 0;
    int threshold_type = 3;
    int cannyLowThreshold;
    int cannyRatio = 3;
  }

  std::ostream& operator<< (std::ostream& stream, const Shading& shading)
  {
    stream << shadingToString(shading);
    return stream;
  }


  void Threshold_Demo( int, void* )
  {
    /* 0: Binary
       1: Binary Inverted
       2: Threshold Truncated
       3: Threshold to Zero
       4: Threshold to Zero Inverted
    */

    Mat detected_edges;

    //threshold(testSrc, testDst, threshold_value, 255, threshold_type);
    //imshow("shading threshold", testDst);
    /// Reduce noise with a kernel 3x3
    blur(testSrc, detected_edges, Size(1, 1) );    

    /// Canny detector
    Canny(detected_edges, detected_edges,
          cannyLowThreshold, cannyLowThreshold*cannyRatio, 3);

    /// Using Canny's output as a mask, we display our result
    testDst = Scalar::all(0);

    testSrc.copyTo(testDst, detected_edges);
    imshow("shading threshold", testDst);
  }

  cv::Mat quantizeImage(const cv::Mat& inImage, int numBits)
  {
    cv::Mat retImage = inImage.clone();

    uchar maskBit = 0xFF;

    // keep numBits as 1 and (8 - numBits) would be all 0 towards the right
    maskBit = maskBit << (8 - numBits);

    for(int j = 0; j < retImage.rows; j++)
      for(int i = 0; i < retImage.cols; i++)
        {
          auto v = retImage.at<uchar>(j, i);
          v = v & maskBit;
          retImage.at<uchar>(j,i) = v;
          /*
          cv::Vec3b valVec = retImage.at<cv::Vec3b>(j, i);
          valVec[0] = valVec[0] & maskBit;
          valVec[1] = valVec[1] & maskBit;
          valVec[2] = valVec[2] & maskBit;
          retImage.at<cv::Vec3b>(j, i) = valVec;
          */
        }

    return retImage;
  }


  Shading computeShading(const Mat& card, const Mat& mask)  {
    // run a threshol to make the image black/white.
    // let the fraction of white pixels determing shading.

    Mat colorBlurred;
    GaussianBlur(card, colorBlurred, Size(7, 7), 0, 0);
    //imshow("quantized", quantizeImage(colorBlurred, 4));
    
    Mat gray;
    cvtColor(card, gray, COLOR_BGR2GRAY);

    Mat blurred2;
    GaussianBlur(gray, blurred2, Size(1,1), 0, 0);

    imshow("quantized", quantizeImage(blurred2, 4));

    gray.copyTo(testSrc);

    Mat masked2;
    blurred2.copyTo(masked2, mask);
    imshow("shading gray", masked2);

    /*
    namedWindow("shading threshold", CV_WINDOW_AUTOSIZE);
    /// Create Trackbar to choose type of Threshold

    createTrackbar("canny value",
                   "shading threshold", &cannyLowThreshold, 100, Threshold_Demo);
    */
    /*
    createTrackbar("threshold type",
                   "shading threshold", &threshold_type,
                   4, Threshold_Demo);

    createTrackbar("threshold value",
                   "shading threshold", &threshold_value,
                   255, Threshold_Demo);
    */

    //Threshold_Demo(0, 0);
    
    Mat binary;
    cv::threshold(gray, binary, 150.0, 255.0, THRESH_BINARY_INV);
    //imshow("shading binary", binary);
    

    Mat blurred;
    GaussianBlur(card, blurred, Size(21, 21), 0, 0);
    
    Mat hsv;
    cvtColor(blurred, hsv, CV_BGR2HSV);
    std::vector<Mat> channels;
    split(hsv, channels);
    // Mat saturationMask = mask & (channels[1] > 10);

    imshow("hue channel", channels[0]);
    imshow("saturation channel", channels[1]);

    std::cout << "showing threshold" << std::endl;
    //imshow("shading threshold", binary);

    // std::cout << "showing shading mask" << std::endl;
    // imshow(wndname, mask);
    // waitKey();
  
    // bitwise_and(binary, mask, binary);
    // imshow(wndname, binary);
    // waitKey();

    std::cout << "gray mean: " << mean(gray, mask) << std::endl;
    std::cout << "gray mean2: " << mean(blurred2, mask) << std::endl;
    std::cout << "shading mean: " << mean(binary, mask) << std::endl;
    std::cout << "saturation mean: " << mean(channels[1], mask) << std::endl;

    //waitKey();
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

}
