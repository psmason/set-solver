#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#include <client.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

namespace {
  const char* PORT = "9000";
  int sockfd = -1;

  void initSocket() {
    struct addrinfo hints, *servinfo, *p;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    assert(0 == getaddrinfo("localhost", PORT, &hints, &servinfo)
           && "Failed to get address info");

    for(p = servinfo; p != NULL; p = p->ai_next) {
      if ((sockfd = socket(p->ai_family, p->ai_socktype,
                           p->ai_protocol)) == -1) {
        continue;
      }

      if (connect(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
        close(sockfd);
        continue;
      }

      break;
    }
    assert(p != NULL && sockfd != -1
           && "Failed to connect to model server");
    freeaddrinfo(servinfo); // all done with this structure
  }

  void getPrediction(char buf[], const std::vector<uchar>& encoded) {

    if (-1 == sockfd) {
      initSocket();
    }

    int bytesSent = send(sockfd, encoded.data(), encoded.size(), 0);
    if (static_cast<int>(encoded.size()) != bytesSent) {
      perror("send");
    }

    int numbytes;
    numbytes = recv(sockfd, buf, 1024*1024, 0);
    assert(-1 != numbytes
           && "Failed to receive prediction response");
    buf[numbytes] = '\0';
  }

}

namespace setsolver {
  
  Color tfColor(const cv::Mat& card) {
    cv::Mat copy;    
    cv::cvtColor(card, copy, CV_BGR2RGB);
    
    std::vector<uchar> encoded;
    assert(imencode(".jpg", copy, encoded) && "Failed to encode image");

    char buf[1024*1024];
    getPrediction(buf, encoded);
    printf("prediction: %s\n", buf);

    return parseColor(std::string(buf));
  }

}




