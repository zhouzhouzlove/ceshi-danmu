#ifndef MODULE_UTILS_HPP 
#define MODULE_UTILS_HPP
#include <sys/time.h>
#define WIDTH_HEIGHT_RATIO 1.25
#define MAKE_BORDER_CRITERION 0.05

class perftimer{
public:
    perftimer();
    int start();
    int time_end(char * name);
    int time_turn(char * name);
    ~perftimer();

private:
    struct timeval begin;
    struct timeval tmp;
    struct timeval end;
};
#endif
#include <opencv2/opencv.hpp>
using namespace cv;
int make_border(cv::Mat & src_mat, cv::Mat & dst_mat);
