#ifndef FACE_MANAGER_H
#define FACE_MANAGER_H

#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"

using namespace std;

class face_veri;
class face_attr;

class FaceManager {

public:
    FaceManager(const char* szModelDir);
    ~FaceManager();

    float get_score(const std::vector<float>& feature1, const std::vector<float>& feature2);

    void get_feature(cv::Mat img, vector<vector<float>>& face_rects, vector<vector<float>>& features);
    void get_gender(cv::Mat img, vector<vector<float>>& face_rects, std::vector<int>& gender_results);

private:
    face_veri* pFaceVeri;
    face_attr* pFaceAttr;
};

#endif // FACE_MANAGER_H
