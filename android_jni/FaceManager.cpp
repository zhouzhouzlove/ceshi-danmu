#include "FaceManager.h"
#include "face_veri.hpp"
#include "face_attr.hpp"

FaceManager::FaceManager(const char* szModelDir) {

    pFaceVeri = face_veri::get(szModelDir);
    pFaceAttr = face_attr::get(szModelDir);

}

FaceManager::~FaceManager() {
    if(pFaceVeri) {
        delete pFaceVeri;
        pFaceVeri = nullptr;
    }
    if(pFaceAttr) {
        delete pFaceAttr;
        pFaceAttr = nullptr;
    }
}

float FaceManager::get_score(const std::vector<float>& feature1, const std::vector<float>& feature2) {
    float score = 0;
    if(pFaceVeri) {
        score = pFaceVeri->get_score(feature1, feature2);
    }
    return score;
}

void FaceManager::get_feature(cv::Mat input, vector<vector<float>>& face_rects, vector<vector<float>>& features) {
    if(pFaceVeri) {
        pFaceVeri->detect_face(input, face_rects);
        for(int i = 0; i < face_rects.size(); i++) {
            vector<float> feature;
            pFaceVeri->get_feature(input, face_rects[i], feature);
            features.push_back(feature);
        }
    }
}

void FaceManager::get_gender(cv::Mat input, vector<vector<float>>& face_rects, std::vector<int>& gender_results) {
    if(pFaceAttr) {
        pFaceAttr->detect_face(input, face_rects);
        for(int i = 0; i < face_rects.size(); i++) {
            int gender_result;
            pFaceAttr->get_gender(input, face_rects[i], gender_result);
            gender_results.push_back(gender_result);
        }
    }
}