#ifndef _RECOG_H_
#define _RECOG_H_

#include "inferxlite.h"
#include "hash.h"
#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#endif
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "pipe.h"
#define FEATURE_LENGTH 512

using namespace cv;
using namespace std;
class Recog 
{
public:
    Recog();
    virtual ~Recog();
    void model_init(char* model_path,char* model_name,char* model_title); 
    void getfeature(Mat&image,vector<float>& feature);
    //float*Recog::getfeature_gpu_set(image,device_id);
    bool   m_force_gray;	
    Mat    m_mean_mat;
    Mat    m_scale_mat;
    int    m_resize_w;
    int    m_resize_h;
    char   m_feature_name[100];
    char   m_inferxlite_model_path[1000]; 
    char   m_model_title[100];
    inferx_context m_handle_ctx;
    int shape[4];
};

#endif
