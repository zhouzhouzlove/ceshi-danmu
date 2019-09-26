#pragma once
#include "inferxlite.h"
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

using namespace cv;

class FaceDetect 
{
public:
    FaceDetect();
    virtual ~FaceDetect();

    void Tiling(int tile_dim_, int height, int width, const float* input, int c_out, std::vector<std::vector<float> >& output,float scale=1.0);
    void softmax(std::vector<std::vector<float> >& input, int size, std::vector<std::vector<float> >& output);
    std::vector<std::vector<float> > NMS(size_t count, const std::vector<std::vector<float> >& box, float nms);
    std::vector<std::vector<float> > GetRois(float* pixel_p,float* bb_p,int feat_rows,int feat_cols);
    void model_init(char* model_path,char* model_name,char* model_title); 
    std::vector<std::vector<float> > detect(Mat &img);
    float  m_expand_scale;    
    bool   m_force_gray;
    Mat    m_mean_mat;
    Mat    m_scale_mat;
    int    m_resize_w;
    int    m_resize_h;
    char   m_pixel_blob_name[100];
    char   m_bb_blob_name[100];
    char   m_inferxlite_model_path[1000];
    char   m_model_title[100];
    inferx_context m_handle_ctx;
    std::vector<Mat> m_face_crop_vec;	
    int shape[4];    

private:
	int preprocess(Mat input, Mat output, int w_new, int h_new);
};
