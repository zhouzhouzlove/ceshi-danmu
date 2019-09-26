#ifndef _GENDER_GUDGE_HEADERS
#include "face_quality.h"
#include <opencv2/opencv.hpp>
#endif
#define INPUT_WIDTH 96
#define INPUT_HEIGHT 112
#define INPUT_SCALE_VALUE 1.0 
//#define INPUT_SCALE_VALUE 0.003921 
#define FACE_QUALITY_UPPER_THRESHOLD 0.8 
#define FACE_QUALITY_LOWER_THRESHOLD 0.2 
#if __ANDROID__
#include<android/log.h>
#endif
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, "perfxlab_detect", __VA_ARGS__);
extern "C" void AlexNet(char* path, char* model, char* data, int *shape, int nshape, void* pdata, void** pout,int *len, struct inferx_handler *hd);

FaceQuality::FaceQuality()
{
	//do nothing
}
FaceQuality::~FaceQuality()
{
	inferx_destroy_context(m_handle_ctx);
}
void FaceQuality::model_init(char* model_path,char* model_name,char* model_title)
{
  m_force_gray=false;
  m_resize_h=INPUT_HEIGHT;
  m_resize_w=INPUT_WIDTH;
  int mat_type=m_force_gray?CV_32FC1:CV_32FC3;
  m_mean_mat=*(new Mat(m_resize_h,m_resize_w,mat_type,cv::Scalar(104,117,123)));
  m_scale_mat=*(new Mat(m_resize_h,m_resize_w,mat_type,cv::Scalar::all(INPUT_SCALE_VALUE)));
 
  strcpy(m_feature_name,"prob");
  //strcpy(m_feature_name,"prob");
  strcpy(m_model_title,model_title);
  strcpy(m_inferxlite_model_path,model_path);
  
  m_handle_ctx=inferx_create_context();
  //int shape[4]={1,3,m_resize_h,m_resize_w};
  shape[0]=1;
  shape[1]=3;
  shape[2]=m_resize_h;
  shape[3]=m_resize_w;
  
  void (*p)(char*, char*, char*, int*, int, void* , void** ,int *, struct inferx_handler *);
  p = AlexNet;
  inferx_insert_model_func(model_name, (void*)p, &(m_handle_ctx->hd));
  inferx_load(m_handle_ctx,m_inferxlite_model_path,model_name,shape,4);

}
void FaceQuality::getQuality(Mat&image,int &quality_result)
{
  Mat m_img_src=image;
  cv::imwrite("face_quality.jpg",m_img_src);
  Mat m_img_resize,m_img_resize_float;
  resize(m_img_src,m_img_resize,Size(m_resize_w,m_resize_h),(0,0),(0,0),INTER_LINEAR);
  m_img_resize.convertTo(m_img_resize_float,CV_32FC3);
  subtract(m_img_resize_float,m_mean_mat,m_img_resize_float,noArray(),CV_32FC3);
  m_img_resize_float=m_img_resize_float.mul(m_scale_mat);
  float *data=(float*)malloc(sizeof(float)*m_resize_h*m_resize_w*3);  
  vector<Mat>input_channels; 
  for(int c_idx=0;c_idx<3;c_idx++){
    Mat channel(m_resize_h,m_resize_w,CV_32FC1,data+m_resize_h*m_resize_w*c_idx);
    input_channels.push_back(channel);
  }
  split(m_img_resize_float,input_channels);
  float *float_data=(float*)malloc(sizeof(float)*m_resize_h*m_resize_w*3);
  split(m_img_resize_float,input_channels);
  float* result;
  int len;
  inferx_run(m_handle_ctx,data);
  inferx_get_result(m_handle_ctx,m_feature_name,(void**)&result,&len);
  std::cout<<"result len = "<<len<<std::endl;
  float score = ((float*)result)[0];
  std::cout<<"quality score is "<<score<<std::endl;
  if(score > FACE_QUALITY_LOWER_THRESHOLD && score < FACE_QUALITY_UPPER_THRESHOLD)
  {
    quality_result=1;
  }
  else
  {
    quality_result=0;
  }
  free((void*)data);
}
