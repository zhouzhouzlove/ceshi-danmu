#ifndef _RECOG_HEADERS
#include "recog.h"
#endif
#if __ANDROID__
#include <android/log.h>
#endif

#define INPUT_IMAGE_HEIGHT 112
#define INPUT_IMAGE_WIDTH 96
#define IMAGE_MEAN_VALUE 127.5
#define IMAGE_SCALE_VALUE 0.0078125
extern "C" void resnet84(char* path, char* model, char* data, int *shape, int nshape, void* pdata, void** pout,int *len, struct inferx_handler *hd);
#if __ANDROID__
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, "perfxlab_recog", __VA_ARGS__);
#endif
Recog::Recog()
{
  //do nothing
}
Recog::~Recog()
{
  inferx_destroy_context(m_handle_ctx);
}
void Recog::model_init(char* model_path,char* model_name,char* model_title)
{
  m_force_gray=false;
  m_resize_h=INPUT_IMAGE_HEIGHT;
  m_resize_w=INPUT_IMAGE_WIDTH;
  int mat_type=m_force_gray?CV_32FC1:CV_32FC3;
  //m_mean_mat=*(new Mat(m_resize_h,m_resize_w,mat_type,Scalar::all(127.5)));
  //m_scale_mat=*(new Mat(m_resize_h,m_resize_w,mat_type,Scalar::all(0.0078125)));
  m_mean_mat=Mat(m_resize_h,m_resize_w,mat_type,Scalar::all(IMAGE_MEAN_VALUE));
  m_scale_mat=Mat(m_resize_h,m_resize_w,mat_type,Scalar::all(IMAGE_SCALE_VALUE));

//  strcpy(m_feature_name,"fc5");
  //strcpy(m_feature_name,"fc5");

  /*******add for resnet84******/

  strcpy(m_feature_name,"BatchNorm_82");

  /*******add for resnet84******/

  strcpy(m_model_title,model_title);
  strcpy(m_inferxlite_model_path,model_path);
  m_handle_ctx=inferx_create_context();
  //m_handle_ctx=inferx_get_context();
  //int shape[4]={1,3,m_resize_h,m_resize_w};
  shape[0]=1;
  shape[1]=3;
  shape[2]=m_resize_h;
  shape[3]=m_resize_w;
  //#if __ANDROID__
        inferx_insert_model_func(model_name, (void*)resnet84, &(m_handle_ctx->hd));
   // #endif

  inferx_load(m_handle_ctx,m_inferxlite_model_path,model_name,shape,4);
}
void Recog::getfeature(Mat&image,vector<float>& feature)
{
    if(image.empty())
    {
      #if __ANDROID__
      LOGE("recog input image is null!");
      #endif
      std::cout<<" recog input image is null!"<<std::endl;
      exit(1);
    }
    /*******add for resnet84******/
    /*
    for(int i=0; i<image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            uchar tmp = image.at<Vec3b>(i, j)[2];
            image.at<Vec3b>(i, j)[2] = image.at<Vec3b>(i, j)[0];
            image.at<Vec3b>(i, j)[0] = tmp;
        }
    }
     */
    /*******add for resnet84******/

    Mat m_img_src=image;
    Mat m_img_resize,m_img_resize_float;
    resize(m_img_src,m_img_resize,Size(m_resize_w,m_resize_h),(0,0),(0,0),INTER_LINEAR);

    m_img_resize.convertTo(m_img_resize_float,CV_32FC3);
    cout<<"recog img resize is ok ,before substract"<<endl;
    subtract(m_img_resize_float,m_mean_mat,m_img_resize_float,noArray(),CV_32FC3);

    m_img_resize_float=m_img_resize_float.mul(m_scale_mat);
    float *data=(float*)malloc(sizeof(float)*m_resize_h*m_resize_w*3);
    vector<Mat>input_channels;
    for(int c_idx=2;c_idx>=0;c_idx--){
      Mat channel(m_resize_h,m_resize_w,CV_32FC1,data+m_resize_h*m_resize_w*c_idx);
      input_channels.push_back(channel);
    }
    split(m_img_resize_float,input_channels);
    /*
    for(int i=0; i<m_resize_h*m_resize_w*3; ++i)
    {
	//std::cout<<"recog input data:["<<i<<"]: "<<data[i]<<std::endl;
	//printf("recog input data:[%d]: %f\n",i,data[i]);
    }
    */
    float* result;
    int len;
    inferx_run(m_handle_ctx,data);
    inferx_get_result(m_handle_ctx,m_feature_name,(void**)&result,&len);
    std::cout<<"recog result length  :"<<len<<std::endl;
    //for(int i=0;i<512;i++)
    for(int i=0;i<FEATURE_LENGTH;i++)
    {
//      std::cout<<"----"<<i<<"---"<<result[i]<<std::endl;
      feature.push_back(result[i]);
    }
    free(data);
}
