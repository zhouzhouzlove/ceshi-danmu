#ifndef _GENDER_GUDGE_HEADERS
#include "gender_judge.h"
#endif

extern "C" void Gender(char* path, char* model, char* data, int *shape, int nshape, void* pdata, void** pout,int *len, struct inferx_handler *hd);

GenderJudge::GenderJudge()
{
  //do nothing
}
GenderJudge::~GenderJudge()
{
  inferx_destroy_context(m_handle_ctx);
//  delete &m_mean_mat;
//  delete &m_scale_mat; 
}
void GenderJudge::model_init(char* model_path,char* model_name,char* model_title)
{
  m_force_gray=false;
  m_resize_h=112;
  m_resize_w=96;
  int mat_type=m_force_gray?CV_32FC1:CV_32FC3;
  //m_mean_mat=*(new Mat(m_resize_h,m_resize_w,mat_type,Scalar::all(127.5)));
  //m_scale_mat=*(new Mat(m_resize_h,m_resize_w,mat_type,Scalar::all(1)));
  m_mean_mat= Mat(m_resize_h,m_resize_w,mat_type,Scalar::all(127.5));
  m_scale_mat= Mat(m_resize_h,m_resize_w,mat_type,Scalar::all(1));
 
  strcpy(m_feature_name,"gender_prob");
  strcpy(m_model_title,model_title);
  strcpy(m_inferxlite_model_path,model_path);
  
  m_handle_ctx=inferx_create_context();
  //int shape[4]={1,3,m_resize_h,m_resize_w};
  shape[0]=1;
  shape[1]=3;
  shape[2]=m_resize_h;
  shape[3]=m_resize_w;
   // #if __ANDROID__
  inferx_insert_model_func(model_name, (void*)Gender, &(m_handle_ctx->hd));
   // #endif

  inferx_load(m_handle_ctx,m_inferxlite_model_path,model_name,shape,4);

}
void GenderJudge::getGender(Mat&image,int &gender_result)
{
    Mat m_img_src=image;
    //namedWindow("gender_img",0); 
    //imshow("gender_img",m_img_src);
    //waitKey(1);

    Mat m_img_resize,m_img_resize_float;
    resize(m_img_src,m_img_resize,Size(m_resize_w,m_resize_h),(0,0),(0,0),INTER_LINEAR);
    //m_img_resize=imread("debug_crop.jpg");    
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
    
    if((float)result[0] > (float)result[1])
    {
       gender_result=0;
    }
    else
    {
       gender_result=1;
    }
    //cout<<"result 0:"<<(float)result[0]<<endl;
   // cout<<"result 1:"<<(float)result[1]<<endl;
   free(data);
   free(float_data);     
}

