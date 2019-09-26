#include<iostream>
#include<string>
#include "face_detector.hpp"
#include "face_detect.h"
#include "face_quality.h"
#include <time.h>
#if __ANDROID__
#include <android/log.h>
#endif 
//#define FEATURE_LENGTH 128
#define visible_detect_expand_ratio 0.2

face_detector *face_detector::face_detector_ =0;
std::string face_detector::model_dir_="";

std::string face_detector::get_model_dir()
{
	return model_dir_;
}

face_detector *face_detector::get()
{
	return face_detector_;
}

face_detector *face_detector::get(const std::string & model_dir)
{

	//
	face_detector_ = new face_detector(model_dir);
	model_dir_ = model_dir;
	
	return face_detector_;
}

face_detector::face_detector(const std::string &model_dir)
{
        //detect
	//d::cout<<"detect init: "<<std::endl;	
	int string_length;
	string_length = model_dir.length();
	char model_path[string_length+1];
	for(int i=0; i<string_length; ++i)
		model_path[i]=model_dir[i];
	model_path[string_length]='\0';
        printf("model_path: %s\n",model_path);
	#if __ANDROID__
	__android_log_print(ANDROID_LOG_INFO,"perfxlab--","model init, load models ");
	__android_log_print(ANDROID_LOG_INFO,"perfxlab--","model dir: %s \n",model_path);
	#endif
	face_detect = new FaceDetect();
	face_detect->model_init(model_path, "FaceNetNew", "cn_1_1_224_320");
	detect_flag = 0;
	face_quality = new FaceQuality();
	//face_quality->model_init(model_path,"AlexNet","cn_1_1_112_96");
 	quality_result =0;
	quality_probs.clear(); 
	last_times_face_rects_vector.clear();
	#if __ANDROID__
	__android_log_print(ANDROID_LOG_INFO,"perfxlab--","complete loading model ");
	#endif
         
}
face_detector::~face_detector()
{

	delete face_detector_;
}
int face_detector::detect_face(cv::Mat img, std::vector<std::vector<float> > & face_rect_vector)
{
        if(img.empty())
	{
		#if __ANDROID__
			__android_log_print(ANDROID_LOG_INFO,"perfxlab--","No any face has been detected!");
		#endif
		std::cout<<"input img is null!"<<std::endl;
		exit(EXIT_FAILURE);
	}
 	cv::Mat quality_img;
	std::vector<std::vector<float> > face_rect_vector_result = face_detect->detect(img);
	cv::Rect quality_rect_face;
	for(int i=0; i<face_rect_vector_result.size(); ++i)
	{
		quality_rect_face.x = std::max(int(face_rect_vector_result[i][0]),0);
		quality_rect_face.y = std::max(int(face_rect_vector_result[i][1]),0);
		quality_rect_face.width = std::min(int(face_rect_vector_result[i][2]-face_rect_vector_result[i][0]),img.cols);
		quality_rect_face.height = std::min(int(face_rect_vector_result[i][3]-face_rect_vector_result[i][1]),img.rows);
                quality_img = img(quality_rect_face);
		//face_quality->getQuality(quality_img,quality_result);
		if(quality_result==1)
		{
			std::cout<<"quality measure up to identify face"<<std::endl;
			face_rect_vector.push_back(face_rect_vector_result[i]);
		}
	} 			
	if(face_rect_vector.size()==0)
	{
		#if __ANDROID__
			__android_log_print(ANDROID_LOG_INFO,"perfxlab--","No any face has been detected!");
		#endif
		std::cout<<"no any face has been detected!"<<std::endl;
		std::cout<<"no any face has been detected----------------!"<<std::endl;
		return 0;
	}
	return 1;
}
void face_detector::detect_face_dualcamera(cv::Mat visible_img, cv::Mat & infared_img, vector<vector<float> > & face_rects)
{
	std::cout<<"visible image size: "<<"cols="<<visible_img.cols<<" rows="<<visible_img.rows<<std::endl;	
	std::cout<<"infared image size: "<<"cols="<<infared_img.cols<<" rows="<<infared_img.rows<<std::endl;	
	std::vector<std::vector<float> > visible_detect_result;
	std::vector<std::vector<float> > infared_detect_result;
	visible_detect_result.clear();
	int detect_result = detect_face(visible_img,visible_detect_result);
	std::cout<<"visible_detect_result size: "<<visible_detect_result.size()<<std::endl;
	if(detect_result == 0)
		return;
	if(visible_detect_result.size() !=0)
	{
		int color_result = judge_color(visible_img, visible_detect_result[0]);
		if(color_result == 1)
		{
			return;
		}
	}
	if(infared_img.empty())
		std::cout<<"infared_img is empty"<<std::endl;
	for(int i = 0; i < visible_detect_result.size(); ++i)
	{   
		cv::Rect infared_rect;
		infared_rect.x = std::max(0, int(visible_detect_result[i][0]-visible_detect_expand_ratio*(visible_detect_result[i][2]-visible_detect_result[i][0])));
		infared_rect.y = std::max(0, int(visible_detect_result[i][1]-visible_detect_expand_ratio*(visible_detect_result[i][3]-visible_detect_result[i][1])));
		infared_rect.width = std::min(
                	int((visible_detect_result[i][2] - visible_detect_result[i][0])*(1.0+2*visible_detect_expand_ratio)),
               		visible_img.cols - infared_rect.x);
		infared_rect.height = std::min(
                	int((visible_detect_result[i][3] - visible_detect_result[i][1])*(1.0+2*visible_detect_expand_ratio)),
               		visible_img.rows - infared_rect.y);
		infared_rect.x= infared_rect.x - 320; //add for visible 1920,infared 1280
		std::cout<<"infared_rect.x: "<<infared_rect.x<<std::endl;
		std::cout<<"infared_rect.y: "<<infared_rect.y<<std::endl;
		std::cout<<"infared_rect.width: "<<infared_rect.width<<std::endl;
		std::cout<<"infared_rect.height: "<<infared_rect.height<<std::endl;
        	if(infared_rect.x<0 || infared_rect.y<0 || infared_rect.width<0 || infared_rect.height<0 ||
		infared_rect.x>=infared_rect.width || infared_rect.y>=infared_rect.height)
		{
			std::cout << "no face found for " << i << "th visible_detect_result" << std::endl;
			continue;
		}
		cv::Mat infared_input=infared_img(infared_rect);
		infared_detect_result.clear();
		detect_face(infared_input,infared_detect_result);
		if(infared_detect_result.size() != 0 )
		{
			face_rects.push_back(visible_detect_result[i]);
       		}
	}
	std::cout<<"face rect size: "<<face_rects.size()<<std::endl;	
    	// check and print
	for(int i=0; i<static_cast<int>(face_rects.size()); ++i)
	{
		vector<float> rect = face_rects[i];
		for(int j=0; j<static_cast<int>(rect.size()); ++j)
		std::cout << rect[j] << " ";
		std::cout << "" << std::endl;
	}
	return;
}

int face_detector::judge_color(cv::Mat input_img,std::vector<float> det_rect)
{
	cv::Mat input = input_img.clone();
    	cv::Rect face_rect;
    	face_rect.x = std::max(0,int(det_rect[0]+0.3*(det_rect[2]-det_rect[0])));
    	face_rect.y = std::max(0,int(det_rect[1]+0.3*(det_rect[3]-det_rect[1])));
    	face_rect.width = std::min(int((det_rect[2]-det_rect[0])*0.4),input.cols-face_rect.x);
    	face_rect.height = std::min(int((det_rect[3]-det_rect[1])*0.4),input.rows-face_rect.y);
    	cv::Mat face_rect_img = input(face_rect);
    	int pixel_sum = face_rect_img.cols * face_rect_img.rows;
    	resize(face_rect_img,face_rect_img,Size(30,30));
	cv::Mat hsv_img,h_float,s_float,v_float;
    	std::vector<cv::Mat> channels;
    	cvtColor(face_rect_img,hsv_img,CV_BGR2HSV);
    	split(hsv_img, channels);
    	channels[1].convertTo(s_float,CV_32F);
    	cv::Mat s_norm = s_float /255;
    	float sum = 0;
    	float sum_h = 0;
    	float sum_s = 0;
    	float sum_v = 0;
    	for(size_t j = 0; j< face_rect_img.cols; ++j)
    	{
        	for(size_t i=0; i<face_rect_img.rows; ++i)
        	{
            		sum_s = sum_s + s_float.at<float>(i,j);
       		}
    	}
    	pixel_sum = face_rect_img.cols * face_rect_img.rows;// Added by Yang. Last pixel_sum is calculated before resizing
    	float s_mean = sum_s / pixel_sum;

    	if(s_mean<20.0)
        	return 1;
    	else
        	return 0;
}
int face_detector::make_border(cv::Mat & src_mat, cv::Mat & dst_mat)
{
	if(src_mat.empty())
	{
		std::cout<<"input mat is null!"<<std::endl;
		return 1;
	}
	int src_width, src_height;
	src_width = src_mat.cols;
	src_height = src_mat.rows;

	cv::Scalar value;
	value = cv::Scalar(0,0,0);	
	float src_ratio = ((float)src_width) /((float)src_height);
	int top_edge,bottom_edge,left_edge,right_edge;
	src_mat.copyTo(dst_mat);	
	//__android_log_print(ANDROID_LOG_INFO,"perfxlab--","width-height ratio is %f \n",src_ratio);
	if((src_ratio-WIDTH_HEIGHT_RATIO)>MAKE_BORDER_CRITERION)
	{
		//__android_log_print(ANDROID_LOG_INFO,"perfxlab--","make border for height!");
		std::cout<<"the height is short, make border for height"<<std::endl;
		//top_edge = (int)((((float)src_width/WIDTH_HEIGHT_RATIO)-(float)src_height)/2);
		bottom_edge = (int)((((float)src_width/WIDTH_HEIGHT_RATIO)-(float)src_height));
		top_edge = 0;
		//cv::copyMakeBorder(src_mat, dst_mat,top_edge,bottom_edge,0,0,cv::BORDER_CONSTANT,value);
		cv::copyMakeBorder(dst_mat, dst_mat,top_edge,bottom_edge,0,0,cv::BORDER_CONSTANT,value);
		return 0;
	}
	if((WIDTH_HEIGHT_RATIO-src_ratio)>MAKE_BORDER_CRITERION)
	{
		//__android_log_print(ANDROID_LOG_INFO,"perfxlab--","make border for width!");
		std::cout<<"the width is short, make border for width"<<std::endl;
		//left_edge = (int)((((float)src_height*WIDTH_HEIGHT_RATIO)-(float)src_width)/2);
		right_edge = (int)((((float)src_height*WIDTH_HEIGHT_RATIO)-(float)src_width));
		left_edge = 0;
		//cv::copyMakeBorder(src_mat, dst_mat,0,0,left_edge,right_edge,cv::BORDER_CONSTANT,value);
		cv::copyMakeBorder(dst_mat, dst_mat,0,0,left_edge,right_edge,cv::BORDER_CONSTANT,value);
		//cv::imwrite("border_img.jpg",dst_mat);
		return 0;
	}
	//__android_log_print(ANDROID_LOG_INFO,"perfxlab--","don't need make border!");
	//src_mat.copyTo(dst_mat);	
	std::cout<<"input Mat ratio is right, no border is needed!"<<std::endl;
	return 0;	
}
