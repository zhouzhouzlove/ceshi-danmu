#include<iostream>
#include<string>
#include "face_attr.hpp"
#include <time.h>
#if __ANDROID__
#include <android/log.h>
#endif 

#include "face_detect.h"
#include "landmark.h"
#include "align.h"
#include "gender_judge.h"
//#define FEATURE_LENGTH 128

face_attr *face_attr::face_attr_ =0;
std::string face_attr::model_dir_="";

std::string face_attr::get_model_dir()
{
	return model_dir_;
}

face_attr *face_attr::get()
{
	return face_attr_;
}

face_attr *face_attr::get(const std::string & model_dir)
{

	//
	face_attr_ = new face_attr(model_dir);
	model_dir_ = model_dir;
	
	return face_attr_;
}

face_attr::face_attr(const std::string &model_dir)
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
	face_detector = new FaceDetect();
	landmark_detector = new Landmark();
	face_align = new Align();
	gender_judge = new GenderJudge(); 
	face_detector->model_init(model_path, "FaceNetNew", "cn_1_1_224_320");
	landmark_detector->model_init(model_path, "FaceLandmark", "bn_1_3_112_96"); 
	gender_judge->model_init(model_path,"Gender","dn_1_3_112_96");
	face_align->init();
	#if __ANDROID__
	__android_log_print(ANDROID_LOG_INFO,"perfxlab--","complete loading model ");
	#endif
         
}
face_attr::~face_attr()
{
	delete face_attr_;
}

int face_attr::detect_face(cv::Mat img, std::vector<std::vector<float> > & face_rect_vector)
{
	cv::Mat border_img;
	make_border(img,border_img);
//	cv::imwrite("detect_border_img.jpg",border_img);
	cv::Mat det_img;
	border_img.copyTo(det_img);
	face_rect_vector = face_detector->detect(det_img);
	if(face_rect_vector.size()==0)
	{
		#if __ANDROID__
			__android_log_print(ANDROID_LOG_INFO,"perfxlab--","No any face has been detected!");
		#endif
		std::cout<<"no any face has been detected!"<<std::endl;
		return 0;
	}
	return 1;
}


int face_attr::get_gender(cv::Mat img,std::vector<float> & det_rect,int & gender_result)
{

	//get landmark points
	std::vector<cv::Point2f> landmark;
	landmark = landmark_detector->detect(img,det_rect);
	std::vector<cv::Point2d> face_landmark;
	for(int j=0; j<5; ++j)
	{
		face_landmark.push_back(cv::Point2d(int(landmark[j].x),int(landmark[j].y)));
	}
	//align face
	cv::Mat aligned_crop;
	cv::Rect fdet_rect;
	fdet_rect.x = std::max(0,int(det_rect[0]));
	fdet_rect.y = std::max(0,int(det_rect[1]));
	fdet_rect.width = std::min(int(det_rect[2]-det_rect[0]),img.cols-fdet_rect.x);
	fdet_rect.height = std::min(int(det_rect[3]-det_rect[1]), img.rows-fdet_rect.y);
	face_align->align_face(img,fdet_rect,face_landmark,aligned_crop);
	//get feature
	gender_result=-1;
	gender_judge->getGender(aligned_crop,gender_result);
	
	if(gender_result==-1)
	{
		#if __ANDROID__
		__android_log_print(ANDROID_LOG_INFO,"perfxlab--","get gender fail!");
		__android_log_print(ANDROID_LOG_INFO,"perfxlab--","face feature length should be 0 or 1");
		#endif
		std::cout<<"get gender fail!"<<std::endl;
		std::cout<<"face gender should be 0 or 1"<<std::endl;	
		return 1;
	}
	
	return 0;	
}

int face_attr::make_border(cv::Mat & src_mat, cv::Mat & dst_mat)
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
