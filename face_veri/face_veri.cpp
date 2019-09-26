#include "face_veri.hpp"
#include <time.h>
#if __ANDROID__
#include <android/log.h>
#include <fcntl.h>
#include <unistd.h>

#endif

#include "face_detect.h"
#include "landmark.h"
#include "align.h"
#include "face_quality.h"
#include "recog.h"
//#define FEATURE_LENGTH 128
#if __ANDROID__
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, "perfxlab_face_veri", __VA_ARGS__);
#endif

face_veri *face_veri::face_veri_ =0;
std::string face_veri::model_dir_="";

std::string face_veri::get_model_dir()
{
	return model_dir_;
}

face_veri *face_veri::get()
{
	return face_veri_;
}

face_veri *face_veri::get(const std::string & model_dir)
{

	//
	face_veri_ = new face_veri(model_dir);
	model_dir_ = model_dir;
	
	return face_veri_;
}


face_veri::face_veri(const std::string &model_dir)
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
	face_detector =new FaceDetect();
	landmark_detector = new Landmark();
	face_align = new Align();
	face_quality = new FaceQuality();
	face_recog = new Recog(); 
	face_detector->model_init(model_path, "FaceNetNew", "cn_1_1_224_320");
	//face_detector.model_init(model_path, "FaceNet", "cn_1_1_240_320");
	landmark_detector->model_init(model_path, "FaceLandmark", "bn_1_3_112_96"); 
	//face_recog.model_init(model_path, "CaffeMobileFaceNet", "an_1_3_112_96");
	//face_quality.model_init(model_path, "FaceQuality", "an_1_1_56_48");
	face_recog->model_init(model_path, "resnet84", "an_1_3_112_96");
	//face_recog.model_init(model_path, "SpherefaceNet20", "an_1_3_112_96");
	face_align->init();

    face_quality = new FaceQuality();
    //face_quality->model_init(model_path,"AlexNet","cn_1_1_112_96");
	#if __ANDROID__
	__android_log_print(ANDROID_LOG_INFO,"perfxlab--","complete loading model ");
	#endif
}
face_veri::~face_veri()
{
	delete face_detector;
	delete landmark_detector;
	delete face_align;
	delete face_quality;
	delete face_recog;
	delete face_veri_; 
}

int face_veri::detect_face(cv::Mat img, std::vector<std::vector<float> > & face_rect_vector)
{
	face_rect_vector = face_detector->detect(img);
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

int face_veri::get_feature(cv::Mat img,std::vector<float> & det_rect, std::vector<float> & face_feature)
{

	//get landmark points
	std::vector<cv::Point2f> landmark;
	landmark = landmark_detector->detect(img,det_rect);
//        timer.time_turn("********************************************************* getfeature: landmark->detect");
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
/*
	//face quality
 	std::vector<float>quality_probs;
	int quality =-1;
	face_quality.getQuality(aligned_crop, quality,quality_probs);
	if(quality!=1)
	{
		#if __ANDROID__
		LOGE("Input face is bad face,is not qualified to get feature!");
		#endif 
		std::cout<<"Input face is bad face,is not qualified to get feature!"<<std::endl;
		return 0;
	}
*/
	//get feature
//        timer.time_turn("********************************************************* get_feature: align_face");
    //face_quality->getQuality(aligned_crop,quality_result);
    //if(quality_result==1)
    //{
        std::cout<<"quality measure up to identify face"<<std::endl;
        face_recog->getfeature(aligned_crop,face_feature);
    //}
	if(face_feature.size()!=FEATURE_LENGTH)
	{
		#if __ANDROID__
		//__android_log_print(ANDROID_LOG_INFO,"perfxlab--","get feature fail!");
		//__android_log_print(ANDROID_LOG_INFO,"perfxlab--","face feature length should be %d",FEATURE_LENGTH);
		LOGE("get feature fail!");
		__android_log_print(ANDROID_LOG_INFO,"perfxlab--","face feature length should be %d",FEATURE_LENGTH);
		#endif
		std::cout<<"get feature fail!"<<std::endl;
		std::cout<<"face feature length should be "<<FEATURE_LENGTH<<std::endl;	
		return 0;
	}
	
	return 1;	
}

float  face_veri::get_score(const std::vector<float> & feature_vec1, const std::vector<float> & feature_vec2)
{
//	#if __ANDROID__
//	__android_log_print(ANDROID_LOG_INFO,"perfxlab--","get score start");
//	#endif
	
	if(feature_vec1.size()==0 || feature_vec2.size()==0)
	{
		std::cout<<"feature vectors are empty!"<<std::endl;
		#if __ANDROID__
		LOGE("feature vectors are empty!");
		#endif
		return 0.0; 
	}
	cv::Mat feature1(1,FEATURE_LENGTH,CV_32FC1);
	for(int i=0;i<FEATURE_LENGTH; ++i)
	{
		feature1.at<float>(0,i)=feature_vec1[i];
	}
	cv::Mat feature2(1,FEATURE_LENGTH,CV_32FC1);
	for(int i=0;i<FEATURE_LENGTH; ++i)
	{
		feature2.at<float>(0,i)=feature_vec2[i];
	}
	float feature1_length = sqrt(feature1.dot(feature1));
	float feature2_length = sqrt(feature2.dot(feature2));
	cv::Mat unit_feature1 = feature1/feature1_length;
	cv::Mat unit_feature2 = feature2/feature2_length;
	double score = unit_feature1.dot(unit_feature2);
	//score = 1.0/(1.0 + exp(-11.945*score + 4.97));
	score = 1.0/(1.0 + exp(-12.4*score + 3.763));
	#if __ANDROID__
	LOGE("score is %f",score);
	#endif
	return score;	
}

int face_veri::make_border(cv::Mat & src_mat, cv::Mat & dst_mat)
{
	if(src_mat.empty())
	{
		#if __ANDROID__
		LOGE("make_border input mat is null");
		#endif
		std::cout<<"make_border input mat is null!"<<std::endl;
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
