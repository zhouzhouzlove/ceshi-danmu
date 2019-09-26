#include<iostream>
#include<string>
#include "face_detector.hpp"
#include "face_veri.hpp"
#include "face_attr.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <time.h>
#include <map>
#include "utils.hpp"
#include <sstream>

#define FEATURE_LENGTH 512
using namespace cv;
perftimer timer;
int register_face(face_detector *face_detector_, face_veri *face_verifier, face_attr *face_attrer, cv::Mat img,std::map<std::vector<float>, std::string > & feature_map)
{
	std::vector<std::vector<float> >  face_rect_vec;
        face_rect_vec.clear();
	//检测人脸
	face_detector_->detect_face(img, face_rect_vec);
        if(face_rect_vec.size()==0)
	{
		std::cout<<"no face have been detected!"<<std::endl;	
		return 1;
	}
	//判断性别
	int result_flag;
	int gender_result=-1;
	result_flag=face_attrer->get_gender(img,face_rect_vec[0],gender_result);
	if(result_flag!=0)
	{
		std::cout<<"get gender result fail!"<<std::endl;
		return (1);
	}
	std::cout<<"0: female, 1: male"<<std::endl;
	std::cout<<"gender result: "<<gender_result<<std::endl;
	if(face_rect_vec.size()==0)
	{
		std::cout<<"no face have been deteced!"<<std::endl;
		return(-1);
	}
        //在注册人脸是保证图片中只有一张人脸
        std::vector<float> feature;
	feature.clear();
	face_verifier->get_feature(img,face_rect_vec[0],feature);
	if(feature.size()!=FEATURE_LENGTH)
	{
		std::cout<<"fail to get feature!"<<std::endl;
		return(-2);
	}
	//提取完人脸信息之后，输入自己个人信息
	std::string register_name;
	std::cout<<"please input your name: "<<std::endl;
	//std::cin>>register_name;
        register_name ="zhangbin";
	std::cout<<"Your name is: "<<register_name<<std::endl;
	feature_map.insert(std::make_pair(feature,register_name));
	return 0;
}
int match_face(face_veri * face_verifier, cv::Mat img,std::map<std::vector<float> ,std::string> & registered_feature, std::vector<std::vector<float> > & register_face_rect, std::vector<std::vector<float> > & black_list_rect)
{
	//验证人脸,在同一张照片中可以有多张人脸，
        std::vector<std::vector<float> > detect_face_rect_vec;
        detect_face_rect_vec.clear();
        face_verifier->detect_face(img, detect_face_rect_vec);
        timer.time_turn("-------------------------------------------- match_face: detect_face");
        if(detect_face_rect_vec.size()==0)
        {
                std::cout<<"no face has been deteced!"<<std::endl;
                return -1;
        }
        for(int i=0; i<detect_face_rect_vec.size(); ++i)
        {
                std::vector<float> feature;
                feature.clear();
                face_verifier->get_feature(img,detect_face_rect_vec[i],feature);
        	timer.time_turn("------------------------------------------------- match_face:  get_feature");
                if(feature.size()!=FEATURE_LENGTH)
                {
                        std::cout<<"fail to get feature!"<<std::endl;
                        return -1;
                }
                float score=0.0f;
                std::map<std::vector<float>, std::string>::iterator it;
                for(it=registered_feature.begin(); it!=registered_feature.end(); ++it)
                {
                        score = face_verifier->get_score(feature,it->first);
                        std::cout<<"the score is "<<score<<std::endl;
                        if(score>0.5)
                        {
                                std::cout<<"-- "<<it->second<<" register successfully"<<std::endl;
				register_face_rect.push_back(detect_face_rect_vec[i]);
                        }
                        else
                        {
                                std::cout<<"This guy is dangerous,doesn't belong to our company!"<<std::endl;
				black_list_rect.push_back(detect_face_rect_vec[i]);
                                //把这个人脸框的位置信息保存到黑名单数据库
                        }
                }
        }

        	timer.time_turn("------------------------------------------------- match_face: after get_feature");
	return 0;

}

int main(int argc, char *argv[])
{

	if(argc != 4)
	{
		std::cout<<"Usage: "<<argv[0]<<" [model dir] [input_img1] [input_img2] "<<std::endl;
		return -1;
	}
	std::string model_dir, input_img_1,input_img_2;
	model_dir = argv[1];
	input_img_1 = argv[2];
	input_img_2 = argv[3];
	std::cout<<"img1: "<<input_img_1<<"  img2: "<<input_img_2<<std::endl;	
	//read image
	cv::Mat img1 = cv::imread(input_img_1);	
	cv::Mat img2 = cv::imread(input_img_2);
	if(img1.empty() || img2.empty())
	{
		std::cout<<"images don't exist!"<<std::endl;
		return -1;
	}
	std::cout<<"Get face_detector: "<<std::endl;
	face_detector * face_detector_ = face_detector::get(model_dir);

	std::cout<<"Get face_verifier: "<<std::endl;
	face_veri * face_verifier = face_veri::get(model_dir);
	std::cout<<"Get face_attr: "<<std::endl;
	face_attr * face_attrer = face_attr::get(model_dir);


	//注册的人脸feature数据应该保存到数据库，本demo暂时保存到vector
	std::map<std::vector<float>, std::string> register_feature;
        int register_result;
        int i;
        for(i=0; i<1; ++i)
        {
		register_result = register_face(face_detector_,face_verifier,face_attrer,img1,register_feature);
	}
	if(register_result!=0)
	{
		std::cout<<"fail to register face!"<<std::endl;
		return -1;
	}


	//比对人脸
	std::vector<std::vector<float> > register_face_rect;//存储注册成功的人脸位置的四点坐标
	std::vector<std::vector<float> > black_list_rect;//存储注册失败的人脸位置的四点坐标
	register_face_rect.clear();
	black_list_rect.clear();

	
	timer.start();
        printf("\n\n\n------------------------------------------------------------\n");
	int match_result=match_face(face_verifier,img2,register_feature, register_face_rect, black_list_rect);
	timer.time_end("------------------ match_face");
	if(match_result!=0)
	{
		std::cout<<"fail to match face!"<<std::endl;
		return -1;
	}
	return 0;
}
