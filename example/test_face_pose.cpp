#include<iostream>
#include<string>
#include "face_pose.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <time.h>
#include <map>
#include "utils.hpp"
#include <sstream>


int main(int argc, char *argv[])
{
	if(argc != 4)
	{
		std::cout<<"Usage: "<<argv[0]<<" [model dir] [pose_num]"<<std::endl;
		return -1;
	}
	std::string model_dir;
	std::string pose_num_str;
	std::string image_path;
	pose_num_str = argv[2];
	unsigned char  pose_num;
	std::stringstream ss;
	ss << pose_num_str;
	ss >> pose_num;	
	model_dir = argv[1];
	image_path = argv[3];
	face_pose * pose_detector = new face_pose(model_dir);
	cv::VideoCapture  capture;
	cv::Mat input_img;
	//test face pose	
	std::cout<<"-----test face pose-----"<<std::endl;
	std::cout<<"-----head down=========="<<std::endl;
	//capture.open(image_path);
	input_img = cv::imread(image_path);
	while(1)
	{
		//capture >>input_img;
		enum pose_val pose_result;
		pose_result=pose_detector->get_pose(input_img);
		std::cout<<"pose_result: "<<pose_result<<std::endl;
		printf("pose_num %hhu\n",pose_num);
		if(pose_result==(pose_num-48))
		{
			std::cout<<"test pose successfully!"<<std::endl;
			break; 
		}
		break;		
	}
	return 0;
}
