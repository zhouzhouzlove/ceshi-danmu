#include<iostream>
#include<string>
#include "face_detector.hpp"
#include "face_veri.hpp"
#include "face_attr.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <time.h>
#include <map>
#include <fstream>
#include <sys/types.h>
#include <sys/time.h>
#include <dirent.h>

//#define FEATURE_LENGTH 128
#define FEATURE_LENGTH 512

void get_files(std::string path, std::vector<std::string>& file_names)
{
        std::cout<<"image path :"<<path<<std::endl;
        DIR *dir;
        struct dirent *ptr;
        dir = opendir(path.c_str());
        int i=0;
        while((ptr =readdir(dir))!=NULL)
        {   
                std::cout<<"file_name "<<ptr->d_name<<std::endl;
                std::string file_name = ptr->d_name;
                if(file_name !="." && file_name != "..")
                {   
                        std::cout<<"file name :"<<file_name<<std::endl;
                        file_names.push_back(file_name);
                }   
                ++i;
                if(i>100)
                break;

        }   
        closedir(dir);
        return;
}

void detect_face_dualcamera(face_veri * face_verifier, cv::Mat visible_img, cv::Mat infared_img, vector<vector<float> > & face_rects);
int judge_color(cv::Mat input_img,std::vector<float> det_rect);

int register_face(face_detector *face_detector_, face_veri *face_verifier, face_attr *face_attrer, cv::Mat img, std::string register_name, std::map<std::vector<float>, std::string > & feature_map)
{
	std::vector<std::vector<float> >  face_rect_vec;
        face_rect_vec.clear();
	//检测人脸
	face_detector_->detect_face(img, face_rect_vec);
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
	//std::string register_name;
	//std::cout<<"please input your name: "<<std::endl;
	//std::cin>>register_name;
        //register_name ="zhangbin";
	//std::cout<<"Your name is: "<<register_name<<std::endl;
	feature_map.insert(std::make_pair(feature,register_name));
	return 0;
}

#define IS_INFARED_IMG (2)
#define IS_VISIBLE_IMG (3)
int check_color_mat(const cv::Mat &input){
    if(input.empty()){
        std::cout << "[check_color_mat] input mat is empty" << std::endl;
        return 1;
    }
    if(input.at<cv::Vec3b>(input.rows/2,input.cols/2)[0] ==
       input.at<cv::Vec3b>(input.rows/2,input.cols/2)[1] 
         && 
       input.at<cv::Vec3b>(input.rows/2,input.cols/2)[0] ==
       input.at<cv::Vec3b>(input.rows/2,input.cols/2)[2])
    {
        return IS_INFARED_IMG;
    } else
        return IS_VISIBLE_IMG;
}

//int match_face(face_veri * face_verifier, cv::Mat vis_img, cv::Mat infrared_img, std::map<std::vector<float> ,std::string> & registered_feature, std::vector<std::vector<float> > & register_face_rect, std::vector<std::vector<float> > & black_list_rect)
int match_face(face_veri * face_verifier, cv::Mat input_0, cv::Mat input_1, std::map<std::vector<float> ,std::string> & registered_feature, std::map<std::vector<float>, std::string> & register_name_and_face_rect, std::vector<std::vector<float> > & black_list_rect)
{
	if(input_0.empty() && !input_1.empty())
	{
		//pFaceManager->detect_face(input_1, face_rects);
		std::cout << "camera 0 image is empty,camera 1 image is normal!" << std::endl;
		exit(0);
	}
	if(input_1.empty() && !input_0.empty())
	{
		//pFaceManager->detect_face(input_0, face_rects);
		std::cout << "camera 1 image is empty,camera 0 image is normal!" << std::endl;
		exit(0);
	}
	vector<vector<float>> face_rects;
	cv::Mat visible_img;
	cv::Mat infared_img;

	if(check_color_mat(input_0)!=check_color_mat(input_1))
	{
		if (check_color_mat(input_0) == IS_VISIBLE_IMG &&
				check_color_mat(input_1) == IS_INFARED_IMG) 
		{
			visible_img = input_0;
			infared_img = input_1;
			detect_face_dualcamera(face_verifier, visible_img, infared_img, face_rects);
			std::cout << "camera 0 image is visible,camera 1 image is infared!" << std::endl;
		} 
		else
		{
			visible_img = input_1;
			infared_img = input_0;
			detect_face_dualcamera(face_verifier, visible_img, infared_img, face_rects);
			std::cout << "camera 1 image is visible,camera 0 image is infared!" << std::endl;
		}
	}
	else
	{
		std::cout << "these two camera have same color!" << std::endl;
		return -1;
	}
	// get live detected faces regions' features
	std::vector<std::vector<float>> face_rect_features(static_cast<int>(face_rects.size()));
	for(int i=0; i<static_cast<int>(face_rects.size()); ++i)
	{
		std::vector<float> rect = face_rects[i];
		face_verifier->get_feature(visible_img, face_rects[i], face_rect_features[i]);
	}

        std::cout << "----------- start compare similarity -----------------" << std::endl;
        std::cout << "registered_feature.size():" << registered_feature.size() << std::endl;
        std::cout << "face_rect_features.size():" << face_rect_features.size() << std::endl;
	// loop1 registered faces 
	std::map<std::vector<float>, std::string>::iterator riter;
	int ridx=0;
	for(riter = registered_feature.begin(); riter != registered_feature.end(); riter++)
	{
		std::vector<float> r_feat = riter->first;
		std::string name = riter->second;
		// loop2 detected live faces
		for(int vidx=0; vidx < static_cast<int>(face_rect_features.size()); ++vidx)
		{
			std::vector<float> v_feat = face_rect_features[vidx];
			float score = face_verifier->get_score(r_feat, v_feat);
			std::cout << "score: " << score;
			std::cout << " registered_idx: " << ridx << " vidx: " << vidx << " score: " << score << std::endl;
			if(score > 0.5)
			{
				std::cout << "register successfully guy " << name << " detected" << std::endl;
				//register_name_and_face_rect.push_back(face_rects[vidx]);
                                register_name_and_face_rect.insert(pair<std::vector<float>, std::string>(face_rects[vidx], name));
			}
			else // score <= 0.5
			{
				std::cout << "dangerous guy detected" << std::endl;
				black_list_rect.push_back(face_rects[vidx]);
			}
		}
		++ridx;
	}
	std::cout << "------------- finish compare similarity ----------------" << std::endl;
	return 0;
}

#define visible_detect_expand_ratio (0.2)
void detect_face_dualcamera(face_veri * face_verifier, cv::Mat visible_img, cv::Mat infared_img, vector<vector<float> > & face_rects)
{
	std::vector<std::vector<float> > visible_detect_result;
	std::vector<std::vector<float> > infared_detect_result;
	visible_detect_result.clear();
	face_verifier->detect_face(visible_img, visible_detect_result);
	if(visible_detect_result.size() != 0)
	{   
		int color_result = judge_color(visible_img, visible_detect_result[0]);
		if(color_result == 1)
		{   
			return;
		}   
    }   
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
        if(infared_rect.x<0 || infared_rect.y<0 || infared_rect.width<0 || infared_rect.height<0 ||
           infared_rect.x>=infared_rect.width || infared_rect.y>=infared_rect.height)
        {
            std::cout << "no face found for " << i << "th visible_detect_result" << std::endl;
            continue;
        }
        cv::Mat infared_input=infared_img(infared_rect);
        infared_detect_result.clear();
        face_verifier->detect_face(infared_input,infared_detect_result);
        if(infared_detect_result.size() != 0 )
        {
           face_rects.push_back(visible_detect_result[i]);
        }
    }
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

int judge_color(cv::Mat input_img,std::vector<float> det_rect)
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


int main(int argc, char *argv[])
{

	if(argc != 5)
	{
		std::cout<<"Usage: "<<argv[0]<<" [model dir] [resgister_img_path] [input_img2] [input_img3]"<<std::endl;
		return -1;
	}
	std::string model_dir,path,input_img_2,input_img_3;
	model_dir = argv[1];
	path = argv[2];
	input_img_2 = argv[3];
	input_img_3 = argv[4];

        std::cout<< "register_img_path:" << path << std::endl;
	std::cout<< "img2: "<<input_img_2<<"  img3: "<<input_img_3<<std::endl;	
	//read image
	cv::Mat img2 = cv::imread(input_img_2);
        cv::Mat img3 = cv::imread(input_img_3);
	if(img2.empty() || img3.empty())
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

        ////////////////////// register start ///////////////////
        //std::string path = "./register";
        std::vector<std::string> file_names;
        std::vector<std::string> file_paths;
        get_files(path, file_names);
        for(int fidx=0; fidx<static_cast<int>(file_names.size()); ++fidx)
        {
            std::string file_path = path + "/" + file_names[fidx];
            file_paths.emplace_back(file_path);
            //std::cout << "<<<" << file_path << std::endl;
            //std::cout << fidx << " " << file_names[fidx] << std::endl;
        }
	
	std::cout << "start register" << std::endl;
	std::map<std::vector<float>, std::string> register_feature;
	int register_result;
	for(int fidx=0; fidx<static_cast<int>(file_paths.size()); ++fidx)
	{
            cv::Mat register_img = cv::imread(file_paths[fidx]);
            if(register_img.empty())
            {
                std::cout << "[ERROR] image is empty. image path:" << file_paths[fidx] << std::endl;
                return -1;
            }
	    register_result = register_face(face_detector_, face_verifier, face_attrer, register_img, file_names[fidx], register_feature);
            if(register_result!=0)
            {
                std::cout << "[ERROR] fail to register face!" << std::endl;
		return -1;
            }
	}
        std::cout << "finished registered " << register_feature.size() << " people/images"  <<std::endl;      
        ////////////////////// register end ///////////////////

	cv::VideoCapture capture;
        //clock_t start, end;
        struct timeval start, end;
        double duration;
	while(true)
	{
		//比对人脸
		std::map<std::vector<float>, std::string> register_name_and_face_rect;//存储注册成功的人脸位置的四点坐标
		std::vector<std::vector<float> > black_list_rect;//存储注册失败的人脸位置的四点坐标
		register_name_and_face_rect.clear();
		black_list_rect.clear();
		capture.open(0);
		capture >> img2;
		capture.open(1);
		capture >> img3;

		if(img2.empty())
		{
			std::cout << "camera0 finish" << std::endl;
			break;
		}

		if(img3.empty())
		{
			std::cout << "camera1 finish" << std::endl;
			break;
		}
                gettimeofday(&start, NULL);
		int match_result=match_face(face_verifier,img2,img3,register_feature, register_name_and_face_rect, black_list_rect);
 	        gettimeofday(&end, NULL);
  	        if(match_result!=0)
		{
			std::cout<<"fail to match face!"<<std::endl;
			continue;
		}
                duration = (end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
                std::cout << "[INFO] per image time: "<< duration/1000. << "ms" << std::endl;
                for(auto &name_and_face_rect_iter : register_name_and_face_rect)
		{
			vector<float> face_rect = name_and_face_rect_iter.first;
                        std::string name        = name_and_face_rect_iter.second;

			rectangle( img2, cv::Point(face_rect[0], face_rect[1]),
					cv::Point(face_rect[2], face_rect[3]),
					cv::Scalar(0, 0, 255), 3, 4, 0 );
			rectangle( img3, cv::Point(face_rect[0], face_rect[1]),
					cv::Point(face_rect[2], face_rect[3]),
					cv::Scalar(0, 0, 255), 3, 4, 0 );

                        // draw text
			int font_face = cv::FONT_HERSHEY_COMPLEX; 
			double font_scale = 2;
			int thickness = 2;
			int baseline;
			cv::Size text_size = cv::getTextSize(name, font_face, font_scale, thickness, &baseline);
			cv::Point origin(face_rect[0], face_rect[3]);
			cv::putText(img3, name, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);
		}
		//cv::imshow( "camera0", img2 );
		cv::imshow( "camera1", img3 );
		cv::waitKey(30);
		//cv::imwrite( "result.png", img2 );
	}
	capture.release();
	return 0;
}
