#include<iostream>
#include<string>
#include "face_detector.hpp"
#include "face_veri.hpp"
#include "face_attr.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <time.h>
#include <map>
#include <getopt.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <fstream>
#include "per_gpio.h"
#include <pthread.h>
#include <sstream>
#define FEATURE_LENGTH 512
#include "utils.hpp"

perftimer timer;
typedef struct ThreadArgs{

	cv::Mat *input_img;
	cv::Mat *visible_img;
	cv::Mat *infared_img;	
	face_veri * face_verifier;
	face_detector * face_detector_;
	std::map<std::vector<float>, std::string> * register_feature;
	std::map<std::vector<float>, std::string> * register_face_rect;
	std::vector<std::vector<float> > * black_list_rect;
	pthread_mutex_t thread_work_mutex;
	bool isWaiting;
}ThreadArgs;

std::vector<std::string> split(std::string str,std::string pattern)
{
  std::string::size_type pos;
  std::vector<std::string> result;
  str+=pattern;//扩展字符串以方便操作
  int size=str.size();

  for(int i=0; i<size; i++)
  {
    pos=str.find(pattern,i);
    if(pos<size)
    {
      std::string s=str.substr(i,pos-i);
      result.push_back(s);
      i=pos+pattern.size()-1;
    }
  }
  return result;
}
void get_files(std::string path, std::vector<std::string>& file_names)
{
	std::cout<<"image path :"<<path<<std::endl;
	DIR *dir;
        struct dirent *ptr;
        dir = opendir(path.c_str());
        int i=0;
        while((ptr =readdir(dir))!=nullptr)
        {
                std::cout<<"file_name "<<ptr->d_name<<std::endl;
                std::string file_name = ptr->d_name;
                if(file_name !="." && file_name != "..")
                {
                        file_names.push_back(file_name);
                }
                ++i;
                if(i>100)
                break;

        }
        closedir(dir);
        return;
}


int register_face(face_veri * face_verifier, cv::Mat img,std::string &register_name, std::map<std::vector<float>, std::string > & feature_map)
{
        std::vector<std::vector<float> >  face_rect_vec;
        face_rect_vec.clear();
        //检测人脸
        face_verifier->detect_face(img, face_rect_vec);
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
	if(register_name.empty())
        {
		std::cout<<"please input your name: "<<std::endl;
        	std::cin>>register_name;
        	std::cout<<"Your name is: "<<register_name<<std::endl;
	}
        feature_map.insert(std::make_pair(feature,register_name));
        return 0;
}
int match_face(face_veri * face_verifier, cv::Mat img,std::map<std::vector<float> ,std::string> & registered_feature, std::map<std::vector<float>,std::string > & register_face_rect, std::vector<std::vector<float> > & black_list_rect)
{
	//验证人脸,在同一张照片中可以有多张人脸，
        std::vector<std::vector<float> > detect_face_rect_vec;
        detect_face_rect_vec.clear();
        face_verifier->detect_face(img, detect_face_rect_vec);
        if(detect_face_rect_vec.size()==0)
        {
		register_face_rect.clear();
                std::cout<<"no face has been deteced!"<<std::endl;
                return -1;
        }
        for(int i=0; i<detect_face_rect_vec.size(); ++i)
        {
                std::vector<float> feature;
                feature.clear();
                face_verifier->get_feature(img,detect_face_rect_vec[i],feature);
                if(feature.size()!=FEATURE_LENGTH)
                {
                        std::cout<<"fail to get feature!"<<std::endl;
                        return -1;
                }
                float score=0.0f;
                std::map<std::vector<float>, std::string>::iterator it;
		register_face_rect.clear();
                for(it=registered_feature.begin(); it!=registered_feature.end(); ++it)
                {
                        score = face_verifier->get_score(feature,it->first);
                        if(score>0.5)
                        {
                                std::cout<<"-- "<<it->second<<" match successfully"<<std::endl;
				register_face_rect.insert(std::make_pair(detect_face_rect_vec[i],it->second));
				return 0;
                        }
                        else
                        {
                                std::cout<<"This guy is dangerous,doesn't belong to our company!"<<std::endl;
				black_list_rect.push_back(detect_face_rect_vec[i]);
                                //把这个人脸框的位置信息保存到黑名单数据库
				continue; 
                        }
                }
        }
	return 1;

}
/*
int match_face(face_detector * face_detector_, face_veri * face_verifier, cv::Mat img,cv::Mat &infared_img, std::map<std::vector<float> ,std::string> & registered_feature, std::map<std::vector<float>,std::string > & register_face_rect, std::vector<std::vector<float> > & black_list_rect)
{
	//验证人脸,在同一张照片中可以有多张人脸，
        std::vector<std::vector<float> > detect_face_rect_vec;
        detect_face_rect_vec.clear();
	std::vector<std::vector<float> > infared_detect_face_vec;
	infared_detect_face_vec.clear();
        face_verifier->detect_face(infared_img, infared_detect_face_vec);
	std::cout<<"infared image face num: "<<infared_detect_face_vec.size()<<std::endl;
        face_detector_->detect_face_dualcamera(img,infared_img,detect_face_rect_vec);
        if(detect_face_rect_vec.size()==0)
        {
		register_face_rect.clear();
                std::cout<<"no face has been deteced!"<<std::endl;
                return -1;
        }
        for(int i=0; i<detect_face_rect_vec.size(); ++i)
        {
                std::vector<float> feature;
                feature.clear();
                face_verifier->get_feature(img,detect_face_rect_vec[i],feature);
                if(feature.size()!=FEATURE_LENGTH)
                {
                        std::cout<<"fail to get feature!"<<std::endl;
                        return -1;
                }
                float score=0.0f;
                std::map<std::vector<float>, std::string>::iterator it;
		register_face_rect.clear();
                for(it=registered_feature.begin(); it!=registered_feature.end(); ++it)
                {
                        score = face_verifier->get_score(feature,it->first);
                        if(score>0.5)
                        {
                                std::cout<<"-- "<<it->second<<" match successfully"<<std::endl;
				register_face_rect.insert(std::make_pair(detect_face_rect_vec[i],it->second));
				return 0;
                        }
                        else
                        {
                                std::cout<<"This guy is dangerous,doesn't belong to our company!"<<std::endl;
				black_list_rect.push_back(detect_face_rect_vec[i]);
                                //把这个人脸框的位置信息保存到黑名单数据库
				return 1; 
                        }
                }
        }
	return 1;

}
z*/
int mark_match_result(cv::Mat & input_img, std::map<std::vector<float>, std::string> & match_result)
{
	std::map<std::vector<float>, std::string>::iterator iter;
	for(iter=match_result.begin(); iter!=match_result.end(); ++iter)
	{
		//rectangle(input_img,cv::Point(iter->first[0],iter->first[1]),
	//j		cv::Point(iter->first[2],iter->first[3]),
	//		cv::Scalar(0,0,255),3,4,0);
		int font_face = cv::FONT_HERSHEY_COMPLEX; 
		double font_scale = 2;
		int thickness = 2;
		int baseline;
		cv::Size text_size = cv::getTextSize(iter->second, font_face, font_scale, thickness, &baseline);
		cv::Point origin(iter->first[0], iter->first[3]);
                std::cout<<"mark name :" <<iter->second<<std::endl;
		std::string note_info = iter->second; 
		cv::putText(input_img, note_info, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 12, 0);

	}
	match_result.clear();
	return 0;
}
void Usage()
{
	std::cout<<"Usage: "<<"entrancd_guard "<<" -m [model dir] -d [database dir] -i [input image dir/input image file] -v"<<std::endl;
	return;
}

void * match_face_thread(void * arg)
{
	ThreadArgs * thread_args = (ThreadArgs*) arg;
	pthread_detach(pthread_self());
	while(1)
	{
		//std::cout<<"this is match face thread!"<<std::endl;
		usleep(1);
		if(thread_args->isWaiting == false || thread_args -> input_img == nullptr || thread_args -> infared_img == nullptr)
		{

			//if(thread_args->input_img == nullptr)
			//	std::cout<<"input img is nullptr"<<std::endl;
			continue;
		}
		cv::Mat input_img;
		cv::Mat infared_img;
		thread_args->infared_img->copyTo(infared_img);
		thread_args->input_img->copyTo(input_img);
		std::vector<std::vector<float> > black_list;
		std::vector<std::vector<float> > face_rects;
		
		//thread_args->register_face_rect->clear();
		//thread_args->face_detector_->detect_face_dualcamera(input_img, infared_img,face_rects); 
		int match_result =  match_face(thread_args->face_verifier,input_img,*(thread_args->register_feature),*(thread_args->register_face_rect),*(thread_args->black_list_rect));
		//int match_result=match_face(thread_args->face_detector_, thread_args->face_verifier,input_img,infared_img,*(thread_args->register_feature),*(thread_args->register_face_rect),*(thread_args->black_list_rect));
		thread_args->isWaiting = false;
	}
}	

int main(int argc, char *argv[])
{

	std::string model_dir;
	std::string database_dir;
	std::string input_image_dir;
        std::string input_image_file;
	char * mode_flag;
	bool veri_flag=false;
	int result;
	while((result=getopt(argc,argv,"m:d:i:v"))!=-1)
	{
		switch(result)
		{
			case 'm':
				model_dir = optarg;
				break;
			case 'd':
				database_dir = optarg;
				break;
			case 'i':
				input_image_dir = optarg;
				break;
			case 'v':
				veri_flag=true;
				input_image_file = input_image_dir;
				break;
			default :
				break;	
		}
			
	}	
	if(model_dir.empty() || database_dir.empty())
	{
		Usage();
		return 1;
	}
	if(veri_flag)
	{
		std::cout<<"-----verify-----"<<std::endl;
	}	
	else
	{
		std::cout<<"-----register-----"<<std::endl;
	}
	std::cout<<"model_dir: "<<model_dir<<std::endl;
	std::cout<<"database_dir "<<database_dir<<std::endl;
	std::cout<<"input_image_dir "<<input_image_dir<<std::endl;
	std::cout<<"Get face_verifier: "<<std::endl;
	face_veri * face_verifier = face_veri::get(model_dir);
	face_detector * face_detector_ = face_detector::get(model_dir);
	//注册的人脸feature数据应该保存到数据库，本demo暂时保存到vector
	std::map<std::vector<float>, std::string> register_feature;
	if(!veri_flag)
	{
		std::cout<<"register "<<std::endl;
		veri_flag =false;
		std::vector<std::string> input_image_files;
		input_image_files.clear();
		get_files(input_image_dir,input_image_files);
		int i;
		for(i=0; i<input_image_files.size();++i)
		{
			std::string complete_image_path(input_image_dir);
			complete_image_path = complete_image_path.append(input_image_files[i]);
			std::cout<<" complete_image_path: "<<complete_image_path<<std::endl; 
			cv::Mat img=cv::imread(complete_image_path,1);
			std::vector<std::string> input_file_split= split(input_image_files[i], ".");	
			int register_result = register_face(face_verifier,img,input_file_split[0],register_feature);
			if(register_result!=0)
			{
				std::cout<<input_image_files[i]<<" fail to register face!"<<std::endl;
				continue;
			}
		}
		std::map<std::vector<float>, std::string>::const_iterator it;
		for(it =register_feature.begin();it!=register_feature.end(); ++it)
		{
			std::string database_path(database_dir);
			std::string feature_file_name = it->second + ".txt";
			std::string feature_file = database_path.append("/");
			feature_file = feature_file.append(feature_file_name);
			std::ofstream fout;
			fout.open(feature_file);
			if(!fout)
			{
				std::cout<<"open feature file ["<<feature_file<<"] fail!"<<std::endl;
				return 1;
			}
			int j;
			for(j=0; j<it->first.size();++j)
			{
				fout<<it->first[j]<<"\n";
			}
			fout.close();
		}	
		return 0;
	}
	if(veri_flag)
	{
		//比对人脸
		std::map<std::vector<float>,std::string> register_face_rect;//存储注册成功的人脸位置的四点坐标
		std::map<std::vector<float>,std::string> register_face_rect_tmp;//存储注册成功的人脸位置的四点坐标
		std::vector<std::vector<float> > black_list_rect;//存储注册失败的人脸位置的四点坐标
		std::vector<std::string> face_feature_files;
		register_face_rect.clear();
		register_face_rect_tmp.clear();
		black_list_rect.clear();
		face_feature_files.clear();
		get_files(database_dir,face_feature_files);

		//人脸匹配线程初始化
		pthread_t thread_id;
		ThreadArgs thread_args;
		thread_args.face_verifier = face_verifier;
		thread_args.face_detector_ = face_detector_;
		thread_args.input_img = nullptr;
		thread_args.visible_img = nullptr;
		thread_args.infared_img = nullptr;
		thread_args.register_feature = &register_feature;
		thread_args.register_face_rect = &register_face_rect;
		thread_args.black_list_rect = &black_list_rect;
		thread_args.isWaiting = false;
		pthread_mutex_init(&(thread_args.thread_work_mutex),nullptr);
		pthread_create(&thread_id,nullptr, match_face_thread, (void*)&thread_args);
		//人脸匹配线程初始化
		int i;
		for(i=0; i<face_feature_files.size(); ++i)
		{
			std::vector<float> feature_vec;
			feature_vec.clear();
			std::string  feature_file(database_dir);
			feature_file = feature_file.append(face_feature_files[i]);
			std::ifstream fin(feature_file);
			if(!fin)
			{
				std::cout<<"open feature file ["<<feature_file<<"] fail!"<<std::endl;
				return 1;
			}
			while(!fin.eof())
			{
				float feature_data;
				fin>>feature_data;
				feature_vec.push_back(feature_data);
			}
			std::vector<std::string> feature_split_tmp = split(face_feature_files[i],".");	
			register_feature.insert(std::make_pair(feature_vec,feature_split_tmp[0]));

		}
		//get input image 
		cv::VideoCapture capture;
		cv::VideoCapture capture1;
                capture.open(0);
                //capture1.open(1);
		cv::Mat input_img,visible_img, infared_img;
		cv::Mat tmp_img, tmp_visible, tmp_infared;
		if(capture.isOpened())
		{
			std::cout<<"open camera successfully!"<<std::endl;
		}
		else
		{
			std::cout<<"fail to open camera!"<<std::endl;
			exit(1);
		}
                int count =0;
                while(1)		
		{   
		        count++; 
			capture >> input_img;
			//capture1 >> infared_img;
			if(input_img.empty())
			{
				std::cout<<"capture is null!"<<std::endl;
				continue;
			}
			//input_img = cv::imread(input_image_file,1);
			if(thread_args.isWaiting == true)
			{
				//do nothing
			}	
			else
			{
				input_img.copyTo(tmp_img);
				infared_img.copyTo(tmp_infared);
				thread_args.input_img = &tmp_img;
				thread_args.infared_img = &tmp_infared;
				thread_args.isWaiting = true;
				register_face_rect_tmp.clear();
				if(!register_face_rect.empty())
				{
					register_face_rect_tmp.insert(register_face_rect.begin(), register_face_rect.end());
				}	
				//imwrite("infared_img.jpg",*(thread_args.infared_img));
			}
			mark_match_result(input_img, register_face_rect_tmp);
			cv::imshow( "camera1", input_img);
			cv::waitKey(30);	
		//	if(count==20) break;	
		}			
		return 0;
	}
}
