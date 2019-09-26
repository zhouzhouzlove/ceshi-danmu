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
#include "utils.hpp"

perftimer timer;

//#define FEATURE_LENGTH 128
#define FEATURE_LENGTH 512
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

int match_face(face_veri * face_verifier, cv::Mat img,std::map<std::vector<float> ,std::string> & registered_feature, std::vector<std::vector<float> > & register_face_rect, std::vector<std::vector<float> > & black_list_rect)
{
	//验证人脸,在同一张照片中可以有多张人脸，
        std::vector<std::vector<float> > detect_face_rect_vec;
        detect_face_rect_vec.clear();
        face_verifier->detect_face(img, detect_face_rect_vec);
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
                        if(score>0.5)
                        {
                                std::cout<<"-- "<<it->second<<" match successfully"<<std::endl;
				register_face_rect.push_back(detect_face_rect_vec[i]);
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
void Usage()
{
	std::cout<<"Usage: "<<"entrancd_guard "<<" -m [model dir] -d [database dir] -i [input image dir] -v"<<std::endl;
	return;
}

int main(int argc, char *argv[])
{

	std::string model_dir;
	std::string database_dir;
	std::string input_image_dir;
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
                        std::cout<<"input image "<<complete_image_path<<std::endl;
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
			std::string face_database_path(database_dir);
			std::string feature_file_name = it->second + ".txt";
			std::string feature_file = face_database_path.append("/");
			feature_file = face_database_path.append(feature_file_name);
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
		gpio_int();
		int i = 1;
		int count=0;
		int state=0;
		while(1)
		{
			while(1)
			{
				i=0;
				//std::cout<<"read signal!"<<std::endl;
				i= gpio_read(42);
				if(i==0&&state==0)
				{
					count++;
					state=1;
					std::cout<<"there is somebody!"<<std::endl;
					break;
				}
				if(i==1 && state==1)
				{
					state=0;
					std::cout<<"there is nobody!"<<std::endl;
					continue;

				}
			}
			std::vector<std::vector<float> > register_face_rect;//存储注册成功的人脸位置的四点坐标
			std::vector<std::vector<float> > black_list_rect;//存储注册失败的人脸位置的四点坐标
			std::vector<std::string> face_feature_files;
			register_face_rect.clear();
			black_list_rect.clear();
			face_feature_files.clear();
			get_files(database_dir,face_feature_files);
			int i=0;
			for(int i=0; i<face_feature_files.size(); ++i)
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
		
			cv::VideoCapture capture(1);
			cv::Mat input_img;
			int  run_veri_flag=0;
			//std::cout<<"input 's' or 'S' to stop verify"<<std::endl;
			//char stop_flag;
			while(run_veri_flag<30)
			{
				capture>>input_img;
				int match_result=match_face(face_verifier,input_img,register_feature, register_face_rect, black_list_rect);
				++run_veri_flag;
				if(match_result!=0)
				{
					std::cout<<"fail to match face!"<<std::endl;
					continue;
				}
				gpio_out(0);
				std::cout<<"open the door!"<<std::endl;	
				break;
				
			}
			
		
		}
	return 0;
	}
}
