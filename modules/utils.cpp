#include "utils.hpp"
#include <stdio.h>
#include <string.h>
#include "opencv2/opencv.hpp"
#if __ANDROID__
#include <android/log.h>
#endif
using namespace cv;
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, "perfxlab_recog", __VA_ARGS__);
perftimer::perftimer(){
    memset(&begin,0,sizeof(struct timeval));
    memset(&end,0,sizeof(struct timeval)); 
    gettimeofday(&begin,NULL);
    gettimeofday(&tmp,NULL);
}

int perftimer::start()
{
    gettimeofday(&begin,NULL);
    gettimeofday(&tmp,NULL);
    return 0;
}

int perftimer::time_end(char * name)
{
    gettimeofday(&end,NULL);
    double tmp = ((double)((end.tv_sec - begin.tv_sec)*1000000 +(end.tv_usec - begin.tv_usec)))/1000; 
    int sec = (int)(tmp/1000);
    double ms = tmp -1000*sec;
    printf("%s runs %ds %fms\n", name, sec, ms);
    return 0;
}

int perftimer::time_turn(char * name)
{
    gettimeofday(&end,NULL);

    double tmpdata = ((double)(end.tv_sec - tmp.tv_sec)*1000000 +(end.tv_usec - tmp.tv_usec))/1000;
    
    int sec = tmpdata/1000;
    double ms = tmpdata -1000*sec;
    printf("%s runs %ds %fms\n", name, sec, ms);
    tmp.tv_sec = end.tv_sec;
    tmp.tv_usec = end.tv_usec;
    return 0;
}
perftimer::~perftimer(){}

int make_border(cv::Mat & src_mat, cv::Mat & dst_mat)
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
        //printf("cols:%d, rows:%d\n", dst_mat.cols, dst_mat.rows);
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
        //printf("cols:%d, rows:%d\n", dst_mat.cols, dst_mat.rows);
		//cv::imwrite("border_img.jpg",dst_mat);
		return 0;
	}
	//__android_log_print(ANDROID_LOG_INFO,"perfxlab--","don't need make border!");
	//src_mat.copyTo(dst_mat);	
	std::cout<<"input Mat ratio is right, no border is needed!"<<std::endl;
	return 0;	
}
