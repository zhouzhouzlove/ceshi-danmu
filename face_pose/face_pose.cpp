#include "face_pose.hpp"
#include "face_detect.h"
#include "landmark.h"
#include <cmath>
#include <time.h>
#if __ANDROID__
#include <android/log.h>
#include <fcntl.h>
#include <unistd.h>
#endif


#define PITCH_UPPER_LIMIT 180
#define PITCH_LOWER_LIMIT 100
#define YAW_UPPER_LIMIT 70
#define YAW_LOWER_LIMIT 15
#define ROLL_UPPER_LIMIT 10
#define ROLL_LOWER_LIMIT 70

#define PI acos(-1)
#define JUDGE_PITCH_DOWN(pitch) ( - PITCH_UPPER_LIMIT < pitch && pitch < - PITCH_LOWER_LIMIT)
#define JUDGE_PITCH_UP(pitch) ( PITCH_LOWER_LIMIT < pitch && pitch < PITCH_UPPER_LIMIT)
#define JUDGE_YAW_LEFT(yaw) ( - YAW_UPPER_LIMIT < yaw && yaw < - YAW_LOWER_LIMIT)
#define JUDGE_YAW_RIGHT(yaw) ( YAW_LOWER_LIMIT < yaw && yaw < YAW_UPPER_LIMIT)
#define JUDGE_ROLL_LEFT(roll) ( ROLL_LOWER_LIMIT < roll && roll < ROLL_UPPER_LIMIT)
#define JUDGE_ROLL_RIGHT(roll) ( - ROLL_UPPER_LIMIT < roll && roll < - ROLL_LOWER_LIMIT)

//#define FEATURE_LENGTH 128
#if __ANDROID__
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, "perfxlab_face_pose", __VA_ARGS__);
#endif
face_pose::face_pose()
{
  //do nothing
}

face_pose::~face_pose() 
{ 
	delete face_detector; 
}

face_pose::face_pose(const std::string & model_dir)
{
	int string_length;
	string_length = model_dir.length();
	char model_path[string_length+1];
	for(int i=0; i < string_length; ++i)
		model_path[i] = model_dir[i];
	model_path[string_length] = '\0';
	
	face_detector = new FaceDetect();
	landmark_detector = new Landmark();
	
	face_detector->model_init(model_path, "FaceNetNew","cn_1_1_224_320");
	landmark_detector->model_init(model_path, "FaceLandmark","bn_1_1_224_320");
}
enum pose_val face_pose::get_pose(cv::Mat  input_img)
{
	std::vector<std::vector<float> > face_rect_vector;
	std::vector<cv::Point2f> landmark_points;
	face_rect_vector.clear();
	face_rect_vector = face_detector->detect(input_img);
	if(face_rect_vector.size()==0)
	{	
		#if __ANDROID__
		LOGE("no face has been detected!")
		#endif 
		std::cout<<"no face has been detected!"<<std::endl;
		return NONE;
	}	
	landmark_points = landmark_detector->detect(input_img,face_rect_vector[0]);
	for(int i = 0; i<landmark_points.size(); ++i)
	{
		landmark_points[i].x = landmark_points[i].x - face_rect_vector[0][0];
		landmark_points[i].y = landmark_points[i].y - face_rect_vector[0][1];
		std::cout<<"landmark_points["<<i<<"].x: "<<landmark_points[i].x<<std::endl;
		std::cout<<"landmark_points["<<i<<"].y: "<<landmark_points[i].y<<std::endl;
		//circle(input_img,landmark_points[i],4,cv::Scalar(255,0,0));
	
	}
	//cv::imwrite("test_pose_img.jpg",input_img);
	enum pose_val pose_result = calculate_pose(input_img, landmark_points);
	return pose_result;	
}

enum pose_val face_pose::calculate_pose(const cv::Mat & input_img, const std::vector<cv::Point2f> & landmark_points)
{
	float focal_length = float(input_img.cols);
	float center[2] ={(float)input_img.cols/2.0f,(float)input_img.rows/2.0f};
	//camera matrix mat data
	float camera_matrix_val[9]={focal_length,0.0,center[0],
                                       0.0,focal_length,center[1],
                                       0.0,0.0,1.0};
	//dist_coeffs mat data 
	float coeffs_val[4]={0.0,0.0,0.0,0.0};
	//image_points mat data
	float image_points_val[10] ={landmark_points[2].x,landmark_points[2].y,
                                       landmark_points[0].x,landmark_points[0].y, 	
                                       landmark_points[1].x,landmark_points[1].y, 	
                                       landmark_points[3].x,landmark_points[3].y, 	
                                       landmark_points[4].x,landmark_points[4].y, 
					};
	

	/*
	for(int i = 0; i<5;++i)
	{	
		for(int j=0; j<2;++j)
		{
			std::cout<<"image_points_val["<<i<<"]["<<j<<"]: "<<image_points_val[i*2+j]<<std::endl;
		}
	}
	for(int i = 0; i<3;++i)
	{	
		for(int j=0; j<3;++j)
		{
			std::cout<<"camera_matrix_val["<<i<<"]["<<j<<"]: "<<camera_matrix_val[i*3 + j]<<std::endl;
		}
	}
	*/
        cv::Mat camera_matrix(3,3,CV_32FC1,camera_matrix_val);
	cv::Mat dist_coeffs(4,1,CV_32FC1,coeffs_val);
	cv::Mat model_points(5,3,CV_32FC1,model_points_val);
	cv::Mat image_points(5,2,CV_32FC1,image_points_val);
	
	cv::Mat rotation_vector, translation_vector;
	bool result =cv::solvePnP(model_points,image_points,camera_matrix,dist_coeffs,rotation_vector,translation_vector,false,CV_ITERATIVE);
	double pitch =0.0f; 
	double yaw =0.0f; 
	double roll =0.0f; 
	calculate_euler_angle(rotation_vector,pitch,yaw,roll);	
	enum pose_val pose_result = judge_pose(pitch,yaw,roll);
	return  pose_result;					
}
void face_pose::init_pose_count()
{
	for(int i=0; i<9; ++i)
	{
		pose_count[i]=0;
	}
}
void face_pose::calculate_euler_angle(const cv::Mat &rotation_vector,double &pitch, double &yaw, double &roll)
{
	float theta = cv::norm(rotation_vector);
	std::cout<<"theta: "<<theta<<std::endl;
	#if __ANDROID__
	LOGE("theta: %f\b", theta);
	#endif
	/*
	std::cout<<"rotation_vector<0,0>: "<<rotation_vector.at<float>(0,0)<<std::endl;
	std::cout<<"rotation_vector<1,0>: "<<rotation_vector.at<float>(1,0)<<std::endl;
	std::cout<<"rotation_vector<2,0>: "<<rotation_vector.at<float>(2,0)<<std::endl;
	*/
	float w,x,y,z;
	w = cos(theta/2.0);
	x = sin(theta/2.0)*rotation_vector.at<float>(0,0) / theta;
	y = sin(theta/2.0)*rotation_vector.at<float>(1,0) / theta;
	z = sin(theta/2.0)*rotation_vector.at<float>(2,0) / theta;
	//calculate pitch
	pitch = atan2(2.0*(w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
	pitch = pitch * 180.0 / PI;
	//calculate yaw
	float sin_yaw = 2.0 * (w * y - z * x);
	if(sin_yaw > 1.0)
		sin_yaw = 1.0;
	if(sin_yaw < -1.0)
		sin_yaw = -1.0;
	yaw = asin(sin_yaw);
	yaw = yaw * 180.0 / PI;
	// calculate roll 
	roll = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 *(y * y + z * z));
	roll = roll * 180.0 / PI;
	
	if(roll > 90.0)
		roll = fmod((roll - 180.0), 180.0);	
	if(roll < -90.0)
		roll = fmod((roll + 180.0),180.0);

	return;	
} 
enum pose_val face_pose::judge_pose(double pitch, double yaw, double roll)
{
	std::cout<<"----- pitch, yaw, roll"<<std::endl;
	std::cout<<"pitch: "<<pitch<<std::endl; 		
	std::cout<<"yaw: "<<yaw<<std::endl; 		
	std::cout<<"roll: "<<roll<<std::endl; 		
	#if __ANDROID__
	LOGE("pitch: ",pitch);
	LOGE("yaw: ",yaw);
	LOGE("roll: ",roll);
	#endif
	if(JUDGE_PITCH_DOWN(pitch))
	{
		std::cout<<"pitch down"<<std::endl; 		
		pose_count[2]+=1;		
	}
	if(JUDGE_PITCH_UP(pitch))
	{
		std::cout<<"pitch up"<<std::endl; 		
		pose_count[1]+=1;		
	}
	if(JUDGE_YAW_LEFT(yaw))
	{
		pose_count[5]+=1;
	}	
	if(JUDGE_YAW_RIGHT(yaw))
	{
		pose_count[6]+=1;
	}
	if(JUDGE_ROLL_LEFT(roll))
	{
		pose_count[3]+=1;
	}
	if(JUDGE_ROLL_RIGHT(roll))
	{	
		pose_count[1]+=1;
	}
	//pitch 
	/*	
	if((!JUDGE_YAW_LEFT(yaw))&&(!JUDGE_YAW_RIGHT(yaw))&& (!JUDGE_ROLL_LEFT(roll)) && (!JUDGE_ROLL_RIGHT(roll)))
	{
		if(JUDGE_PITCH_DOWN(pitch))
		{
			std::cout<<"this is DOWN"<<std::endl;
			return DOWN;
		}
		else if(JUDGE_PITCH_UP(pitch));
			return UP;	
	}
	if((!JUDGE_PITCH_UP(pitch)) && (!JUDGE_PITCH_DOWN(pitch)) && (!JUDGE_ROLL_LEFT(roll)) && (!JUDGE_ROLL_RIGHT(roll)))	
	{
		if(JUDGE_YAW_LEFT(yaw))
			return TURN_LEFT;
		else if(JUDGE_YAW_RIGHT(yaw))
			return TURN_RIGHT;
	}
	if((!JUDGE_PITCH_UP(pitch)) && (!JUDGE_PITCH_DOWN(pitch)) && (!JUDGE_YAW_LEFT(yaw)) && (!JUDGE_YAW_RIGHT(yaw)))
	{
		if(JUDGE_ROLL_LEFT(roll))
			return TURN_LEFT;
		else if(JUDGE_ROLL_RIGHT(roll))
			return TURN_RIGHT;
	}*/
	int max_count_pose=-1;
	int max_pose=0;
	for(int i=0; i <9; ++i)
	{
		std::cout<<"pose_count["<<i<<"]: "<<pose_count[i]<<std::endl;
		if(pose_count[i]>max_count_pose)
		{
			max_count_pose=pose_count[i];
			max_pose=i;
		}		
	}
	std::cout<<"max pose"<<max_pose<<std::endl; 	
	return pose_val(max_pose);	
}
