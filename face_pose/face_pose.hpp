#ifndef FACE_POSE_HPP
#define FACE_POSE_HPP

#include<iostream>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>


class FaceDetect;
class Landmark;
enum pose_val {NONE,UP,DOWN, SHAKE_LEFT, SHAKE_RIGHT, TURN_LEFT, TURN_RIGHT, EYE_OPEN, EYE_CLOSED};
class face_pose{
	public:
		~face_pose();
                face_pose();
                face_pose(const std::string & model_dir);
		enum pose_val get_pose(cv::Mat  input_img);
		enum pose_val calculate_pose(const cv::Mat &input_img, const std::vector<cv::Point2f> & landmark_points);
		void calculate_euler_angle(const cv::Mat &rotate_vector,double &pitch, double &yaw, double &roll);
		void init_pose_count();
		enum pose_val judge_pose(double yaw,double pitch,double roll);
	private:
		FaceDetect *face_detector;
		Landmark * landmark_detector;
		int pose_count[9] = {0,0,0,0,0,0,0,0,0};
		float model_points_val[15] = {0.0, 0.0, 0.0, 
                                          -165.0, 170.0, -135.0, 
                                          165.0, 170.0, -135.0, 
					  -150.0, -150.0, -125.0, 
                                          150.0, -150.0, -125.0
					 };
		
		
};
#endif
