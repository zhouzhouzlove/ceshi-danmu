#ifndef FACE_ATTR_HPP_
#define FACE_ATTR_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using std::string;
using std::vector;

#define WIDTH_HEIGHT_RATIO 1.25
#define MAKE_BORDER_CRITERION 0.05

class FaceDetect;
class Landmark;
class Align;
class GenderJudge;
class face_attr{
	public:
		~face_attr();
		static face_attr *get();
		static face_attr *get(const std::string &model_dir);
		static std::string get_model_dir();
		int get_gender(cv::Mat img,std::vector<float> & face_rect,int & gender_result);
		int detect_face(cv::Mat img, std::vector<std::vector<float> > & face_rect_vector);	
	private:
		static face_attr * face_attr_;
		static std::string model_dir_;
		std::vector<std::vector<float> > register_feature_;	
		face_attr(const std::string & model_dir);
		int make_border(cv::Mat & src_mat, cv::Mat & dst_mat);
		FaceDetect *face_detector;	
		Landmark *landmark_detector;
		Align *face_align;
		GenderJudge *gender_judge;
		
};

#endif //FACE_VERI_HPP_	
