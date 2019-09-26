#ifndef FACE_DETECTOR_HPP_
#define FACE_DETECTOR_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using std::string;
using std::vector;

#define WIDTH_HEIGHT_RATIO 1.25
#define MAKE_BORDER_CRITERION 0.05
class FaceDetect;
class FaceQuality;
class face_detector{
	public:
		~face_detector();
		static face_detector *get();
		static face_detector *get(const std::string &model_dir);
		static std::string get_model_dir();
		float get_score(const std::vector<float> & feature1,const std::vector<float> & feature_vec2);
		int get_feature(cv::Mat img,std::vector<float> & face_rect, std::vector<float> & face_feature);
		int detect_face(cv::Mat img, std::vector<std::vector<float> > & face_rect_vector);	
		void detect_face_dualcamera(cv::Mat visible_img, cv::Mat &infared_img, vector<vector<float> > & face_rects);
		int judge_color(cv::Mat input_img,std::vector<float> det_rect);
	private:
		static face_detector * face_detector_;
		static std::string model_dir_;
		std::vector<std::vector<float> > register_feature_;	
		face_detector(const std::string & model_dir);
		int make_border(cv::Mat & src_mat, cv::Mat & dst_mat);
		FaceDetect *face_detect;	
		FaceQuality *face_quality;
		std::vector<std::vector<float> > last_times_face_rects_vector;
		int detect_flag;
                int quality_result;
                std::vector<float> quality_probs;	
};

#endif //FACE_VERI_HPP_	
