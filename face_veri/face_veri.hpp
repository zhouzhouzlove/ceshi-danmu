#ifndef FACE_VERI_HPP_
#define FACE_VERI_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using std::string;
using std::vector;

#define WIDTH_HEIGHT_RATIO 1.25
#define MAKE_BORDER_CRITERION 0.05
//class in utils//
class FaceDetect;
class Landmark;
class Align;
class FaceQuality;
class Recog;
//class in utils//

class face_veri{
	public:
		~face_veri();
		static face_veri *get();
		static face_veri *get(const std::string &model_dir);
		static std::string get_model_dir();
		float get_score(const std::vector<float> & feature1,const std::vector<float> & feature_vec2);
		int get_feature(cv::Mat img,std::vector<float> & face_rect, std::vector<float> & face_feature);
		int detect_face(cv::Mat img, std::vector<std::vector<float> > & face_rect_vector);	
	private:
		static face_veri * face_veri_;
		static std::string model_dir_;
		std::vector<std::vector<float> > register_feature_;	
		face_veri(const std::string & model_dir);
		int make_border(cv::Mat & src_mat, cv::Mat & dst_mat);
		FaceDetect* face_detector;	
		Landmark* landmark_detector;
		Align* face_align;
		FaceQuality* face_quality;
		Recog* face_recog;
		int quality_result;
		
};

#endif //FACE_VERI_HPP_	
