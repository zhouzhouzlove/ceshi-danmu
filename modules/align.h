
#ifndef _Align_H_
#define _Align_H_

#include "inferxlite.h"
#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#endif
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;
class Align
{
public:
	Align();
	virtual ~Align();
	void init();

	Mat tformfwd(Mat trans, Mat uv);
	Mat mergeRows(Mat A, Mat B);
	Mat mergeCols(Mat A, Mat B);
	int findNonreflectiveSimilarity(Mat uv, Mat xy, Mat &T_mat, Mat &Tinv_mat);
	Mat warp_and_crop_face(Mat src_img, Mat pts_mat);
	void align_face(Mat& img,Rect& face_rect,vector<Point2d>& face_landmark,Mat& aligned_crop);

	Mat m_ref_mat;
	Mat m_src_mat;	
	int m_resize_w;
	int m_resize_h;
	int m_landmark_num;
};

#endif

