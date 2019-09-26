#ifndef _Align_HEADERS
#include "align.h"
#endif

#define INPUT_WIDTH 96
#define INPUT_HEIGHT 112
#define LAND_MARK_NUM 5
#if __ANDROID__
#include <android/log.h>
#endif 

#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, "perfxlab_align", __VA_ARGS__);

Align::Align()
{
  //do nothing
}
Align::~Align()
{
  //do nothing
}
void Align::init()
{
  m_resize_w = INPUT_WIDTH;
  m_resize_h = INPUT_HEIGHT;
  m_landmark_num = LAND_MARK_NUM;
  m_src_mat= Mat::zeros(m_landmark_num, 2, CV_32F);
  m_ref_mat = (Mat_<float>(m_landmark_num, 2)<< 30.29459953, 51.69630051,65.53179932, 51.50139999,48.02519989, 71.73660278,33.54930115, 92.3655014,62.72990036, 92.20410156);
}
 Mat Align::tformfwd(Mat trans,Mat uv)
{
	Mat constant_mat_tmp = Mat::ones(1,m_landmark_num, CV_32F);
	uv = uv.t();
	uv.push_back(constant_mat_tmp);
	uv = uv.t();
	Mat mat_tmp = uv * trans;
	Mat xy = Mat::ones(m_landmark_num, 2, CV_32F);	
	xy = mat_tmp.colRange(0,2);
	return xy;
}
Mat Align::mergeRows(Mat A, Mat B)
{
	CV_Assert(A.cols == B.cols&&A.type() == B.type());
	int totalRows = A.rows + B.rows;
	Mat mergedDescriptors(totalRows, A.cols, A.type());
	Mat submat = mergedDescriptors.rowRange(0, A.rows);
	A.copyTo(submat);
	submat = mergedDescriptors.rowRange(A.rows, totalRows);
	B.copyTo(submat);

	return mergedDescriptors;
}
Mat Align::mergeCols(Mat A, Mat B)
{
	CV_Assert(A.rows == B.rows&&A.type() == B.type());
	int totalCols = A.cols + B.cols;
	Mat mergedDescriptors(totalCols, A.rows, A.type());
	Mat submat = mergedDescriptors.colRange(0, A.cols);
	A.copyTo(submat);
	submat = mergedDescriptors.rowRange(A.cols, totalCols);
	B.copyTo(submat);

	return mergedDescriptors;
}
int Align::findNonreflectiveSimilarity( Mat uv, Mat xy,Mat &T_mat,Mat &Tinv_mat)
{
	Mat m_T_mat, m_Tinv_mat;
	int K = 2;
	int M = xy.rows;  //M=5  landmark num
	vector<Point2f> xy_vec;
	xy_vec = Mat_<Point2f>(xy);
		
	vector<float>x_vec,y_vec;
	for (size_t i = 0; i < M; i++)
	{
		float x_tmp = xy_vec[i].x;
		x_vec.push_back(x_tmp);
		float y_tmp = xy_vec[i].y;
		y_vec.push_back(y_tmp);
	}
	
	Mat mat_tmp_xy_1 = Mat::ones(M, 4, CV_32F);
	Mat mat_ones = Mat::ones(M, 1, CV_32F);
	Mat mat_zeros = Mat::zeros(M, 1, CV_32F);

	Mat dsttemp = mat_tmp_xy_1.col(0);
	(Mat(x_vec)).copyTo(dsttemp);
	dsttemp = mat_tmp_xy_1.col(1);
	(Mat(y_vec)).copyTo(dsttemp);
        dsttemp = mat_tmp_xy_1.col(2);
	mat_ones.copyTo(dsttemp);
	dsttemp = mat_tmp_xy_1.col(3);
	mat_zeros.copyTo(dsttemp);


	Mat mat_tmp_xy_2 = Mat::ones(M, 4, CV_32F);
	dsttemp = mat_tmp_xy_2.col(0);
	(Mat(y_vec)).copyTo(dsttemp);
	dsttemp = mat_tmp_xy_2.col(1);
	Mat x_mat_tmp = Mat(x_vec)*(-1);
	(x_mat_tmp).copyTo(dsttemp);
	dsttemp = mat_tmp_xy_2.col(2);
	mat_zeros.copyTo(dsttemp);
	dsttemp = mat_tmp_xy_2.col(3);
	mat_ones.copyTo(dsttemp);

	Mat X_mat = Mat::ones(2*M, 4, CV_32F);
	X_mat = mergeRows(mat_tmp_xy_1, mat_tmp_xy_2);
	
	vector<Point2f> uv_vec;
	uv_vec = Mat_<Point2f>(uv);
	vector<float>u_vec, v_vec;
	for (size_t i = 0; i < M; i++)
	{
		float u_tmp = uv_vec[i].x;
		u_vec.push_back(u_tmp);
		float v_tmp = uv_vec[i].y;
		v_vec.push_back(v_tmp);
	}
	Mat U_mat = Mat::ones(2 * M, 1, CV_32F);
	U_mat = mergeRows(Mat(u_vec), Mat(v_vec));
	

	// We know that X * r = U
	Mat s_svd, u_svd, vt_svd;
	SVD::compute(X_mat, s_svd, u_svd, vt_svd);
	int rank_X = countNonZero(s_svd);

	cv::Mat_<float> r_mat;
	if (rank_X > 2 * K || rank_X == 2 * K)
	{
		solve(X_mat, U_mat, r_mat,DECOMP_NORMAL);
	}	     
	else
	{
		cout << "can not do lstsq" << endl;
		return -1;
	}

	Mat sc_mat = r_mat.row(0);
	Mat ss_mat = r_mat.row(1);
	Mat tx_mat = r_mat.row(2);
	Mat ty_mat = r_mat.row(3);

	Mat tinv_tmp1, tinv_tmp2, tinv_tmp3;

	Mat constanm_T_mat = Mat::zeros(1, 3, CV_32F);
	constanm_T_mat.col(2) = 1;

	tinv_tmp1 = mergeRows(sc_mat, ss_mat);
	tinv_tmp1 = mergeRows(tinv_tmp1, tx_mat);

	tinv_tmp2 = mergeRows(ss_mat*(-1),sc_mat);
	tinv_tmp2 = mergeRows(tinv_tmp2, ty_mat);  
	
	m_Tinv_mat = mergeRows(tinv_tmp1.t(), tinv_tmp2.t()); 
	m_Tinv_mat.push_back(constanm_T_mat);
	m_Tinv_mat = m_Tinv_mat.t(); //

	m_T_mat = m_Tinv_mat.inv();
	m_T_mat = m_T_mat.t();
	//m_T_mat.push_back(constanm_T_mat);
	m_T_mat = m_T_mat.t();
	T_mat = m_T_mat.clone();
	Tinv_mat = m_Tinv_mat.clone();
	return 0;
}
//extern perftimer timer;
Mat Align::warp_and_crop_face(Mat src_img, Mat pts_mat)
{	
	std::cout<<"warp_and_crop"<<std::endl;
	m_src_mat = pts_mat;
	Mat trans1, trans1_inv;
	findNonreflectiveSimilarity(m_src_mat, m_ref_mat, trans1, trans1_inv);

	Mat xyR_mat = m_ref_mat.clone();
	xyR_mat.col(0) = (-1)* xyR_mat.col(0);
	//std::cout<<"print1"<<std::endl;
/*
       	for(int i=0; i<m_ref_mat.rows*m_ref_mat.cols*m_ref_mat.channels();++i)
	{
		
		printf("pts_mat:[%d] %f\n",i,((float*)pts_mat.data)[i]);
		printf("m_ref_mat:[%d] %f\n",i,((float*)m_ref_mat.data)[i]);

	}
*/
	Mat trans2r, trans2r_inv;
	findNonreflectiveSimilarity(m_src_mat, xyR_mat, trans2r, trans2r_inv);
//	std::cout<<"print2"<<std::endl;
/*
	for(int i=0; i<m_ref_mat.rows*m_ref_mat.cols*m_ref_mat.channels();++i)
	{
		printf("pts_mat:[%d] %hhu\n",i,pts_mat.data[i]);
		printf("m_ref_mat:[%d] %hhu\n",i,m_ref_mat.data[i]);

	}
*/

	Mat TreflectY = (Mat_<float>(3, 3)<< -1, 0, 0, 0, 1, 0, 0, 0, 1);  
	Mat trans2 = trans2r * TreflectY;
	Mat xy1 = tformfwd(trans1, m_src_mat);
	double norm1;
	norm1=norm(xy1, m_ref_mat);

	Mat xy2 = tformfwd(trans2, m_src_mat);
	double norm2;
	norm2 = norm(xy2, m_ref_mat);
//	std::cout<<"print3"<<std::endl;
/*
       	for(int i=0; i<m_ref_mat.rows*m_ref_mat.cols*m_ref_mat.channels();++i)
	{
		printf("pts_mat:[%d] %hhu\n",i,pts_mat.data[i]);
		printf("m_ref_mat:[%d] %hhu\n",i,m_ref_mat.data[i]);

	}
*/
	Mat trans, trans_inv;
	Mat trans2_inv,tfm;
	if (norm1<norm2||norm1==norm2)
	{
		trans = trans1;
		trans_inv = trans1_inv;
		//std::cout<<"norm1: "<<norm1<<std::endl;
		//std::cout<<"norm2: "<<norm2<<std::endl;
		//std::cout<<"this is trans1"<<std::endl;
	}
	else
	{
		trans = trans2;
		trans2_inv = trans2.inv();
		trans_inv = trans2_inv;	
		//std::cout<<"norm1: "<<norm1<<std::endl;
		//std::cout<<"norm2: "<<norm2<<std::endl;
		//std::cout<<"this is trans2"<<std::endl;
	}
	tfm = trans.colRange(0, 2).t();
	Mat affine_img;
        ///std::cout<<"crop image\n"<<std::endl;;
       	/*
	for(int i=0; i<m_ref_mat.rows*m_ref_mat.cols*m_ref_mat.channels();++i)
	{
		printf("pts_mat:[%d] %hhu\n",i,pts_mat.data[i]);
		printf("m_ref_mat:[%d] %hhu\n",i,m_ref_mat.data[i]);

	}
*/
/*
       	for(int i=0; i<trans.rows*trans.cols*trans.channels();++i)
	{
		printf("trans:[%d] %hhu\n",i,trans.data[i]);

	}
       	for(int i=0; i<tfm.rows*tfm.cols*tfm.channels();++i)
	{
		printf("tfm:[%d] %hhu\n",i,tfm.data[i]);

	}
*/
        //timer.time_turn("1");
	warpAffine(src_img, affine_img, tfm, Size(m_resize_w, m_resize_h));
        //for(int i=0; i<src_img.rows*src_img.cols*src_img.channels(); ++i)
        //{
//		printf("src image:[%d] %hhu\n",i,src_img.data[i]);
        //}
//        std::cout<<"affine image\n"<<std::endl;;
/*
        for(int i=0; i<affine_img.rows*affine_img.cols*affine_img.channels(); ++i)
	{
		printf("src image:[%d] %hhu\n",i,src_img.data[i]);
		printf("affine image:[%d]: %hhd\n",i,affine_img.data[i]);
	}
*/
	//warpAffine(src_img, face_img, tfm, Size(200, 200), INTER_LINEAR + WARP_INVERSE_MAP);
	//cout << "TEST warp_and_crop_face success" << endl;
	return affine_img;
}

void Align::align_face(Mat& img,Rect& face_rect,vector<Point2d>& face_landmark,Mat& aligned_crop)
{
	if(img.empty())
	{
		#if __ANDROID__
		LOGE("Align input image is null!");
		#endif
		std::cout<<"Align input image is null"<<std::endl;
		exit(1);
	}
	float expand_scale=0.2;
	int face_w = face_rect.width;
	int face_h = face_rect.height;
	int x_offset = max(int(face_rect.x - face_rect.width*expand_scale), 0);
	int y_offset = max(int(face_rect.y - face_rect.height*expand_scale), 0);
	int expand_w = min(int(face_rect.width*(1 + 2 * expand_scale)),img.cols-face_rect.x);
	int expand_h = min(int(face_rect.height*(1 + 2 * expand_scale)),img.rows-face_rect.y);
	Rect face_rect_expand = Rect(x_offset, y_offset, expand_w,  expand_h);
	Mat face_crop = img(face_rect_expand);//
	Mat landmark_src = (Mat_<float>(LAND_MARK_NUM, 2) <<
		face_landmark[0].x - x_offset, face_landmark[0].y - y_offset,
		face_landmark[1].x - x_offset, face_landmark[1].y - y_offset,
		face_landmark[2].x - x_offset, face_landmark[2].y - y_offset,
		face_landmark[3].x - x_offset, face_landmark[3].y - y_offset,
		face_landmark[4].x - x_offset, face_landmark[4].y - y_offset);
	std::cout<<"landmark_src"<<std::endl;
	for(int i=0; i<5; ++i)
	{
		std::cout<<"x: "<<landmark_src.at<float>(i,0)<<", y: "<<landmark_src.at<float>(i,1)<<std::endl;
	}
	if(face_crop.empty())
	{
		#if __ANDROID__
		LOGE("Align face_crop is null!");
		#endif
		std::cout<<"Align face_crop is null"<<std::endl;
		exit(1);
	}
	aligned_crop = warp_and_crop_face(face_crop, landmark_src);
      return ; 
      
}
