#ifndef _landmark_HEADERS
#include "landmark.h"
#endif

#define INPUT_WIDTH 96
#define INPUT_HEIGHT 112
#define INPUT_SCALE_VALUE -127.5
#if __ANDROID__
#include <android/log.h>
#endif 
#include <sys/time.h>

#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, "perfxlab_landmark", __VA_ARGS__);
extern "C" void FaceLandmark(char* path, char* model, char* data, int *shape, int nshape, void* pdata, void** pout,int *len, struct inferx_handler *hd);
Landmark::Landmark()
{
	//do nothing
}
Landmark::~Landmark()
{
	inferx_destroy_context(m_handle_ctx);
	//  delete &m_mean_mat;
}

void Landmark::model_init(char* model_path,char* model_name,char* model_title)
{
	m_resize_w=INPUT_WIDTH;
	m_resize_h=INPUT_HEIGHT;
	//m_mean_mat=*(new Mat(m_resize_h, m_resize_w,CV_32FC3, Scalar(127.5, 127.5, 127.5)));
	m_mean_mat=Mat(m_resize_h, m_resize_w,CV_32FC3, Scalar(INPUT_SCALE_VALUE, INPUT_SCALE_VALUE, INPUT_SCALE_VALUE));
	strcpy(m_ip3_scale_name,"ip2_scale");
	strcpy(m_inferxlite_model_path,model_path);

	m_handle_ctx=inferx_create_context();
	//m_handle_ctx=inferx_get_context();
	//  int shape[4]={1,3,m_resize_h,m_resize_w};
	shape[0]=1;
	shape[1]=3;
	shape[2]=m_resize_h;
	shape[3]=m_resize_w;
//	#if __ANDROID__
		inferx_insert_model_func(model_name, (void*)FaceLandmark, &(m_handle_ctx->hd));
//	#endif
	inferx_load(m_handle_ctx,m_inferxlite_model_path,model_name,shape,4);
}	


#define  PERFCV_INTER_RESIZE_SCALE (1 << 11)
#define PERFCV_INC(x,l) ((x+1) >= (l) ? (x):((x)+1))
int Landmark::preprocess(Mat input, Mat output, int w_new, int h_new){
	int dst_cols = output.cols;
	int dst_rows = output.rows/3;
	double scale_x = (double)dst_cols / w_new;
	double scale_y = (double)dst_rows / h_new;
	double inverse_scale_x = 1./ scale_x;
	double inverse_scale_y = 1./ scale_y;
	int src_cols = w_new;
	int src_rows = h_new;
	int channel = 3;
	int k;
	float fx, fy;
	int sx, sy, dx, dy;
	int xmin, xmax;
	int dst_step, src_step;
	dst_step = dst_cols;
	src_step = input.step[0];//input.cols * channel;

	float cbuf[2];
	short _fx[dst_cols*2];
	short _fy[dst_rows*2];
	int _sx[dst_cols];
	int _sy[dst_rows];

	for(int i=0;i<dst_cols;i++){
		float tmp = (i + 0.5f) * inverse_scale_x - 0.5f;
		_sx[i] = (int)(tmp); 
		tmp -= _sx[i];
		if(_sx[i]<0){_sx[i]=0;tmp=0;}
		if(_sx[i]+1>=src_cols){_sx[i] = src_cols -1;tmp = 0;}
		_fx[2*i  ] = (int)((1.f - tmp) * PERFCV_INTER_RESIZE_SCALE);
		_fx[2*i+1] = (int)((      tmp) * PERFCV_INTER_RESIZE_SCALE);
	}
	for(int i=0;i<dst_rows;i++){
		float tmp = (i + 0.5f) * inverse_scale_y - 0.5f;
		_sy[i] = (int)(tmp); 
		tmp -= _sy[i];
		if(_sy[i]<0){_sy[i]=0;tmp=0;}
		if(_sy[i]+1>=src_rows){_sy[i] = src_rows -1;tmp = 0;}
		_fy[2*i  ] = (int)((1.f - tmp) * PERFCV_INTER_RESIZE_SCALE);
		_fy[2*i+1] = (int)((      tmp) * PERFCV_INTER_RESIZE_SCALE);
	}
	unsigned char * src_data = (uchar*)(input.data);
	float * dst_data0 = (float*)(output.data);
	float * dst_data1 = (float*)(output.data)+dst_rows*dst_cols;
	float * dst_data2 = (float*)(output.data)+2*dst_rows*dst_cols;
	uchar srctmp = 0;
	uchar src_out[3] = {0};
	int ii=0;
	for(int dy=0;dy<dst_rows;dy++){
		float fy = _fy[dy];
		int index_y = _sy[dy] * src_step; 
		int dst_y = dy * dst_step; 
		int sy = _sy[dy];
		if(sy<input.rows){
			for(int dx=0;dx<dst_cols;dx++){
				float fx = _fx[dx];
				int src_index = index_y + _sx[dx]*3;
				int dst_index = dst_y + dx*3;
				short cbuf[4];
				int dst_data_tmp_c[3] = {0};
				int dst_data_tmp_r[3] = {0};

				cbuf[0] = _fy[2*dy];
				cbuf[1] = _fx[2*dx];
				cbuf[2] = _fy[2*dy+1];
				cbuf[3] = _fx[2*dx+1];
				for(int r = 0; r < 2; r++)
				{
					int src_index_tmp = src_index + r * src_step;
					int sx = sy+r>=input.rows ? input.cols : _sx[dx];
					src_index_tmp = src_index_tmp < 0 ? src_index_tmp + src_step : src_index_tmp;

					uchar * srcdata = sx >= input.cols ? src_out : src_data + src_index_tmp;
					dst_data_tmp_c[0] += cbuf[1] * srcdata[0];
					dst_data_tmp_c[1] += cbuf[1] * srcdata[1];
					dst_data_tmp_c[2] += cbuf[1] * srcdata[2];
					srcdata = sx >= input.cols-1 ? src_out : src_data + src_index_tmp + 3;
					dst_data_tmp_c[0] += cbuf[3] * srcdata[0];
					dst_data_tmp_c[1] += cbuf[3] * srcdata[1];
					dst_data_tmp_c[2] += cbuf[3] * srcdata[2];
					dst_data_tmp_r[0] += ((dst_data_tmp_c[0] >> 4) * cbuf[2 * r]) >> 16;
					dst_data_tmp_c[0] = 0;
					dst_data_tmp_r[1] += ((dst_data_tmp_c[1] >> 4) * cbuf[2 * r]) >> 16;
					dst_data_tmp_c[1] = 0;
					dst_data_tmp_r[2] += ((dst_data_tmp_c[2] >> 4) * cbuf[2 * r]) >> 16;
					dst_data_tmp_c[2] = 0;
				}
				*dst_data0++ = (uchar)((dst_data_tmp_r[0] + 2) >> 2) - INPUT_SCALE_VALUE;
				*dst_data1++ = (uchar)((dst_data_tmp_r[1] + 2) >> 2) - INPUT_SCALE_VALUE;
				*dst_data2++ = (uchar)((dst_data_tmp_r[2] + 2) >> 2) - INPUT_SCALE_VALUE;
			}
		}
		else
		{
			for(int dx=0;dx<dst_cols;dx++){
				*dst_data0++ = (float)(0 - INPUT_SCALE_VALUE);
				*dst_data1++ = (float)(0 - INPUT_SCALE_VALUE);
				*dst_data2++ = (float)(0 - INPUT_SCALE_VALUE);
			}
		}
	}
	return 0;
}

vector<Point2f>  Landmark::detect(Mat img_src,std::vector<float> & det_rect)
{ 
#if __ANDROID__
	// __android_log_print(ANDROID_LOG_INFO,"perfxlab--","start get_feature");
	LOGE("start get_feature");
#endif
	if(img_src.empty())
	{

		std::cout<<"input image is empty!"<<std::endl;
		exit(1);
	}
	cv::Mat img;
#if 0
	make_border(img_src, img);
	if(img.empty())
	{
#if __ANDROID__
		LOGE("input img is null!");
#endif 
		std::cout<<"landmark input img is null!"<<std::endl;
		exit(1);
	}  
	cv::Rect face_rect;
	face_rect.x = std::max(0,int(det_rect[0]));
	face_rect.y = std::max(0,int(det_rect[1]));
	face_rect.width = std::min(int(det_rect[2]-det_rect[0]), img.cols-face_rect.x);
	face_rect.height = std::min(int(det_rect[3]-det_rect[1]), img.rows-face_rect.y);

	Mat src_img=img;
	float src_landmark_ratio_h=float(m_resize_h)/(face_rect.height);
	float src_landmark_ratio_w=float(m_resize_w)/(face_rect.width);
	Mat img_resize,img_float,img_float_resize;
	Mat face_crop=img(face_rect);
	resize(face_crop,face_crop,Size(m_resize_w,m_resize_h),(0,0),(0,0),INTER_LINEAR);
	int face_crop_w=face_crop.cols;
	int face_crop_h=face_crop.rows;
	face_crop.convertTo(img_float_resize,CV_32FC3);//
	subtract(img_float_resize,m_mean_mat,img_float,noArray(),CV_32FC3);
	vector<Mat> input_channels;
	struct timeval start,finish;
	float duration;
	gettimeofday(&start,NULL);
	float *data=(float*)malloc(sizeof(float)*m_resize_h*m_resize_w*3);    
	gettimeofday(&finish,NULL);
	duration = ((double)(finish.tv_sec - start.tv_sec)*1000000 +(finish.tv_usec - start.tv_usec))/1000;
	std::cout<<"malloc time: "<<duration<<std::endl;

	for(int c_idx=0;c_idx<3;c_idx++){
		Mat channel(m_resize_h,m_resize_w,CV_32FC1,data+m_resize_h*m_resize_w*c_idx);
		input_channels.push_back(channel);
	}
	split(img_float,input_channels);
	Mat resout = Mat(m_resize_h*3, m_resize_w, CV_32FC1,data);
	imwrite("resout.bmp", resout);
//	timer.time_turn("__________: landmark preprocess");
#else

	cv::Rect face_rect;
	face_rect.x = std::max(0,int(det_rect[0]));
	face_rect.y = std::max(0,int(det_rect[1]));
	face_rect.width = std::min(int(det_rect[2]-det_rect[0]), img_src.cols-face_rect.x);
	face_rect.height = std::min(int(det_rect[3]-det_rect[1]), img_src.rows-face_rect.y);

	float src_landmark_ratio_h=float(m_resize_h)/(face_rect.height);
	float src_landmark_ratio_w=float(m_resize_w)/(face_rect.width);
	Mat img_resize,img_float,img_float_resize;
	Mat face_crop=img_src(face_rect);
	//imwrite("input.bmp", face_crop);

	Mat resout = Mat(m_resize_h*3, m_resize_w, CV_32FC1);
	preprocess(face_crop, resout, face_crop.cols, face_crop.rows);
	//imwrite("resout.bmp", resout);

	float *data=(float *)resout.data; 
//	timer.time_turn("__________: make border landmark");
#endif
	int len;
	inferx_run(m_handle_ctx, data);//change
	float *result1;
	inferx_get_result(m_handle_ctx,m_ip3_scale_name,(void**)&result1,&len);

	vector<Point2f>landmark_pts;
	for(int j=0;j<10;j=j+2)
	{ 
		float x_scale=(result1[j])*m_resize_w;
		float y_scale=(result1[j+1])*m_resize_h;
		x_scale=x_scale/src_landmark_ratio_w+face_rect.x;
		y_scale=y_scale/src_landmark_ratio_h+face_rect.y;
		Point2f landmark_point(x_scale,y_scale);
		//circle(img,landmark_point,4,cv::Scalar(255,0,0));
		landmark_pts.push_back(landmark_point);
	}
	//free(data);
	return landmark_pts;  
}		
