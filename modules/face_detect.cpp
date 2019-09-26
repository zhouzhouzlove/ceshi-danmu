#ifndef _FACEDETECT_HEADERS
#include "face_detect.h"
#include <iostream>
#include<fstream> 
#endif
#if __ANDROID__
#include<android/log.h>
#endif
#include "utils.hpp"
using namespace std;
extern "C" void FaceNetNew(char* path, char* model, char* data, int *shape, int nshape, void* pdata, void** pout,int *len, struct inferx_handler *hd);
#define INPUT_WIDTH 320
#define INPUT_HEIGHT 224
#define INPUT_MEAN_VALUE 128.0
#define INPUT_SCALE_VALUE 1.0
#if __ANDROID__
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, "perfxlab_detect", __VA_ARGS__);
#endif 
FaceDetect::FaceDetect()
{
	//do nothing 
}
FaceDetect::~FaceDetect()
{
	inferx_destroy_context(m_handle_ctx);
	//  delete &m_mean_mat;
	//  delete &m_scale_mat;
}
void FaceDetect::model_init(char* model_path,char* model_name,char* model_title)
{ 
	m_expand_scale=0.0;
	m_force_gray=true;
	m_resize_w=INPUT_WIDTH;
	m_resize_h=INPUT_HEIGHT;
	//m_resize_w=160;
	//m_resize_h=128;
	int mat_type=m_force_gray?CV_32FC1:CV_32FC3;
	//m_mean_mat=*(new Mat(m_resize_h,m_resize_w,mat_type,Scalar::all(128.0)));
	//m_scale_mat=*(new Mat(m_resize_h,m_resize_w,mat_type,Scalar::all(1.0)));
	m_mean_mat=Mat(m_resize_h,m_resize_w,mat_type,Scalar::all(INPUT_MEAN_VALUE));
	m_scale_mat=Mat(m_resize_h,m_resize_w,mat_type,Scalar::all(INPUT_SCALE_VALUE));
	strcpy(m_pixel_blob_name,"pixel-conv");
	strcpy(m_bb_blob_name,"bb-output");
	strcpy(m_inferxlite_model_path,model_path);
	printf("m_inferxlite_model_path: %s\n",m_inferxlite_model_path);

	//__android_log_print(ANDROID_LOG_INFO,"perfxlab--","create context");
	m_handle_ctx=inferx_create_context();
	//__android_log_print(ANDROID_LOG_INFO,"perfxlab--","create context");
	//m_handle_ctx=inferx_get_context();
	//int shape[4]={1,1,m_resize_h,m_resize_w};
	shape[0]=1;
	shape[1]=1;
	shape[2]=m_resize_h;
	shape[3]=m_resize_w;

	// __android_log_print(ANDROID_LOG_INFO,"perfxlab--","load model");
//	#if __ANDROID__
        void (*p)(char*, char*, char*, int*, int, void* , void** ,int *, struct inferx_handler *);
	p=FaceNetNew;
	inferx_insert_model_func(model_name, (void*)p, &(m_handle_ctx->hd));
//	#endif

	inferx_load(m_handle_ctx,m_inferxlite_model_path,model_name,shape,4);
	//__android_log_print(ANDROID_LOG_INFO,"perfxlab--","load model");
}	

#define  PERFCV_INTER_RESIZE_SCALE (1 << 11)
#define PERFCV_INC(x,l) ((x+1) >= (l) ? (x):((x)+1))
int FaceDetect::preprocess(Mat input, Mat output, int w_new, int h_new){
	double scale_x = (double)output.cols / w_new;
	double scale_y = (double)output.rows / h_new;
	double inverse_scale_x = 1./ scale_x;
	double inverse_scale_y = 1./ scale_y;
	int dst_cols = output.cols;
	int dst_rows = output.rows;
	int src_cols = w_new;
	int src_rows = h_new;
	int channel = 3;
	int k;
	float fx, fy;
	int sx, sy, dx, dy;
	int xmin, xmax;
	int dst_step, src_step;
	dst_step = dst_cols * channel;
	src_step = input.cols * channel;

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
//	timer.time_turn("**********: init");
	unsigned char * src_data = input.data;
	float * dst_data = (float*)(output.data);//unsigned char * dst_data = output.data;
	uchar srctmp = 0;
	uchar src_out[3] = {0};
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

			cbuf[0] = _fy[2*dy];//(short)((1.f - fy) * PERFCV_INTER_RESIZE_SCALE);
			cbuf[1] = _fx[2*dx];//(short)((1.f - fx) * PERFCV_INTER_RESIZE_SCALE);
			cbuf[2] = _fy[2*dy+1];//(short)(fy * PERFCV_INTER_RESIZE_SCALE);
			cbuf[3] = _fx[2*dx+1];//(short)(fx * PERFCV_INTER_RESIZE_SCALE);
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
			unsigned char tmp[3] = {0};
			tmp[0]     = (uchar)((dst_data_tmp_r[0] + 2) >> 2);
			tmp[1]     = (uchar)((dst_data_tmp_r[1] + 2) >> 2);
			tmp[2]     = (uchar)((dst_data_tmp_r[2] + 2) >> 2);
			int gray = (1868*tmp[0]+9617*tmp[1]+4899*tmp[2]+8192) >> 14;
			*dst_data++ = (float)(gray - INPUT_MEAN_VALUE) * INPUT_SCALE_VALUE;
		}
	    }
	    else
	    {
		for(int dx=0;dx<dst_cols;dx++)
			*dst_data++ = (float)(0 - INPUT_MEAN_VALUE) * INPUT_SCALE_VALUE;
	    }
	}
	return 0;
}

void resize_rect(Rect rect_crop, int rows, int cols){
	rect_crop.x = rect_crop.x < 0? 0 : rect_crop.x; 
	rect_crop.y = rect_crop.y < 0? 0 : rect_crop.y; 
	rect_crop.width = rect_crop.width < 1? 1 : rect_crop.width; 
	rect_crop.height = rect_crop.height < 1? 1 : rect_crop.height; 
	rect_crop.x = rect_crop.x >= cols-1? cols-2 : rect_crop.x; 
	rect_crop.y = rect_crop.y >= rows-1? rows-2 : rect_crop.y; 
	rect_crop.width = rect_crop.x+rect_crop.width >= cols? cols - 1 - rect_crop.x: rect_crop.width; 
	rect_crop.height = rect_crop.y+rect_crop.height >= rows? rows - 1 - rect_crop.y: rect_crop.height; 
}

vector<vector<float> >FaceDetect::detect(Mat &img)
{
	if(img.empty())
	{
#if __ANDROID__
		LOGE("detect input img is null!");
#endif
		std::cout<<"detect input img is null!"<<std::endl;
		exit(1);
	}
#if 0
	cv::Mat border_img;
	make_border(img,border_img);
	//cv::Mat det_img;
	//border_img.copyTo(det_img);
	if(border_img.empty())
	{
#if __ANDROID__
		LOGE("detect input det_img is null!");
#endif
		std::cout<<"detect input det_img is null!"<<std::endl;
		exit(1);
	}
	Mat m_img_src = border_img;
	Mat m_img_resize,m_img_float;
	resize(m_img_src,m_img_resize,Size(m_resize_w,m_resize_h),(0,0),(0,0),INTER_LINEAR);
	cvtColor(m_img_resize,m_img_resize,COLOR_BGR2GRAY);// 0.114*B+0.587*G+0.299*R
	m_img_resize.convertTo(m_img_float ,CV_32FC1);// 123 -> 123.0, no 1.0/255.0 
	subtract(m_img_float ,m_mean_mat,m_img_float);// -= INPUT_MEAN_VALUE
	m_img_float =m_img_float.mul(m_scale_mat); // *= INPUT_SCALE_VALUE
	float *data=(float *)m_img_float.data; 
	timer.time_turn("__________: make border + resize + cvtColor + convertTo +subtract + mul");
#else
	float w_in = img.cols;
	float h_in = img.rows;
	float src_ratio = w_in/h_in;
	int w_new = w_in;
	int h_new = h_in;
	float width_height_ratio = (float)(m_resize_w) / m_resize_h;
	//printf("ratio=%f\n", width_height_ratio);
	if((src_ratio-width_height_ratio)>MAKE_BORDER_CRITERION)
	{
		//std::cout<<"the height is short, make border for height"<<std::endl;
		h_new += (int)(w_in/width_height_ratio-h_in);
	}
	if((width_height_ratio-src_ratio)>MAKE_BORDER_CRITERION)
	{
		//std::cout<<"the width is short, make border for width"<<std::endl;
		w_new += (int)(h_in*width_height_ratio-w_in);
	}

	Mat resout = Mat(m_resize_h, m_resize_w, CV_32FC1);
	preprocess(img, resout, w_new, h_new);
	imwrite("prpr.bmp", resout);
	float *data=(float *)resout.data; 
//	timer.time_turn("__________: make border 2");
#endif
	int len; 
	inferx_run(m_handle_ctx,data);
	float *pixel_p;
	float *bb_p;
	inferx_get_result(m_handle_ctx,m_pixel_blob_name,(void**)&pixel_p,&len);
	inferx_get_result(m_handle_ctx,m_bb_blob_name,(void**)&bb_p,&len);
//	timer.time_turn("__________: inferx run & get_res");

	////////////
	int feat_rows=(m_resize_h/32)*8;int feat_cols=(m_resize_w/32)*8;
	vector<vector<float> >res= GetRois(pixel_p,bb_p,feat_rows,feat_cols); 

	float src_detect_ratio_h=float(h_new)/m_resize_h;
	float src_detect_ratio_w=float(w_new)/m_resize_w;

	vector<vector<float> >res_final;
//	timer.time_turn("__________: after process need to be done");
	for(int i=0;i<res.size();i++)
	{ 
		//draw roi 
		vector<float>& face_box=res[i];
		//Rect roi;
		//roi.x=int(res[i][0]);roi.y=int(res[i][1]);roi.width=int(res[i][2]-res[i][0]);roi.height=int(res[i][3]-res[i][1]);
		//  rectangle(m_img_resize,roi,Scalar(255,0,0),2,1);

		//face crop 
		Rect rect_crop;//
		rect_crop.x=res[i][0]*src_detect_ratio_w;
		rect_crop.y=res[i][1]*src_detect_ratio_h;
		rect_crop.width=(res[i][2]-res[i][0])*src_detect_ratio_w;
		rect_crop.height= (res[i][3]-res[i][1])*src_detect_ratio_h;
		resize_rect(rect_crop, img.rows, img.cols);
		//cv::Mat face_crop = img(cv::Rect(rect_crop.x,rect_crop.y,rect_crop.width,rect_crop.height)); 
		//imwrite("face_crop.jpg",face_crop);
		vector<float> res_tmp;
		res_tmp.push_back(rect_crop.x);
		res_tmp.push_back(rect_crop.y);
		res_tmp.push_back(rect_crop.width+rect_crop.x); 
		res_tmp.push_back(rect_crop.height+rect_crop.y); 
		res_final.push_back(res_tmp);
	}
//	timer.time_turn("__________: after process");
	return res_final;
}		

void FaceDetect::Tiling(int tile_dim_, int height, int width, const float* input, int c_out, vector<vector<float> >& output,float scale) {
	int output_width_ = width * tile_dim_;
	int output_height_ = height * tile_dim_;
	int count_per_output_map_ = output_width_ * output_height_;
	int count_per_input_map_ = height * width;
	for (int c = 0; c < c_out; ++c) {
		for (int iy = 0, oy = 0; iy < height; ++iy, oy += tile_dim_) {
			for (int ix = 0, ox = 0; ix < width; ++ix, ox += tile_dim_) {
				int input_channel_offset = 0;
				for (int ty = 0; ty < tile_dim_; ++ty) {
					for (int tx = 0; tx < tile_dim_; ++tx, input_channel_offset += count_per_input_map_) {
						output[c][(oy + ty) * output_width_ + ox + tx] =
							input[input_channel_offset + iy * width + ix] * scale;
					}
				}
			}
		}
		input += count_per_output_map_;
		//      output += count_per_output_map_;
	}
}
std::vector<vector<float> > FaceDetect::NMS(size_t count, const vector<vector<float> >& box, float nms) {
	vector<pair<int, float> > order(count);
	for (int i = 0; i < count; ++i) {
		order[i].first = i;
		order[i].second = box[i][4];
	}
	std::sort(order.begin(), order.end(),
			[](const std::pair<int, float>& ls, const std::pair<int, float>& rs)
			{return ls.second > rs.second;});
	vector<int> keep;
	vector<bool> exist_box(count, true);
	for (int _i = 0; _i < count; ++_i) {
		int i = order[_i].first;
		float x1, y1, x2, y2, w, h, iarea, jarea, inter, ovr;
		if (!exist_box[i]) continue;
		keep.push_back(i);
		for (int _j = _i+1; _j < count; ++_j) {
			int j = order[_j].first;
			if (!exist_box[j]) continue;
			x1 = max(box[i][0], box[j][0]);
			y1 = max(box[i][1], box[j][1]);
			x2 = min(box[i][2], box[j][2]);
			y2 = min(box[i][3], box[j][3]);
			w = max(float(0.0), x2 - x1 + 1);
			h = max(float(0.0), y2 - y1 + 1);
			iarea = (box[i][2] - box[i][0] +1) * (box[i][3] - box[i][1] + 1);
			jarea = (box[j][2] - box[j][0] +1) * (box[j][3] - box[j][1] + 1);
			inter = w * h;
			ovr = inter / (iarea + jarea - inter);
			if (ovr >= nms) exist_box[j] = false;
		}
	}
	vector<vector<float> > result;
	result.reserve(keep.size());
	for(int i = 0; i < keep.size(); ++i) {
		result.push_back(box[keep[i]]); 
	}
	return result;
}


void FaceDetect::softmax(vector<vector<float> >& input, int size, vector<vector<float> >& output) {
	for(int n = 0; n < size; ++n) {
		float sum = 0.0f;
		for(int i = 0; i < 2; ++i) {
			output[i][n] = exp(input[i][n]);
			sum += output[i][n];
		}
		for(int i = 0; i < 2; ++i) {
			output[i][n] /= sum;
		}
	} 
}

//TODO : need to merge to inferxlite
vector<vector<float> > FaceDetect::GetRois(float* pixel_p,float* bb_p,int feat_rows,int feat_cols){
	int tile_dim=8;
	const float* pixel_conv_data=pixel_p; 
	int pixel_out_num=2;
	vector<vector<float> > tiled_pixel_out(pixel_out_num);
	for(size_t i = 0; i<pixel_out_num; ++i)
		tiled_pixel_out[i].resize(feat_rows*feat_cols);
	Tiling(tile_dim, feat_rows/8, feat_cols/8, pixel_conv_data, pixel_out_num, tiled_pixel_out);
	const float* bb_data=bb_p;
	int bb_out_num=4;
	vector<vector<float> >tiled_bb_out(bb_out_num);
	for(size_t i=0;i<bb_out_num;i++)
		tiled_bb_out[i].resize(feat_rows*feat_cols);

	Tiling(tile_dim,feat_rows/8,feat_cols/8,bb_data,bb_out_num,tiled_bb_out);
	vector<vector<float> > pred(pixel_out_num);
	for(size_t i = 0; i <pixel_out_num; ++i)
		pred[i].resize(2*feat_rows*feat_cols);
	softmax(tiled_pixel_out, feat_rows*feat_cols, pred);
	vector<vector<float> > boxes;
	int gx = feat_rows;
	int gy = feat_cols;
	for(int i = 0; i < gx; ++i) {
		for(int j = 0; j < gy; ++j) {
			vector<float> box;
			box.reserve(5);
			int os = i*gy +j;
			if(pred[1][os] > 0.8) {
				box.push_back(tiled_bb_out[0][os] + j*4);
				box.push_back(tiled_bb_out[1][os] + i*4);
				box.push_back(tiled_bb_out[2][os] + j*4);
				box.push_back(tiled_bb_out[3][os] + i*4);
				box.push_back(pred[1][os]);
				boxes.push_back(box);
			}
		}
	}
	vector<vector<float> > res = NMS(boxes.size(), boxes, 0.3);
	return res;
} 

