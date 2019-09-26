/*
	This file is generated by net_compiler.py.
	The use of included functions list as follows:

	Input:
	inferx_input(int *shape, void *pdata, char *top,char *name,char *model_pre,char *data_pre,struct inferx_handler *hd)
	

	Convolution:
	inferx_convolution(int num_input,int num_output,int kernel_h,int kernel_w,int stride_h,int stride_w,int pad_h,int pad_w,int group,int dilation,int axis,bool bias_term,bool force_nd_im2col,char *bottom,char *top, char *name,char *model_pre, char *data_pre, int activation_type, float negative_slope, struct inferx_handler *hd)
	

	ReLU:
	inferx_relu(float slope, char *bottom,char *top,char *name, char *model_pre, char *data_pre, struct inferx_handler *hd)

	LRN:
	inferx_lrn(int local_size,float alpha,float beta,char *bottom,char *top,char *name, char *model_pre, char *data_pre, struct inferx_handler *hd)

	Pooling:
	inferx_pooling(int kernel_h,int kernel_w,int stride_h,int stride_w,int pad_h,int pad_w,char *pool_mode,char *bottom,char *top,char *name,char *model_pre, char *data_pre, struct inferx_handler *hd)
	

	InnerProduct:
	inferx_inner_product(int num_input,int num_output,bool bias_term,bool transpose,char *bottom,char *top,char *name,char *model_pre, char *data_pre, struct inferx_handler *hd)

	Dropout:
	inferx_dropout(float drop_ration, char *bottom, char *top, char *name, char *model_pre, char *data_pre, struct inferx_handler *hd)

	Sigmoid:
	inferx_sigmoid(char *bottom, char *top, char *name, char *model_pre, char *data_pre, struct inferx_handler *hd)
*/


#include "interface.h"
#include "quantization.h"
#include <stdbool.h>

struct inferx_handler;
void AlexNet(char* path, char* model, char* data, int *shape, int nshape, void* pdata, void** pout,int *len, struct inferx_handler *hd)
{
	inferx_net_preprocess(data,model,hd);

	inferx_input(shape,pdata,"data_data","data",model,data,hd);
	inferx_convolution(3,96,11,11,4,4,0,0,1,1,1,true,false,"data_data","conv1_data","conv1",model,data, 0, 0, hd);
	inferx_relu(0.0,"conv1_data","conv1_data","relu1",model,data,hd);
	inferx_lrn(5,0.0001,0.75,"conv1_data","norm1_data","norm1",model,data,hd);
	inferx_pooling(3,3,2,2,0,0,"MAX","norm1_data","pool1_data","pool1",model,data,hd);
	
	inferx_convolution(96,256,5,5,1,1,2,2,2,1,1,true,false,"pool1_data","conv2_data","conv2",model,data, 0, 0, hd);
	inferx_relu(0.0,"conv2_data","conv2_data","relu2",model,data,hd);
	inferx_lrn(5,0.0001,0.75,"conv2_data","norm2_data","norm2",model,data,hd);
	inferx_pooling(3,3,2,2,0,0,"MAX","norm2_data","pool2_data","pool2",model,data,hd);
	
	inferx_convolution(256,384,3,3,1,1,1,1,1,1,1,true,false,"pool2_data","conv3_data","conv3",model,data, 0, 0, hd);
	inferx_relu(0.0,"conv3_data","conv3_data","relu3",model,data,hd);
	inferx_convolution(384,384,3,3,1,1,1,1,2,1,1,true,false,"conv3_data","conv4_data","conv4",model,data, 0, 0, hd);
	inferx_relu(0.0,"conv4_data","conv4_data","relu4",model,data,hd);
	inferx_convolution(384,256,3,3,1,1,1,1,2,1,1,true,false,"conv4_data","conv5_data","conv5",model,data, 0, 0, hd);
	inferx_relu(0.0,"conv5_data","conv5_data","relu5",model,data,hd);
	inferx_pooling(3,3,2,2,0,0,"MAX","conv5_data","pool5_data","pool5",model,data,hd);
	
	inferx_inner_product(256,4096,true,false,"pool5_data","fc6_data","fc6",model,data,hd);
	inferx_relu(0.0,"fc6_data","fc6_data","relu6",model,data,hd);
	inferx_inner_product(4096,4096,true,false,"fc6_data","fc7_data","fc7",model,data,hd);
	inferx_relu(0.0,"fc7_data","fc7_data","relu7",model,data,hd);
	inferx_inner_product(4096,1,true,false,"fc7_data","fc8_data","fc8",model,data,hd);
	inferx_sigmoid("fc8_data","prob_data","prob",model,data,hd);

	// only one output feature map 
	//inferx_sort_data("prob_data",data);
	inferx_print_data("fc8_data",data,hd);
	//if(pout)
	  //*pout = inferx_get_data("prob_data",data,len,hd);
	inferx_finalize("AlexNet",hd);

	return;
}
