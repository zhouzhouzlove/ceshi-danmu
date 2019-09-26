// Copyright (c) 2018, PerfXLab Technology Co., Ltd. 
// All rights reserved. 
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met: 
// 
//  * Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer. 
//  * Redistributions in binary form must reproduce the above copyright 
//    notice, this list of conditions and the following disclaimer in the 
//    documentation and/or other materials provided with the distribution. 
//  * Neither the name of PerfXLab Technology Co., Ltd. nor the names of its 
//    contributors may be used to endorse or promote products derived from 
//    this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
#ifndef INTERFACE_H
#define INTERFACE_H
#if defined INFERXLTE_FAKE_LIB
#include <stdbool.h>
struct inferx_handler;
#else
#include "inferxlite_common.h"
#endif
#ifdef __cplusplus
extern "C"
{
#endif //cpp

void inferx_elem_wise_operate(int coeffs_num, float* coeffs, char *op_mode, bool stabel_prod_grad, int bottom_num, char **bottoms_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_convolution(int input_c, int output_c, int kernel_h, int kernel_w, int str_h, int str_w, int pad_h, int pad_w, int group, int dilation, int axis, bool bias, bool force_nd_im2col, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, int activation_type, float negative_slope, struct inferx_handler *hd);
void inferx_deconvolution(int input_c, int output_c, int kernel_h, int kernel_w, int str_h, int str_w, int pad_h, int pad_w, int group, int dilation, int axis, bool bias, bool force_nd_im2col, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_global_pooling(char *pool_mode, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
//void inferx_global_pooling(enum INFERX_OPERATION op, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_inner_product(int num_input, int num_output, bool bias, bool transpose, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_relu(float slope, char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_prelu(int shared_channels, float default_slope, char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_power(float power, float scale, float shift, char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_tanh(char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_sigmoid(char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_log_softmax(char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_softmax(int axis, char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_lrn(int local_size, float alpha, float beta, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_batchnorm(float moving_average_fraction, float eps, bool use_global_stats, char *bottom_pre,char *top_pre,char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_scale(int axis,int num_axes, bool bias, char* bottom_pre, char* top_pre, char* iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_input(int *nchw, void *pdata, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_print_data(char *bottom_pre, char *data_pre, struct inferx_handler *hd);
void inferx_print_model(char *bottom_pre, char *data_pre, struct inferx_handler *hd);
void inferx_sort_data(char *bottom_pre, char *data_pre, struct inferx_handler *hd);
void inferx_save_data(char *path, char *bottom_pre, char *data_pre, struct inferx_handler *hd);
void inferx_zero_data(struct inferx_data_pipeline *p);
void inferx_concat(int num_output,int axis,int concat_dim,int bottom_num,char **bottoms_pre,char *top_pre,char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_slice(int axis, char *bottom_pre, char *top1_pre, char *top2_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_elem_wise_operate(int coeffs_num, float* coeffs,char *op_mode,bool stabel_prod_grad,int bottom_num,char **bottoms_pre,char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_crop(int axis,int offset,char* bottom_pre, char* bottom_mode_pre,char *top_pre,char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_finalize(char* modelname, struct inferx_handler *hd);
void inferx_reshape(int n, int c, int h, int w, char *bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_reorg(int stride, int n, int c, int h, int w, char *bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_resize(float resize_ratio, float out_height_scale, float out_width_scale, bool is_pyramid_test, char *resize_method, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void* inferx_get_model(char *bottom_pre, char *data_pre, struct inferx_handler *hd);
void inferx_pooling_yolo(int kernel_h, int kernel_w, int str_h, int str_w, int pad_h, int pad_w,char *pool_mode, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_pooling(int kernel_h, int kernel_w, int str_h, int str_w, int pad_h, int pad_w,char *pool_mode, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_net_preprocess_v2(char *data_c, char *model, int nchw[4], struct inferx_handler *hd);
void inferx_net_preprocess(char *data, char *model, struct inferx_handler *hd);
void* inferx_get_data(char *bottom_pre, char *data_pre, int *len, struct inferx_handler *hd);
void inferx_tiling(int tile_dim, float scale, char* bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_flatten(char* bottom_pre, char* top_pre, char* iname_pre, char* model_pre, char* data_pre, struct inferx_handler* hd);
void inferx_upsample(int stride, int forward, float scale, char *bottom_pre, char *top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
void inferx_permute(int blob_dim1,int blob_dim2,int blob_dim3,int blob_dim4,int axis_num,char *bottom_pre,char *top_pre,char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);

// void inferx_reshape(int n, int c, int h, int w, char *bottom_pre, char* top_pre, char *iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);

void inferx_yolo(int l_n, int l_classes, float *mask, int masklen, float *anchors, int anchorslen, char* bottom_pre, char* top_pre, char* iname_pre, char *model_pre, char *data_pre, struct inferx_handler *hd);
#ifdef __cplusplus
}
#endif//cpp

#endif// INTERFACE_H
