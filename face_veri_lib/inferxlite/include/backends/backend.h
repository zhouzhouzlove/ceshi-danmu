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
#ifndef BACKEND_H
#define BACKEND_H

#include "inferxlite_common.h"

#ifdef __cplusplus
extern "C"
{
#endif //cpp

void prelu_forward_func(const inferx_data_tensor input_t, inferx_data_tensor output_t, int nchw, int hw, int channels, inferx_data_tensor slope_t, const int div_factor, struct inferx_handler *hd);
void convolution_backward_perfdnn_ocl_nchw_impl(struct inferx_conv_arg *conv_arg_pt, const inferx_data_tensor output_t, inferx_data_tensor *filter, inferx_data_tensor input_t, struct inferx_handler *hd);
void convolution_forward_func(struct inferx_conv_arg *conv_arg_pt, const inferx_data_tensor input_t, inferx_data_tensor *filter_t1, const inferx_data_tensor filter_t2, inferx_data_tensor output_t, bool *bias, int activation_type, float negative_slope, struct inferx_handler *hd);
void convolution_backward_func(struct inferx_conv_arg *conv_arg_pt, const inferx_data_tensor output_t, inferx_data_tensor *filter_t, inferx_data_tensor input_t, struct inferx_handler *hd);
void pooling_forward_func(struct inferx_conv_arg *conv_arg_pt, const inferx_data_tensor input_t, inferx_data_tensor output_t, enum INFERX_OPERATION pool, struct inferx_handler *hd);
void pooling_forward_func_yolo(struct inferx_conv_arg *conv_arg_pt, const inferx_data_tensor input_t, inferx_data_tensor output_t, enum INFERX_OPERATION pool, struct inferx_handler *hd);
void lrn_forward_func(const inferx_data_tensor input_t, inferx_data_tensor output_t, enum INFERX_LRN_WAY lrn, int local_size, float alpha, float beta, int channels, int height, int width, struct inferx_handler *hd);
void relu_forward_func(const inferx_data_tensor input_t, inferx_data_tensor output_t, int nchw, float slope, struct inferx_handler *hd);
void power_forward_func(const inferx_data_tensor input_t, inferx_data_tensor output_t, int nchw, float power, float scale, float shift, struct inferx_handler *hd);
void tanh_forward_func(const inferx_data_tensor input_t, inferx_data_tensor output_t, int nchw, struct inferx_handler *hd);
void sigmoid_forward_func(const inferx_data_tensor input_t, inferx_data_tensor output_t, int nchw, struct inferx_handler *hd);
void softmax_forward_func(const inferx_data_tensor input_t, inferx_data_tensor *output_t,int softmax_axis, struct inferx_handler *hd);
void log_softmax_forward_func(const inferx_data_tensor input_t, inferx_data_tensor output_t, int n, int chw, struct inferx_handler *hd);
void bias_forward_func(const inferx_data_tensor input_t, const inferx_data_tensor bias_t, inferx_data_tensor output_t, int n, int k, int output_hw, struct inferx_handler *hd);
void batch_norm_forward_func(const inferx_data_tensor input_t, inferx_data_tensor output_t, inferx_data_tensor scale_factor_t, const inferx_data_tensor bn_scale1_t, const inferx_data_tensor bn_scale2_t, float eps, int n, int c, int h, int w, struct inferx_handler *hd);
void scale_forward_func(const inferx_data_tensor input_t, inferx_data_tensor output_t, const inferx_data_tensor gama_t, const inferx_data_tensor beta_t, int size_n, int size_c, int size_h, int size_w, bool is_bias, struct inferx_handler *hd);
void matrix_multiply_func(const inferx_data_tensor left_t, const inferx_data_tensor right_t, inferx_data_tensor output_t, bool transa, bool transb, float alpha, float beta, int m_left, int n_left, int m_right, int n_right, struct inferx_handler *hd);
/*
extern float evaluate_classify_forward_func(const inferx_data_tensor output_t, const inferx_data_tensor target_t, int n, int len);
extern float evaluate_regress_forward_func(const inferx_data_tensor output, const inferx_data_tensor target, int n, int len);
extern float cross_entropy_binary_forward_func(const inferx_data_tensor input, const inferx_data_tensor target, int len);
*/
void elem_wise_operate_func(int num_input, inferx_data_tensor* input_t, inferx_data_tensor output_t, int len, enum INFERX_OPERATION op, struct inferx_handler *hd);
void crop_forward_func(const inferx_data_tensor input_t, inferx_data_tensor output_t,  int axis, int input_n, int input_c, int input_h, int input_w, int output_n, int output_c, int output_h, int output_w, int offset_n, int offset_c, int offset_h, int offset_w, struct inferx_handler *hd);
void tensor_concat_func(int axis, int num_input, inferx_data_tensor* input_t, inferx_data_tensor output_t, struct inferx_handler *hd);
void reorg_forward_func(inferx_data_tensor input_t, inferx_data_tensor output_t, int n, int c, int h, int w, int stride, int forward, struct inferx_handler *hd);
void resize_forward_func(inferx_data_tensor input_t, inferx_data_tensor output_t, int *output_shape, bool is_pyramid_test, enum INFERX_RESIZE_METHOD resize_method, float resize_ratio, float out_height_scale, float out_width_scale, struct inferx_handler *hd);
void tensor_slice_func(int axis, inferx_data_tensor input, inferx_data_tensor output1, inferx_data_tensor output2, struct inferx_handler *hd);
void tiling_forward_func(int tile_dim, float scale, inferx_data_tensor input_t, inferx_data_tensor output_t, struct inferx_handler *hd);
void upsample_forward_func(int stride, int forward, float scale, inferx_data_tensor input_t, inferx_data_tensor output_t, struct inferx_handler *hd);

void yolo_forward_func(const inferx_data_tensor input_t, inferx_data_tensor *output_t, float *mask, int masklen, float *anchors, int anchorslen, int l_n, int l_classes, struct inferx_handler *hd);
#ifdef __cplusplus
}
#endif //cpp

#endif //BACKEND_H
