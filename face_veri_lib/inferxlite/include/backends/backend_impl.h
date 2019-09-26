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
#ifndef BACKEND_IMPL_H_
#define BACKEND_IMPL_H_
#include "inferxlite_common.h"
#ifdef __cplusplus
extern "C"
{
#endif //cpp

typedef int size_int;
typedef unsigned char uchar;
static inline float safe_log(float input);

#ifdef _MSC_VER
    int gettimeofday(struct timeval *tp, void *tzp);
#endif

void prelu_forward_impl(float *input, float *output, int nchw, int hw, int channels, float *slope, const int div_factor);
void unpack_no_dilation_nchw_impl(struct inferx_conv_arg * conv_arg_pt, const float* input, float* unpack);
void unpack_dilation_nchw_impl(struct inferx_conv_arg * conv_arg_pt,const float* input, float* unpack);
void unpack_nhwc_impl(struct inferx_conv_arg * cov_arg_pt, const float* input, float* unpack);
void pack_nchw_impl(struct inferx_conv_arg * conv_arg_pt, const float* unpack, float* input);
void pack_nhwc_impl(struct inferx_conv_arg * conv_arg_pt, const float* unpack, float* input);
void gemm(const float* A, const float* B, float* C, bool transa, bool transb, float alpha, float beta, int m, int n, int k);
void gemv(const float* A, const float *x, float *y, bool transa,const float alpha, const float beta,const int m, const int n);
void quick_sort_descend(float* arr, int start, int end);
void convolution_forward_naive_no_pad_nchw_impl(struct inferx_conv_arg * conv_arg_pt, float *input, float *filter, float *output);
void convolution_forward_naive_pad_nchw_impl(struct inferx_conv_arg * conv_arg_pt, float *input, float *filter, float *output);
void convolution_forward_naive_pad_nhwc_impl(struct inferx_conv_arg * conv_arg_pt,float *input, float *filter, float *output);
void convolution_forward_perfdnn_nchw_impl(struct inferx_conv_arg * conv_arg_pt,const inferx_data_tensor input_t, inferx_data_tensor *filter_t1, inferx_data_tensor filter_t2, inferx_data_tensor output_t, bool *is_bias, int activation_type,float negative_scope);
void convolution_forward_perfdnn_ocl_nchw_impl(struct inferx_conv_arg *conv_arg_pt, const inferx_data_tensor input_t, inferx_data_tensor *filter_t1, inferx_data_tensor filter_t2, inferx_data_tensor output_t, bool *is_with_bias, int is_activation_merged, float activation_negative_slope, struct inferx_handler *hd);
void convolution_forward_gemm_nchw_impl(struct inferx_conv_arg * conv_arg_pt, float *input, float *filter, float *output);
void convolution_backward_gemm_nchw_impl(struct inferx_conv_arg * conv_arg_pt,float *input, float *filter, float *output);
void convolution_backward_naive_no_pad_nchw_impl(struct inferx_conv_arg * conv_arg_pt,float *Input, float *Filter, float *Output);
void convolution_backward_naive_pad_nchw_impl(struct inferx_conv_arg * conv_arg_pt,float *input, float *filter, float *output);
void max_pooling_forward_nchw_impl(struct inferx_conv_arg * conv_arg_pt,float *input, float *output);
void max_pooling_forward_nchw_yolo_impl(struct inferx_conv_arg * conv_arg_pt,float *input, float *output);
void max_pooling_forward_nhwc_impl(struct inferx_conv_arg * conv_arg_pt,float *input, float *output);
void ave_pooling_forward_nchw_impl(struct inferx_conv_arg * conv_arg_pt,float *input, float *output);
void ave_pooling_forward_nhwc_impl(struct inferx_conv_arg * conv_arg_pt,float *input, float *output);
void max_pooling_forward_nc3x3s2_impl(struct inferx_conv_arg * conv_arg_pt,float *input, float *output);
void relu_forward_impl(float *input, float *output, int nchw, float slope);
void sigmoid_forward_impl(float *input, float *output, int nchw);
void softmax_forward_impl(float *input, float *output,int softmax_axis, int * shape);
void log_softmax_forward_impl(float *input, float *output, int n, int chw);
void bias_forward_impl(float *input, float *bias, float *output, int k, int pq);
void batch_norm_forward_impl(float *input, float *output, float scale_factor, float *bn_scale1, float *bn_scale2, float eps, int n, int c, int h, int w);
void scale_forward_impl(float *input, float *output, float *gama, float *beta, int n, int c, int h, int w, bool bias);
void lrn_across_forward_impl(float *input, float *output, int local_size, float alpha, float beta, int channels, int height, int width);
void lrn_within_forward_impl(float *input, float *output, int local_size, float alpha, float beta, int channels, int height, int width);
void elem_wise_operate_impl(int num_input, float **input, float *output, int len, enum INFERX_OPERATION op);
void crop_forward_impl(float *input, float *output,  int axis, int n, int c, int h, int w, int on, int oc, int oh, int ow, int offset_n, int offset_c, int offset_h, int offset_w);
void tensor_concat_impl(int axis, int num_input, float **input, float *output, int *n, int *c, int *h, int *w, int no, int co, int ho, int wo);
void power_forward_impl(float *input, float *output, int nchw, float power, float scale, float shift);
void tanh_forward_impl(float *input, float *output, int nchw);
void reorg_forward_impl(float *x, int n, int c, int h, int w, int stride, int forward, float *out);
void resize_nearest_forward_impl(float *input, float *output, int *output_shape, float resize_ratio, int out_height_scale, int out_width_scale);
void resize_bilinear_forward_impl(float *input, float *output, int *output_shape, float resize_ratio, int out_height_scale, int out_width_scale);
void tiling_forward_impl(int tile_dim, float scale, float *input, int *input_nchw, float *output);
void upsample_forward_impl(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);
#ifdef __cplusplus
}
#endif //cpp
#endif //BACKEND_IMPL_H_
