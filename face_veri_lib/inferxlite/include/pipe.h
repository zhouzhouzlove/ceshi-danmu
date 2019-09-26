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
#ifndef PIPE_H
#define PIPE_H

#include "inferxlite_common.h"
#include "hash.h"
#ifdef __cplusplus
extern "C"
{
#endif
void inferx_insert_model_func(char *name, void *func, struct inferx_handler *hd);
void inferx_pipeline_init(struct inferx_handler *hd);
void inferx_weight_bias_rand_init(struct inferx_model_pipeline* m, int nw, struct inferx_handler *hd);
void inferx_input_rand_init(struct inferx_data_pipeline* input, struct inferx_handler *hd);
struct inferx_model_pipeline* inferx_create_model_pipeline(char *name, int nw, int *nv, int *vw, struct inferx_data_arg dg, struct inferx_handler *hd);
void load_model_and_data_from_ball(char *fname, struct inferx_handler *hd);
void inferx_load_model_data(char *fname_pre, struct inferx_handler *hd);
void inferx_print_time_by_layer(struct inferx_handler *hd);
void inferx_check_size(struct inferx_model_pipeline* model, int nw, int *nv, int *vw, char *name);
void inferx_handler_init(inferx_context ctx);
void inferx_func_pointer_init(struct inferx_handler *hd);
void inferx_keep_data_shape(struct inferx_data_pipeline *input, struct inferx_data_pipeline *output);
void inferx_update_data_shape(struct inferx_data_pipeline *input, struct inferx_data_pipeline *output, int output_c, int pad_h, int pad_w, int kernel_h, int kernel_w, int str_h, int str_w, int dila_h, int dila_w, char func);
int inferx_get_data_len(inferx_data_tensor input, int *nchw, int output_c, int pad_h, int pad_w, int kernel_h, int kernel_w, int str_h, int str_w, char func);
void inferx_update_input_data(char *bottom, struct inferx_handler *hd);
void inferx_update_layer_timer(struct timeval end_layer, char *mname, struct inferx_handler *hd);
//void inferx_insert_model_func(char *name, void *func, struct inferx_handler *hd);
void inferx_str_to_int(const char *nchw_c, int *nchw_l);
void inferx_parse_str(char *data, int *nchw);
void inferx_set_init_var(bool *weight_has_load, bool *data_has_init, char *weight, char *data, struct inferx_handler *hd);
bool inferx_var_add_init(char *var, struct inferx_handler *hd);
void inferx_model_prefix_init(char **str);

void time_total(struct inferx_time_event *timevent);
void time_mark(struct inferx_time_event *timevent);
void time_continue(struct inferx_time_event *timevent);
void time_start(struct inferx_time_event *timevent);
void inferx_bench_start(struct inferx_handler *hd);
void inferx_bench_end(const char* comment,struct inferx_handler *hd);
#ifdef __cplusplus
}
#endif //cpp
#endif //PIPE_H
