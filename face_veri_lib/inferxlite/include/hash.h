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
#ifndef HASH_H
#define HASH_H

#ifdef __cplusplus
extern "C"
{
#endif //cpp

#include "inferxlite_common.h"

struct inferx_data_pipeline* inferx_data_map(const char *p, struct inferx_handler *hd);
struct inferx_data_pipeline* inferx_data_map_init(const char *p, int *nchw, struct inferx_handler *hd);
struct inferx_model_pipeline* inferx_weight_bias_map(const char *p, struct inferx_handler *hd);
struct inferx_model_pipeline* inferx_weight_bias_map_init(const char *p, int nw, int *nv, int *vw, struct inferx_data_arg dg, struct inferx_handler *hd);
uint64_t inferx_hash_str(const char* p, int n);
int inferx_hash_long_insert(const uint64_t n_hash, struct inferx_handler *hd);
int inferx_hash_long_get(const uint64_t n_hash, struct inferx_handler *hd);
void inferx_print_elem_value(int index, struct inferx_handler *hd);
void inferx_dltensor_type_init(inferx_data_tensor *tsr, int code, int bits, int lanes);
void inferx_hash_clear_part(struct inferx_handler *hd);

int inferx_existdata_map(const char *p, struct inferx_handler *hd);
#ifdef __cplusplus
}
#endif
#endif
