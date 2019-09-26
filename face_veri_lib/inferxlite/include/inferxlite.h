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
#pragma once

#include "inferxlite_common.h"

#ifdef __cplusplus
extern "C"
{
#endif //cpp

struct inferx_context_t;
typedef struct inferx_context_t *inferx_context;

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

/**
 *@brief  Create the context
 *@detial  Create the inferx_context_t. The framework uses inferx_context_t structure to manage DNN model and feature maps.
 *@return inferx_context a pointer of structure inferx_context_t
*/
inferx_context inferx_create_context();

/**
 *@brief  Initialize the model
 *@detail  Initialize model struct.
 *@param[in]  ctx               The context to manage this model.
 *@param[in]  path              model structure path.
 *@param[in]  model              model name.
*/
void inferx_init(inferx_context ctx, char * path, char * model);

/** @brief  This function load weights from model file.
 *  @detail  Load the weight data and allocate memory for feature map
 *  @param[in]  ctx		The context to manage this model.
 *  @param[in]  model_path      The path of model file.
 *  @param[in]  model	        The model's name.
 *  @param[in]  shape           Input size.
 *                              shape[n][c][h][w]
 *                              Defalut  n = 1 and only support n = 1
 *  @param[in]  nshape          The number of input dimensions(the len of shape).
 *                              Defalut nshape = 4
*/
void inferx_load(inferx_context ctx, char *model_path, char *model, int *shape, int nshape);
void inferx_load_init(inferx_context ctx, char *model_path, char *model, int *shape, int nshape);


/** @brief  Run the model
 *  @detail  Input the data and run the model. You can get the result by pout parameter.
 *  @param[in]  ctx		The context to manage this model.
 *  @param[in]  p_data		Input data pointer
*/
//      get result use inferx_get_result funtion
void inferx_run(inferx_context ctx, void *p_data);
void inferx_run_gpu(inferx_context ctx, void *p);


/** @brief  Run the afterprocess of the model
 *  @detail  Just for test now.
 *  @param[in]  ctx		The context to manage this model.
*/
void inferx_afterprocess(inferx_context ctx);
detection* inferx_yolov3afterprocess(inferx_context ctx, int w, int h, float nms, float thresh, int * totaldetsnum);

void* inferx_get_ocl_context(inferx_context ctx);


/** @breif  Get model result or intermediate feature map.
    @datail  This function is for debugging model. You can  extract intermediate feature map data.
    @param[in]  ctx		The context to manage this model.
    @param[in]  layer_name	The name of the layer from <prototxt> to get result data.
                             	The layer name can get from <model.prototxt>,
                                such as layer defination below,
                                layer {
                                  name: "prob"
                                  type: "Softmax"
                                  bottom: "pool10"
                                  top: "prob"
                                }
                                layer_name = 'prob'

    @param[out] **pout          The  pointer to point result data array.
    @param[out] len             The output data array length;
*/
void inferx_get_result(inferx_context ctx, char *layer_name, void **pout, int *len);

/** @brief  Clear the context
 *  @datail  Destroy the memory of context, including feature map and weights.
 *  @param[in]  ctx             The context to be released.
*/
void inferx_destroy_context(inferx_context ctx);

#ifdef __cplusplus
}
#endif //cpp

