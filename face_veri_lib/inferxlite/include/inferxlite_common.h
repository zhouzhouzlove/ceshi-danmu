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
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<stdint.h>
#include <time.h>
#include <stdbool.h>

#ifdef _MSC_VER
    #include <time.h>
    #include <winsock2.h>

    // Windows下要在此定义这些宏
    #define INFERXLITE_CPU
    #define INFERXLITE_OPENBLAS
    #define INFERXLITE_NATURE
#else
    #include <sys/time.h>
#endif // _MSC_VER

#include "metafmt.h"
#ifdef CUDNN
#include "cuda.h"
#include "cudnn.h"
#endif
#ifdef INFERXLITE_PERFDNN_OCL
#include "core/core.h"
#include "ocl/ocl.h"
#endif

#ifndef INCLUDE_COMMON_H_
#define INCLUDE_COMMON_H_
#ifdef __cplusplus
extern "C"
{
#endif //cpp

#ifndef FREE(pvoid)
    #define FREE(pvoid) \
    if (NULL != (pvoid)) { \
        void** pvTmp = (void**)&(pvoid); \
        free(*pvTmp); \
        *pvTmp = NULL; \
    }
#endif // FREE

#ifdef INFERXLITE_FP16

  #ifdef INFERXLITE_ARM_GPU
    typedef __fp16 inferx__fp16;
    #define INFERX_PERFDNN_FLOAT_ PERFDNN_16F
  #else
    typedef float inferx__fp16;
    #define INFERX_PERFDNN_FLOAT_ PERFDNN_32F
  #endif

  // define perfdnn_cl float_16
  #define INFERX_PERFDNN_FLOAT PERFDNN_16F
#else
  #define INFERX_PERFDNN_FLOAT PERFDNN_32F
  #define INFERX_PERFDNN_FLOAT_ PERFDNN_32F
#endif


/*
 * The device type in inferx_device_context.
 */
typedef enum {
  /* cpu mode.*/
  kCPU = 1,
  /* gpu mode. */
  kGPU = 2,
  /*kCPUPinned = kCPU | kGPU. */
  kCPUPinned = 3,
  /* opencl mode. */
  kOpenCL = 4,
  kMetal = 8,
  kVPI = 9,
  kROCM = 10,
}INFERX_DEVICE_TYPE;

/*
 * A Device context for Tensor and operator.
 */
typedef struct{
  /* The device type used in the device. */
  INFERX_DEVICE_TYPE device_type;
  /* The device index */
  int device_id;
}inferx_device_context;

/*
 * The type code options inferx_data_type.
 */
typedef enum{
  kInt = 0U,
  kUInt = 1U,
  kFloat = 2U,
}INFERX_DATA_TYPE_CODE;

/*
 * The data type the tensor can hold.
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes=1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
 *   - int8: type_code = 0, bits = 8, lanes=1
 */
typedef struct
{
  /*
   * Type code of base types.
   * We keep it uint8_t instead of INFERX_DATA_TYPE_CODE for minimal memory
   * footprint, but the value should be one of DLDataTypeCode enum values.
   * */
  uint8_t code;
  /*
   * Number of bits, common choices are 8, 16, 32.
   */
  uint8_t bits;
  /* Number of lanes in the type, used for vector types. */
  uint16_t lanes;
}inferx_data_type;

/*
 * Plain C Tensor object, does not manage memory.
 */
typedef struct
{
  /*
   * The opaque data pointer points to the allocated data.
   *  This will be CUDA device pointer or cl_mem handle in OpenCL.
   *  This pointer is always aligns to 256 bytes as in CUDA.
   */
  void* data;

  int layer_n;
  int layer_classes;
#ifdef INFERXLITE_PERFDNN_OCL
  perf_dnn_ocl_mat_t mat_ocl;
  perf_dnn_ocl_mat_t tmp_ocl;

  perf_dnn_ocl_mat_t tmp1_ocl;
  perf_dnn_ocl_mat_t tmp2_ocl;
  perf_dnn_ocl_mat_t tmp3_ocl;
  perf_dnn_ocl_mat_t tmp4_ocl;
  perf_dnn_ocl_mat_t tmp5_ocl;
  perf_dnn_mat_t mat_cpu;
  perf_dnn_mat_t mat_cpu1;
#endif

#ifdef INFERXLITE_PERFDNN
  void* workspace_buffer;
  size_t workspace_size;
#endif
  
  /* The device context of the tensor */
  inferx_device_context ctx;
  /* Number of dimensions */
  int ndim;
  /* The data type of the pointer*/
  inferx_data_type dtype;
  /* The shape of the tensor */
  // shape[0]: batch size
  // shape[1]: channels size
  // shape[2]: height size
  // shape[3]: width size
  int* shape;
  /*
   * strides of the tensor,
   *  can be NULL, indicating tensor is compact.
   */
  int* strides;
  /* The offset in bytes to the beginning pointer to data */
  int byte_offset;

  void *desc;
}inferx_data_tensor;

/*
  Convlotion method option.
*/
enum INFERX_CONV_WAY
{
  /* direct convolution. */
  conv_nature=0,
  /* convolution based on gemm. */
  conv_gemm=1,
  /* convolution based on commercial lib perfdnn. */
  conv_perfdnn=2
};

/*
  Local Response Normalization option.
*/
enum INFERX_LRN_WAY
{
  /*normalization span channels */
  across_channels=0,
  /*normalization within every single channel */
  within_channels=1
};

/*
  element operation option for pooling and elementwise operaiton
*/
enum INFERX_OPERATION
{
  /* add operation */
  SUM=0,
  /* multiplicative operation */
  PROD=1,
  /* max value */
  MAX=2,
  /* average operation */
  AVE=3
};

/*
  method for resize operation
*/
enum INFERX_RESIZE_METHOD
{
  NEAREST=1,
  BILINEAR=2,
};

/* ????
enum OPERATE
{
  add=0,
  sub=1,
  mul=2,
  division=3
};
*/

/*
  inferx unit operation
*/
enum INFERX_UNIT_OP
{
  convolution=0,
  linear=1,
  batchnorm=2,
  scale=3
};

#ifdef INFERXLITE_CUDNN
struct context_model
{
  cudnnFilterDescriptor_t pFilterDesc;
  cudnnConvolutionDescriptor_t pConvDesc;
  int sizeInBytes;
  void* workSpace;
};
#endif

#ifdef INFERXLITE_CUDNN
struct context_data
{
  cudnnTensorDescriptor_t pDataDesc;
};
#endif

/*
  pipeline to store model weight data
*/
struct inferx_model_pipeline
{
  /* layer  weight data block */
  inferx_data_tensor *weight;
  /* weight number of the current layer */
  int nw;
};

/*
  pipeline to store feature map data
*/
struct inferx_data_pipeline
{
  /* layer feature map data block */
  inferx_data_tensor datas;
};

/*
  weight data argument for layers, convoution and so on
*/
struct inferx_data_arg
{
  /* inferx unit operation*/
  enum INFERX_UNIT_OP uo;
  /* stride height */
  int str_h;
  /* stride width */
  int str_w;
  /* padding height */
  int pad_h;
  /* padding width */
  int pad_w;
};

enum INFERX_TIME_STAT_WAY
{
  time_forward=0,
  time_layer=1,
  time_no=2
};

struct inferx_time_by_layer
{
  double time;
  char *tname;
  struct inferx_time_by_layer *tp;
};

struct inferx_time_event
{
  struct timeval start, finish;
  double total_time;
};

struct inferx_handler;
/*
  define model function pointer
*/
typedef void (*inferx_model_func_pointer)(char *path, char *model, char *data, int *shape, int nshape, void *pdata, void **pout, int *len, struct inferx_handler *hd);

/*
  function pointer map
*/
struct inferx_func_pointer_map
{
  /* model weight data file path */
  char *path;
  /* inferx_model_func_pointer arrary length */
  unsigned int len;
  /* model function name string array */
  char **name;
  /* model function pointer array */
  inferx_model_func_pointer *func;
};

/*
  convolution arguments
*/
struct inferx_conv_arg
{
  int group;
  int input_n;
  int input_c;
  int input_h;
  int input_w;
  int kernel_h;
  int kernel_w;
  int output_c;
  int output_h;
  int output_w;
  int pad_h;
  int pad_w;
  int str_h;
  int str_w;
  int dila_h;
  int dila_w;
};

/*
  inferxlite handler to control dataflow
*/
struct inferx_handler
{
  /*the number of dataflow */
  unsigned int total_data_pipeline;
  /*max number pipeline, the number upper limit of modelflow and dataflow */
  unsigned int max_num_pipeline;
  /* model_pipeline arrary */
  struct inferx_model_pipeline *modelflow;
  /* data_pipeline arrary */
  struct inferx_data_pipeline *dataflow;
  /* flag, whether the modelflow has been loaded,or not */
  bool weight_has_load;
  /* flag, whether the dataflow has been initialized,or not */
  bool data_has_init;
#ifdef INFERXLITE_RAND_WEIGHTS
  /* flag, tag to check whether random weights model has been executed/malloc when first model running */
  bool weight_has_rand_init;
#endif
  /* flag , to decide update input data */
  bool is_update_input;//???
  enum INFERX_TIME_STAT_WAY tsw;
  struct inferx_time_by_layer *time_head;
  struct inferx_time_by_layer *time_tail;
  int time_layer_cnt;
  struct timeval start_forward, start_layer;
  struct timeval tv_begin, tv_end;
  double elasped;
  /* modelflow and dataflow id arrary  */
  uint64_t *elem;
  unsigned int len_elem;
  /* convolution method option */
  enum INFERX_CONV_WAY cw;
  /* device type */
  INFERX_DEVICE_TYPE dvct;
  /* function pointer map */
  struct inferx_func_pointer_map fpm;
  /* the max number of function pointer the func_pointer_map can hold  */
  unsigned int max_num_func;
  /* model name  array in which the model have been initialized */
  char **is_has_init;
  /*model name array length */
  unsigned int len_init;
  /* tag to check whether the hanlder has been initialized */
  int tag;

#ifdef _MSC_VER
  HMODULE hDll;
#else
  void *  hDll;
#endif

#ifdef INFERXLITE_CUDNN
  cudnnHandle_t hCudNN;
#endif

#ifdef INFERXLITE_PERFDNN_OCL
  bool is_upload_input;//use cpu to upload data or use gpu directly
  perf_dnn_ocl_context_t perf_ocl_context;
#endif
};

/*
  inferxlite context
*/
struct inferx_context_t
{
  /* model function pointer used to run model */
  inferx_model_func_pointer func;
  /* model name */
  char *model;
  /* feature data prefix */
  char *data;
  /* input data shape */
  int *shape;
  /* input data shape dimension */
  int nshape;
  /* tag to check whether the context has been initialized */
  int tag;
  /* inferxlite handler */
  struct inferx_handler hd;
};

/*
  define inferxlite_context_t pointer
*/
typedef struct inferx_context_t *inferx_context;


#ifdef __cplusplus
}
#endif //cpp

#endif //INCLUDE_COMMON_H_
