#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct isa_info {
	bool has_avx;
	bool has_fma3;
	bool has_avx2;
};

struct cache_info {
	uint32_t size;
	uint32_t associativity;
	uint32_t threads;
	bool inclusive;
};

struct cache_hierarchy_info {
	struct cache_info l1;
	struct cache_info l2;
	struct cache_info l3;
	struct cache_info l4;
};

struct cache_blocking_info {
	size_t l1;
	size_t l2;
	size_t l3;
	size_t l4;
};

	#define PERFDNN_COMPLEX_TUPLE_INDEX 1

//typedef void (*perfdnn_transform_2d)(const void*, void*, size_t, size_t, uint32_t, uint32_t);
typedef void (*perfdnn_transform_2d_with_bias)(const float*, float*, const float*, size_t, size_t, uint32_t, uint32_t);
//typedef void (*perfdnn_transform_2d)(const float*, float*,  const float*,size_t, size_t, uint32_t, uint32_t);
typedef void (*perfdnn_transform_2d_with_offset)(const float*, float*, size_t, size_t, uint32_t, uint32_t, uint32_t, uint32_t);

typedef void (*perfdnn_blockmac)(float*, const float*, const float*);

typedef void (*perfdnn_fast_sgemm_function)(size_t, size_t, const float*, const float*, float*, size_t);
typedef void (*perfdnn_full_sgemm_function)(uint32_t, uint32_t, size_t, size_t, const float*, const float*, float*, size_t);

typedef void (*perfdnn_fast_conv_function)(size_t, size_t, const float*, const float*, float*);
typedef void (*perfdnn_full_conv_function)(uint32_t, uint32_t, size_t, size_t, const float*, const float*, float*);

typedef void (*perfdnn_fast_tuple_gemm_function)(size_t, size_t, const float*, const float*, float*, size_t);
typedef void (*perfdnn_full_tuple_gemm_function)(uint32_t, uint32_t, size_t, size_t, const float*, const float*, float*, size_t);

typedef void (*perfdnn_sdotxf_function)(const float*, const float*, size_t, float*, size_t);
typedef void (*perfdnn_shdotxf_function)(const float*, const void*, size_t, float*, size_t);

typedef void (*perfdnn_relu_function)(const float*, float*, size_t, float);
typedef void (*perfdnn_inplace_relu_function)(float*, size_t, float);

typedef void (*perfdnn_softmax_function)(size_t, const float*, float*);
typedef void (*perfdnn_inplace_softmax_function)(size_t, float*);

//struct transforms {
//	perfdnn_transform_2d_with_offset iwt_f6x6_3x3_with_offset_and_stream;
//	perfdnn_transform_2d_with_offset kwt_f6x6_3x3;
//	perfdnn_transform_2d_with_bias owt_f6x6_3x3_with_bias;
//	perfdnn_transform_2d_with_bias owt_f6x6_3x3;
//};

struct blockmac {
	perfdnn_blockmac fourier8x8_mac_with_conj;
	perfdnn_blockmac fourier16x16_mac_with_conj;
	perfdnn_blockmac winograd8x8_mac;
};

struct activations {
	perfdnn_relu_function relu;
	perfdnn_inplace_relu_function inplace_relu;
	perfdnn_softmax_function softmax;
	perfdnn_inplace_softmax_function inplace_softmax;
};

struct convolution {
	perfdnn_fast_conv_function only_mr_x_nr;
	perfdnn_full_conv_function upto_mr_x_nr;
	uint32_t mr;
	uint32_t nr;
};

struct sgemm {
	perfdnn_fast_sgemm_function only_mr_x_nr;
	perfdnn_full_sgemm_function upto_mr_x_nr;
	uint32_t mr;
	uint32_t nr;
};

struct sxgemm {
	perfdnn_fast_tuple_gemm_function only_mr_x_nr;
	perfdnn_full_tuple_gemm_function upto_mr_x_nr;
	uint32_t mr;
	uint32_t nr;
};


struct cxgemm {
	perfdnn_fast_tuple_gemm_function s4cX_only_mr_x_nr;
	perfdnn_full_tuple_gemm_function s4cX_upto_mr_x_nr;
	perfdnn_fast_tuple_gemm_function cX_only_mr_x_nr;
	perfdnn_full_tuple_gemm_function cX_upto_mr_x_nr;
	perfdnn_fast_tuple_gemm_function s4cX_conjb_only_mr_x_nr;
	perfdnn_full_tuple_gemm_function s4cX_conjb_upto_mr_x_nr;
	perfdnn_fast_tuple_gemm_function cX_conjb_only_mr_x_nr;
	perfdnn_full_tuple_gemm_function cX_conjb_upto_mr_x_nr;
	perfdnn_fast_tuple_gemm_function s4cX_conjb_transc_only_mr_x_nr;
	perfdnn_full_tuple_gemm_function s4cX_conjb_transc_upto_mr_x_nr;
	perfdnn_fast_tuple_gemm_function cX_conjb_transc_only_mr_x_nr;
	perfdnn_full_tuple_gemm_function cX_conjb_transc_upto_mr_x_nr;
	uint32_t mr;
	uint32_t nr;
};

struct sdotxf {
	const perfdnn_sdotxf_function* functions;
	uint32_t fusion;
};

struct shdotxf {
	const perfdnn_shdotxf_function* functions;
	uint32_t fusion;
};

struct hardware_info {
	bool initialized;
	bool supported;
	uint32_t simd_width;

	struct cache_hierarchy_info cache;
	struct cache_blocking_info blocking;

//	struct transforms transforms;
	struct blockmac blockmac;
	struct activations activations;
	struct convolution conv1x1;
	struct sgemm sgemm;
	struct sxgemm sxgemm;
	struct cxgemm cxgemm;
	struct sdotxf sdotxf;
	struct shdotxf shdotxf;

	struct isa_info isa;
};

extern struct hardware_info perfdnn_hwinfo;

#ifdef __cplusplus
} /* extern "C" */
#endif
