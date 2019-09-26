#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


void perfdnn_sgemm_only_4x12__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void perfdnn_sgemm_upto_4x12__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
#if 1
void perfdnn_s4gemm_only_3x3__neon(
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c);

void perfdnn_s4gemm_upto_3x3__neon(
	uint32_t mr, uint32_t nr,
	size_t k, size_t update,
	const float a[restrict static 1],
	const float b[restrict static 1],
	float c[restrict static 1],
	size_t row_stride_c);
#endif


void perfdnn_conv1x1_only_2x4__neon(size_t input_channels, size_t image_size, const float* input, const float* kernel, float* output);
void perfdnn_conv1x1_upto_2x4__neon(uint32_t mr, uint32_t nr, size_t input_channels, size_t image_size, const float* input, const float* kernel, float* output);

void perfdnn_s4gemm_only_3x4__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c,size_t column_stride);
void perfdnn_s4gemm_upto_3x4__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c,size_t column_stride);

void perfdnn_s4gemm_only_4x4__aarch64(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c,size_t column_stride);
void perfdnn_s4gemm_only_4x4__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c,size_t column_stride);
void perfdnn_s4gemm_upto_4x4__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c,size_t column_stride);

void perfdnn_c4gemm_only_2x2__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void perfdnn_c4gemm_upto_2x2__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void perfdnn_c4gemm_conjb_only_2x2__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void perfdnn_c4gemm_conjb_upto_2x2__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void perfdnn_c4gemm_conjb_transc_only_2x2__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void perfdnn_c4gemm_conjb_transc_upto_2x2__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void perfdnn_s4c2gemm_only_2x2__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void perfdnn_s4c2gemm_upto_2x2__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void perfdnn_s4c2gemm_conjb_only_2x2__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void perfdnn_s4c2gemm_conjb_upto_2x2__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void perfdnn_s4c2gemm_conjb_transc_only_2x2__neon(size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);
void perfdnn_s4c2gemm_conjb_transc_upto_2x2__neon(uint32_t mr, uint32_t nr, size_t k, size_t update, const float* a, const float* b, float* c, size_t row_stride_c);

void perfdnn_sdotxf1__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void perfdnn_sdotxf2__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void perfdnn_sdotxf3__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void perfdnn_sdotxf4__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void perfdnn_sdotxf5__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void perfdnn_sdotxf6__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void perfdnn_sdotxf7__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);
void perfdnn_sdotxf8__neon(const float* x, const float* y, size_t stride_y, float* sum, size_t n);

#ifdef __cplusplus
} /* extern "C" */
#endif
