#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void perfdnn_iwt8x8_3x3_with_offset__neon(const float d[], float wd[], size_t stride_d, size_t stride_wd, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
void perfdnn_kwt8x8_3x3__neon(const float g[], float wg[], size_t stride_g, size_t stride_wg, uint32_t, uint32_t, uint32_t, uint32_t);
void perfdnn_owt8x8_3x3__neon(const float m[], float s[], size_t stride_m, size_t stride_s, uint32_t row_count, uint32_t column_count, uint32_t, uint32_t);
void perfdnn_owt8x8_3x3_with_bias__neon(const float m[], float s[], const float bias[], size_t stride_m, size_t stride_s, uint32_t row_count, uint32_t column_count);
void perfdnn_owt8x8_3x3_with_bias_with_relu__neon(const float m[], float s[], const float bias[], size_t stride_m, size_t stride_s, uint32_t row_count, uint32_t column_count);
void perfdnn_owt8x8_3x3_with_relu__neon(const float m[], float s[], const float bias[], size_t stride_m, size_t stride_s, uint32_t row_count, uint32_t column_count);
void perfdnn_owt8x8_3x3_with_bias_with_relu__neon(const float transform[restrict static 1],float output[restrict static 1],const float bias[restrict static 1],size_t transform_stride, size_t output_stride,uint32_t row_count, uint32_t column_count);

void perfdnn_iwt8x8_3x3_fp16_with_offset__neonhp(const float d[], void* wd, size_t stride_d, size_t stride_wd, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
void perfdnn_kwt8x8_3x3_fp16__neonhp(const float g[], void* wg, size_t stride_g, size_t stride_wg, uint32_t, uint32_t, uint32_t, uint32_t);
void perfdnn_owt8x8_3x3_fp16__neonhp(const void* m, float* s, size_t stride_m, size_t stride_s, uint32_t row_count, uint32_t column_count, uint32_t, uint32_t);
void perfdnn_owt8x8_3x3_fp16_with_bias__neonhp(const void* m, float* s, const float bias[], size_t stride_m, size_t stride_s, uint32_t row_count, uint32_t column_count);

void perfdnn_ft8x8gemmc__fma3(float acc[], const float x[], const float y[]);
void perfdnn_ft16x16gemmc__fma3(float acc[], const float x[], const float y[]);
void perfdnn_s8x8gemm__fma3(float acc[], const float x[], const float y[]);

void perfdnn_ft8x8gemmc__psimd(float acc[], const float x[], const float y[]);
void perfdnn_ft16x16gemmc__psimd(float acc[], const float x[], const float y[]);
void perfdnn_s8x8gemm__psimd(float acc[], const float x[], const float y[]);

#ifdef __cplusplus
} /* extern "C" */
#endif
