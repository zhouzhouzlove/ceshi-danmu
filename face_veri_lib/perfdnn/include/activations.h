#pragma once
#include "activation/simd_lns_neon.h"
#include "activation/simd_exps_neon.h"
#include "activation/simd_pows_neon.h"
#include "activation/simd_relu_neon.h"
#include "activation/simd_tanhs_neon.h"

static inline float relu(float data, float negative_slope) 
{
    return signbit(data) ? data * negative_slope : data;
}

void perfdnn_pow_noscale_noshift__neon(
        const float input[restrict static 4],
        float output[restrict static 4],
        size_t length,
        float power
        );
void perfdnn_pow__neon(
        const float input[restrict static 4],
        float output[restrict static 4],
        size_t length,
        float power,
        float scale,
        float shift
        );
void perfdnn_tanh__neon(
        const float input[restrict static 4],
        float output[restrict static 4],
        size_t length);
void perfdnn_inplace_tanh__neon(
        float data[restrict static 4],
        size_t length);
void perfdnn_relu__neon(
	const float input[restrict static 4],
	float output[restrict static 4],
	size_t length,
	float negative_slope);
void perfdnn_inplace_relu__neon(
	float data[restrict static 4],
	size_t length,
	float negative_slope);

float perfdnn_softmax__neon(
        const float input[restrict static 4],
        float output[restrict static 4],
        size_t length
        );
void perfdnn_softmax_average__neon(
        float output[restrict static 4],
        float sum_value,
        size_t length
        );
float perfdnn_logsoftmax__neon(
        const float input[restrict static 4],
        float output[restrict static 4],
        size_t length,
        float max_value
        );
void perfdnn_logsoftmax_support__neon(
        float input[restrict static 4],
        float output[restrict static 4],
        size_t length,
        float sum_value,
        float max_value
        );
void perfdnn_sigmoid__neon(
        const float input[restrict static 4],
        float output[restrict static 4],
        size_t length);
