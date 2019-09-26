#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


void perfdnn_relu__neon(const float* input, float* output, size_t length, float negative_slope);
void perfdnn_inplace_relu__neon(float* data, size_t length, float negative_slope);


#ifdef __cplusplus
} /* extern "C" */
#endif
