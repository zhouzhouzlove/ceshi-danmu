#ifndef _SIMD_RELU_KERNEL_H_
#define _SIMD_RELU_KERNEL_H_
#include <math.h>
#include <perfdnn/arm_neon_.h>

static inline float32x4_t neon_relu_f32(float32x4_t data, float32x4_t negative_slope) 
{
    const uint32x4_t negative_mask = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(data), 31));
    return vbslq_f32(negative_mask, vmulq_f32(data, negative_slope), data);
}

#endif
