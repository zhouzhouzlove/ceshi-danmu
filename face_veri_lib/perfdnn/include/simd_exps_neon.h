#ifndef _SIMD_EXP_KERNEL_H_
#define _SIMD_EXP_KERNEL_H_

#include "simd_const.h"
#include "simd_map.h"

_PS128_CONST(exp_LOG2EF, 1.44269504088896341f);
_PS128_CONST(exp_C1, 0.693359375f);
_PS128_CONST(exp_C2, -2.12194440e-4f);
_PS128_CONST(exp_p0, 1.9875691500E-4f);
_PS128_CONST(exp_p1, 1.3981999507E-3f);
_PS128_CONST(exp_p2, 8.3334519073E-3f);
_PS128_CONST(exp_p3, 4.1665795894E-2f);
_PS128_CONST(exp_p4, 1.6666665459E-1f);
_PS128_CONST(exp_p5, 5.0000001201E-1f);
_PS128_CONST_TYPE(exp_inf, int, 0x7f800000);
_PS128_CONST(exp_hi, 88.3762626647949f);
_PS128_CONST(exp_lo, -88.3762626647949f);
_PI32_CONST128(exp_0x7f, 0x7f);


static inline float32x4_t simd_exp4f(const float32x4_t a)
{

    float32x4_t b;
    float32x4_t tmp, fx;
    int32x4_t emm0;
    float32x4_t x, y;
    float32x4_t mask;
    float32x4_t pow2n;
    float32x4_t z, one, tem1;

    // exponent mask
    float32x4_t inf = *(float32x4_t *) _ps128_exp_inf;
    one = *(float32x4_t *) _ps128_1;

    x = simd_mins(a, *(float32x4_t *) _ps128_exp_hi);
    x = simd_maxs(x, *(float32x4_t *) _ps128_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = simd_mas(x, *(float32x4_t *) _ps128_exp_LOG2EF, *(float32x4_t *) _ps128_0p5);
    tmp = simd_floors(fx);

    /* if greater, substract 1 */
    mask = simd_cmplts(fx, tmp);
    mask = simd_ands(mask, one);
    fx = simd_subs(tmp, mask);

    x = simd_nmas(*(float32x4_t *) _ps128_exp_C1, fx, x);
    x = simd_nmas(*(float32x4_t *) _ps128_exp_C2, fx, x);
    z = simd_muls(x, x);

    y = *(float32x4_t *) _ps128_exp_p0;
    tem1 = simd_mas(y, x, *(float32x4_t *) _ps128_exp_p1);
    y = simd_mas(tem1, x, *(float32x4_t *) _ps128_exp_p2);
    tem1 = simd_mas(y, x, *(float32x4_t *) _ps128_exp_p3);
    y = simd_mas(tem1, x, *(float32x4_t *) _ps128_exp_p4);
    tem1 = simd_mas(y, x, *(float32x4_t *) _ps128_exp_p5);
    y = simd_mas(tem1, z, x);
    tem1 = simd_adds(y, one);
    y = tem1;

    /* build 2^n */
    emm0 = simd_cvts_w(fx);
    emm0 = simd_addw(emm0, *(int32x4_t *) _pi32_128_exp_0x7f);
    emm0 = simd_sllw(emm0, 23);
    pow2n = simd_castw_s(emm0);
    y = simd_muls(y, pow2n);

    /* boundary deal */
    mask = simd_cmples(*(float32x4_t *) _ps128_exp_hi, a);
    tem1 = simd_selects(mask, y, inf);
    mask = simd_cmplts(a, *(float32x4_t *) _ps128_exp_lo);
    y = simd_selects(mask, tem1, simd_zeros);

    b = y;
    return b;
}

#endif /* _SIMD_EXP_KERNEL_H_ */
