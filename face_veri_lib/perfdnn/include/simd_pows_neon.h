#ifndef _SIMD_POWS_KERNEL_H_
#define _SIMD_POWS_KERNEL_H_

#include "simd_const.h"
#include "simd_map.h"
#include "simd_exps_neon.h"
#include "simd_lns_neon.h"

_PI32_CONST128(pow_uint_1, 1);
_PI32_CONST128(pow_uint_0, 0);
_PS128_CONST_TYPE(pow_nan, int, 0xffffffff);

static inline float32x4_t simd_pow4f(const float32x4_t aa, const float32x4_t bb)
{
    float32x4_t tem0, tem1, tem2, man;
    float32x4_t abs_a, abs_b, floor_b;
    float32x4_t out1, sign;
    float32x4_t out2, odd;

    const float32x4_t zero = simd_zeros;
    const float32x4_t mask = *(float32x4_t *) _ps128_sign_mask;
    const float32x4_t two = *(float32x4_t *) _ps128_2;
    const float32x4_t sone = *(float32x4_t *) _ps128_1;
    const float32x4_t nan = *(float32x4_t *) _ps128_pow_nan;

    const int32x4_t uint_1 = *(int32x4_t *) _pi32_128_pow_uint_1;
    const int32x4_t uint_0 = *(int32x4_t *) _pi32_128_pow_uint_0;
    int32x4_t exp;
    float32x4_t cc;

    {
        sign = simd_ands(mask, aa);
        abs_a = simd_fabs(aa);    //|a|
        abs_b = simd_fabs(bb);    //|b|

        /* using ln fuction */ 
        tem0 = simd_ln4f(abs_a);
        tem1 = simd_muls(tem0, bb);

        /* using exp fuction */ 
        cc = simd_exp4f(tem1);

        exp = simd_cvts_w(bb);

        man = simd_castu_s(simd_cmpeqw(simd_andw(exp, uint_1), uint_0));   //even or odd

        odd = simd_ors(sign, cc);
        out1 = simd_adds(simd_ands(man, cc), simd_andnots(man, odd));

        floor_b = simd_floors(bb);

        /* x<0 and y != N, then -NAN */
        man = simd_andnots(simd_cmpeqs(bb, floor_b), simd_cmplts(aa, zero));
        out2 = simd_adds(simd_ands(man, nan), simd_andnots(man, out1));

        /* y = 0 then 1 */
        man = simd_cmpeqs(abs_b, zero);
        cc = simd_adds(simd_ands(man, sone), simd_andnots(man, out2));
        return cc;
    }
}

#endif /* _SIMD_POWS_KERNEL_H_ */
