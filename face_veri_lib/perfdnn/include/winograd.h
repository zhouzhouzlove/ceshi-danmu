#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*perfdnn_wt_function)(const float*, float*);

void perfdnn_iwt_f6k3__fma3(const float d[], float w[]);
void perfdnn_kwt_f6k3__fma3(const float g[], float w[]);
void perfdnn_owt_f6k3__fma3(const float m[], float s[]);

void perfdnn_iwt_f6k3__psimd(const float d[], float w[]);
void perfdnn_kwt_f6k3__psimd(const float g[], float w[]);
void perfdnn_owt_f6k3__psimd(const float m[], float s[]);

void perfdnn_iwt_f6k3__neon(const float d[], float w[]);
void perfdnn_kwt_f6k3__neon(const float g[], float w[]);
void perfdnn_owt_f6k3__neon(const float m[], float s[]);

void perfdnn_iwt_f6k3__scalar(const float d[], float w[]);
void perfdnn_kwt_f6k3__scalar(const float g[], float w[]);
void perfdnn_owt_f6k3__scalar(const float m[], float s[]);

#ifdef __cplusplus
} /* extern "C" */
#endif
