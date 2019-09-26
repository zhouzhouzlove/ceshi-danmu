#pragma once


#if defined(__GNUC__)
	#if defined(__clang__) || ((__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 5)))
		#define PERFDNN_UNREACHABLE do { __builtin_unreachable(); } while (0)
	#else
		#define PERFDNN_UNREACHABLE do { __builtin_trap(); } while (0)
	#endif
#else
	#define PERFDNN_UNREACHABLE do { } while (0)
#endif


#if defined(PERFDNN_BACKEND_PSIMD)
	#if !(PERFDNN_BACKEND_PSIMD)
		#error PERFDNN_BACKEND_PSIMD predefined as 0
	#endif
#elif defined(PERFDNN_BACKEND_SCALAR)
	#if !(PERFDNN_BACKEND_SCALAR)
		#error PERFDNN_BACKEND_SCALAR predefined as 0
	#endif
#elif defined(__arm__) || defined(__aarch64__)
	#define PERFDNN_BACKEND_ARM 1
#elif defined(__ANDROID__) && (defined(__i686__) || defined(__x86_64__))
	#define PERFDNN_BACKEND_PSIMD 1
#elif defined(__x86_64__)
	#define PERFDNN_BACKEND_X86_64 1
#elif defined(__ANDROID__) && defined(__mips__)
	#define PERFDNN_BACKEND_SCALAR 1
#else
	#define PERFDNN_BACKEND_PSIMD 1
#endif

#ifndef PERFDNN_BACKEND_PSIMD
	#define PERFDNN_BACKEND_PSIMD 0
#endif
#ifndef PERFDNN_BACKEND_SCALAR
	#define PERFDNN_BACKEND_SCALAR 0
#endif
#ifndef PERFDNN_BACKEND_ARM
	#define PERFDNN_BACKEND_ARM 0
#endif
#ifndef PERFDNN_BACKEND_X86_64
	#define PERFDNN_BACKEND_X86_64 0
#endif

#define PERFDNN_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#define PERFDNN_SIMD_ALIGN PERFDNN_ALIGN(64)
#define PERFDNN_CACHE_ALIGN PERFDNN_ALIGN(64)

#define PERFDNN_COUNT_OF(array) (sizeof(array) / sizeof(0[array]))

#if defined(__GNUC__)
	#define PERFDNN_LIKELY(condition) (__builtin_expect(!!(condition), 1))
	#define PERFDNN_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#else
	#define PERFDNN_LIKELY(condition) (!!(condition))
	#define PERFDNN_UNLIKELY(condition) (!!(condition))
#endif

#if defined(__GNUC__)
	#define PERFDNN_INLINE inline __attribute__((__always_inline__))
#else
	#define PERFDNN_INLINE inline
#endif
