#ifndef __ATOMIC_OPS_UTILS_H__
#define __ATOMIC_OPS_UTILS_H__

#include <stdint.h>
#include <stdlib.h>

#include <assert.h>

/* little macro so inline keyword works */
#if defined(_MSC_VER)
#  define ATOMIC_INLINE static __forceinline
#else
#  define ATOMIC_INLINE static inline __attribute__((always_inline))
#endif

#ifdef __GNUC__
#  define _ATOMIC_LIKELY(x) __builtin_expect(!!(x), 1)
#  define _ATOMIC_UNLIKELY(x) __builtin_expect(!!(x), 0)
#  define _ATOMIC_MAYBE_UNUSED __attribute__((unused))
#else
#  define _ATOMIC_LIKELY(x) (x)
#  define _ATOMIC_UNLIKELY(x) (x)
#  define _ATOMIC_MAYBE_UNUSED
#endif

#if defined(__SIZEOF_POINTER__)
#  define LG_SIZEOF_PTR __SIZEOF_POINTER__
#elif defined(UINTPTR_MAX)
#  if (UINTPTR_MAX == 0xFFFFFFFF)
#    define LG_SIZEOF_PTR 4
#  elif (UINTPTR_MAX == 0xFFFFFFFFFFFFFFFF)
#    define LG_SIZEOF_PTR 8
#  endif
#elif defined(__WORDSIZE) /* Fallback for older glibc and cpp */
#  if (__WORDSIZE == 32)
#    define LG_SIZEOF_PTR 4
#  elif (__WORDSIZE == 64)
#    define LG_SIZEOF_PTR 8
#  endif
#endif

#ifndef LG_SIZEOF_PTR
#  error "Cannot find pointer size"
#endif

#if (UINT_MAX == 0xFFFFFFFF)
#  define LG_SIZEOF_INT 4
#elif (UINT_MAX == 0xFFFFFFFFFFFFFFFF)
#  define LG_SIZEOF_INT 8
#else
#  error "Cannot find int size"
#endif

/* Copied from BLI_utils... */
/* C++ can't use _Static_assert, expects static_assert() but c++0x only,
 * Coverity also errors out. */
#if (!defined(__cplusplus)) && (!defined(__COVERITY__)) && \
    (defined(__GNUC__) && ((__GNUC__ * 100 + __GNUC_MINOR__) >= 406)) /* gcc4.6+ only */
#  define ATOMIC_STATIC_ASSERT(a, msg) __extension__ _Static_assert(a, msg);
#else
/* Code adapted from http://www.pixelbeat.org/programming/gcc/static_assert.html */
/* Note we need the two concats below because arguments to ## are not expanded, so we need to
 * expand __LINE__ with one indirection before doing the actual concatenation. */
#  define ATOMIC_ASSERT_CONCAT_(a, b) a##b
#  define ATOMIC_ASSERT_CONCAT(a, b) ATOMIC_ASSERT_CONCAT_(a, b)
/* These can't be used after statements in c89. */
#  if defined(__COUNTER__) /* MSVC */
#    define ATOMIC_STATIC_ASSERT(a, msg) \
      ; \
      enum { ATOMIC_ASSERT_CONCAT(static_assert_, __COUNTER__) = 1 / (int)(!!(a)) };
#  else /* older gcc, clang... */
/* This can't be used twice on the same line so ensure if using in headers
 * that the headers are not included twice (by wrapping in #ifndef...#endif)
 * Note it doesn't cause an issue when used on same line of separate modules
 * compiled with gcc -combine -fwhole-program. */
#    define ATOMIC_STATIC_ASSERT(a, msg) \
      ; \
      enum { ATOMIC_ASSERT_CONCAT(assert_line_, __LINE__) = 1 / (int)(!!(a)) };
#  endif
#endif

#endif /* __ATOMIC_OPS_UTILS_H__ */
