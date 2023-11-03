#pragma once

#if defined(_MSC_VER)
#  define alloca _alloca
#endif

#if (defined(__GNUC__) || defined(__clang__)) && defined(__cplusplus)
extern "C++" {
/* Some magic to be sure we don't have reference in the type. */
template<typename T> static inline T decltype_helper(T x)
{
  return x;
}
#define typeof(x) decltype(decltype_helper(x))
}
#endif

#if defined(__GNUC__)
#  define BLI_NOINLINE __attribute__((noinline))
#else
#  define BLI_NOINLINE
#endif
