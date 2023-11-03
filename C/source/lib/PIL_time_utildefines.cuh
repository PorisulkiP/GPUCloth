#pragma once

#include "utildefines.h" /* for AT */

//#include "PIL_time.h"        /* for PIL_check_seconds_timer */

// Determines the platform on which Google Test is compiled.
#ifdef __CYGWIN__
# define GTEST_OS_CYGWIN 1
# elif defined(__MINGW__) || defined(__MINGW32__) || defined(__MINGW64__)
#  define GTEST_OS_WINDOWS_MINGW 1
#  define GTEST_OS_WINDOWS 1
#elif defined _WIN32
#  include <windows.h>
#  include <winapifamily.h>
# define GTEST_OS_WINDOWS 1
# ifdef _WIN32_WCE
#  define GTEST_OS_WINDOWS_MOBILE 1
# elif defined(WINAPI_FAMILY)
#  include <winapifamily.h>
#  if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#   define GTEST_OS_WINDOWS_DESKTOP 1
#  elif WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_PHONE_APP)
#   define GTEST_OS_WINDOWS_PHONE 1
#  elif WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP)
#   define GTEST_OS_WINDOWS_RT 1
#  elif WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_TV_TITLE)
#   define GTEST_OS_WINDOWS_PHONE 1
#   define GTEST_OS_WINDOWS_TV_TITLE 1
#  else
    // WINAPI_FAMILY defined but no known partition matched.
    // Default to desktop.
#   define GTEST_OS_WINDOWS_DESKTOP 1
#  endif
# else
#  define GTEST_OS_WINDOWS_DESKTOP 1
# endif  // _WIN32_WCE
#elif defined __OS2__
# define GTEST_OS_OS2 1
#elif defined __APPLE__
# define GTEST_OS_MAC 1
# if TARGET_OS_IPHONE
#  define GTEST_OS_IOS 1
# endif
#elif defined __DragonFly__
# define GTEST_OS_DRAGONFLY 1
#elif defined __FreeBSD__
# define GTEST_OS_FREEBSD 1
#elif defined __Fuchsia__
# define GTEST_OS_FUCHSIA 1
#elif defined(__GLIBC__) && defined(__FreeBSD_kernel__)
# define GTEST_OS_GNU_KFREEBSD 1
#elif defined __linux__
# define GTEST_OS_LINUX 1
# if defined __ANDROID__
#  define GTEST_OS_LINUX_ANDROID 1
# endif
#elif defined __MVS__
# define GTEST_OS_ZOS 1
#elif defined(__sun) && defined(__SVR4)
# define GTEST_OS_SOLARIS 1
#elif defined(_AIX)
# define GTEST_OS_AIX 1
#elif defined(__hpux)
# define GTEST_OS_HPUX 1
#elif defined __native_client__
# define GTEST_OS_NACL 1
#elif defined __NetBSD__
# define GTEST_OS_NETBSD 1
#elif defined __OpenBSD__
# define GTEST_OS_OPENBSD 1
#elif defined __QNX__
# define GTEST_OS_QNX 1
#elif defined(__HAIKU__)
#define GTEST_OS_HAIKU 1
#endif  // __CYGWIN__

#define WINAPI_FAMILY_PARTITION(Partitions)     (Partitions)
#pragma region Application Family or OneCore Family or Games Family
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_SYSTEM | WINAPI_PARTITION_GAMES)

#endif // WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_SYSTEM | WINAPI_PARTITION_GAMES)
#pragma endregion

static cudaEvent_t start, stop;

__host__ inline double PIL_check_seconds_timer(void)
{
    static int hasperfcounter = -1; /* (-1 == unknown) */
    static double perffreq;

    if (hasperfcounter == -1)
    {
        __int64 ifreq;
        hasperfcounter = QueryPerformanceFrequency((LARGE_INTEGER*)&ifreq);
        perffreq = (double)ifreq;
    }

    if (hasperfcounter)
    {
        __int64 count;

        QueryPerformanceCounter((LARGE_INTEGER*)&count);

        return count / perffreq;
    }
    static double accum = 0.0;
    static int ltick = 0;
    const int ntick = GetTickCount();

    if (ntick < ltick) {
	    accum += (0xFFFFFFFF - ltick + ntick) / 1000.0;
    }
    else {
	    accum += (ntick - ltick) / 1000.0;
    }

    ltick = ntick;
    return accum;
}

#define TIMEIT_START(var) \
  { \
    double _timeit_##var = PIL_check_seconds_timer(); \
    printf("time start (" #var "):  " AT "\n"); \
    fflush(stdout); \
    { \
      (void)0
/**
 * \return the time since TIMEIT_START was called.
 */
#define TIMEIT_VALUE(var) (float)(PIL_check_seconds_timer() - _timeit_##var)

#define TIMEIT_VALUE_PRINT(var) \
  { \
    printf("time update   (" #var \
           "): %.6f" \
           "  " AT "\n", \
           TIMEIT_VALUE(var)); \
    fflush(stdout); \
  } \
  (void)0

#define TIMEIT_END(var) \
  } \
  printf("time end   (" #var \
         "): %.6f" \
         "  " AT "\n", \
         TIMEIT_VALUE(var)); \
  fflush(stdout); \
  } \
  (void)0

/**
 * _AVERAGED variants do same thing as their basic counterpart,
 * but additionally add elapsed time to an averaged static value,
 * useful to get sensible timing of code running fast and often.
 */
#define TIMEIT_START_AVERAGED(var) \
  { \
    static float _sum_##var = 0.0f; \
    static float _num_##var = 0.0f; \
    double _timeit_##var = PIL_check_seconds_timer(); \
    printf("time start    (" #var "):  " AT "\n"); \
    fflush(stdout); \
    { \
      (void)0

#define TIMEIT_AVERAGED_VALUE(var) (_num##var ? (_sum_##var / _num_##var) : 0.0f)

#define TIMEIT_END_AVERAGED(var) \
  } \
  const float _delta_##var = TIMEIT_VALUE(var); \
  _sum_##var += _delta_##var; \
  _num_##var++; \
  printf("time end      (" #var \
         "): %.6f" \
         "  " AT "\n", \
         _delta_##var); \
  printf("time averaged (" #var "): %.6f (total: %.6f, in %d runs)\n", \
         (_sum_##var / _num_##var), \
         _sum_##var, \
         (int)_num_##var); \
  fflush(stdout); \
  } \
  (void)0

/**
 * Given some function/expression:
 *   TIMEIT_BENCH(some_function(), some_unique_description);
 */
#define TIMEIT_BENCH(expr, id) \
  { \
    TIMEIT_START(id); \
    (expr); \
    TIMEIT_END(id); \
  } \
  (void)0

#define TIMEIT_BLOCK_INIT(id) double _timeit_var_##id = 0

#define TIMEIT_BLOCK_START(id) \
  { \
    double _timeit_block_start_##id = PIL_check_seconds_timer(); \
    { \
      (void)0

#define TIMEIT_BLOCK_END(id) \
  } \
  _timeit_var_##id += (PIL_check_seconds_timer() - _timeit_block_start_##id); \
  } \
  (void)0

#define TIMEIT_BLOCK_STATS(id) \
  { \
    printf("%s time (in seconds): %f\n", #id, _timeit_var_##id); \
    fflush(stdout); \
  } \
  (void)0


#define CUDA_TIMEIT_START(var) \
  { \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start);

#define CUDA_TIMEIT_END(var) \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    float milliseconds = 0; \
    cudaEventElapsedTime(&milliseconds, start, stop); \
    printf("time end   (" #var "): %.6f seconds\n", milliseconds / 1000.0f); \
    cudaEventDestroy(start); \
    cudaEventDestroy(stop); \
  } \
  (void)0
