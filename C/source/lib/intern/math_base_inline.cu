#ifndef __MATH_BASE_INLINE_C__
#define __MATH_BASE_INLINE_C__

#include <cstdio>
#include <cstdlib>

#include "math_base.h"


/* copied from utildefines.h */
#ifdef __GNUC__
#  define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#  define UNLIKELY(x) (x)
#endif

/* powf is really slow for raising to integer powers. */
float pow2f(const float x)
{
  return x * x;
}
float pow3f(const float x)
{
  return pow2f(x) * x;
}
float pow4f(const float x)
{
  return pow2f(pow2f(x));
}
float pow5f(const float x)
{
  return pow4f(x) * x;
}
float pow7f(const float x)
{
  return pow2f(pow3f(x)) * x;
}

float sqrt3f(const float f)
{
  if (UNLIKELY(f == 0.0f)) {
    return 0.0f;
  }
  else if (UNLIKELY(f < 0.0f)) {
    return -(float)(exp(log(-f) / 3.0));
  }
  else {
    return (float)(exp(log(f) / 3.0));
  }
}

double sqrt3d(const double d)
{
  if (UNLIKELY(d == 0.0)) {
    return 0.0;
  }
  else if (UNLIKELY(d < 0.0)) {
    return -exp(log(-d) / 3.0);
  }
  else {
    return exp(log(d) / 3.0);
  }
}

float sqrtf_signed(const float f)
{
  return (f >= 0.0f) ? sqrtf(f) : -sqrtf(-f);
}

float saacos(const float fac)
{
  if (UNLIKELY(fac <= -1.0f)) {
    return (float)M_PI;
  }
  else if (UNLIKELY(fac >= 1.0f)) {
    return 0.0f;
  }
  else {
    return acosf(fac);
  }
}

float saasin(const float fac)
{
  if (UNLIKELY(fac <= -1.0f)) {
    return (float)-M_PI / 2.0f;
  }
  else if (UNLIKELY(fac >= 1.0f)) {
    return (float)M_PI / 2.0f;
  }
  else {
    return asinf(fac);
  }
}

float sasqrt(const float fac)
{
  if (UNLIKELY(fac <= 0.0f)) {
    return 0.0f;
  }
  else {
    return sqrtf(fac);
  }
}

float saacosf(const float fac)
{
  if (UNLIKELY(fac <= -1.0f)) {
    return (float)M_PI;
  }
  else if (UNLIKELY(fac >= 1.0f)) {
    return 0.0f;
  }
  else {
    return acosf(fac);
  }
}

float saasinf(const float fac)
{
  if (UNLIKELY(fac <= -1.0f)) {
    return (float)-M_PI / 2.0f;
  }
  else if (UNLIKELY(fac >= 1.0f)) {
    return (float)M_PI / 2.0f;
  }
  else {
    return asinf(fac);
  }
}

float sasqrtf(const float fac)
{
  if (UNLIKELY(fac <= 0.0f)) {
    return 0.0f;
  }
  else {
    return sqrtf(fac);
  }
}

float interpf(const float target, const float origin, const float fac)
{
  return (fac * target) + (1.0f - fac) * origin;
}

double interpd(const double target, const double origin, const double fac)
{
  return (fac * target) + (1.0f - fac) * origin;
}

/* used for zoom values*/
float power_of_2(const float val)
{
  return (float)pow(2.0, ceil(log((double)val) / M_LN2));
}

int is_power_of_2_i(const int n)
{
  return (n & (n - 1)) == 0;
}

int power_of_2_max_i(int n)
{
  if (is_power_of_2_i(n)) {
    return n;
  }

  do {
    n = n & (n - 1);
  } while (!is_power_of_2_i(n));

  return n * 2;
}

int power_of_2_min_i(int n)
{
  while (!is_power_of_2_i(n)) {
    n = n & (n - 1);
  }

  return n;
}

uint power_of_2_max_u(uint x)
{
  x -= 1;
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  return x + 1;
}

uint power_of_2_min_u(uint x)
{
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  return x - (x >> 1);
}

uint log2_floor_u(const uint x)
{
  return x <= 1 ? 0 : 1 + log2_floor_u(x >> 1);
}

uint log2_ceil_u(const uint x)
{
  if (is_power_of_2_i((int)x)) {
    return log2_floor_u(x);
  }
  else {
    return log2_floor_u(x) + 1;
  }
}

/* rounding and clamping */

#define _round_clamp_fl_impl(arg, ty, min, max) \
  { \
    float r = floorf(arg + 0.5f); \
    if (UNLIKELY(r <= (float)min)) { \
      return (ty)min; \
    } \
    else if (UNLIKELY(r >= (float)max)) { \
      return (ty)max; \
    } \
    else { \
      return (ty)r; \
    } \
  }

#define _round_clamp_db_impl(arg, ty, min, max) \
  { \
    double r = floor(arg + 0.5); \
    if (UNLIKELY(r <= (double)min)) { \
      return (ty)min; \
    } \
    else if (UNLIKELY(r >= (double)max)) { \
      return (ty)max; \
    } \
    else { \
      return (ty)r; \
    } \
  }

#define _round_fl_impl(arg, ty) \
  { \
    return (ty)floorf(arg + 0.5f); \
  }
#define _round_db_impl(arg, ty) \
  { \
    return (ty)floor(arg + 0.5); \
  }

signed char round_fl_to_char(const float a){_round_fl_impl(a, signed char)} 
    unsigned char round_fl_to_uchar(const float a){_round_fl_impl(a, unsigned char)} 
    short round_fl_to_short(const float a){_round_fl_impl(a, short)} 
    unsigned short round_fl_to_ushort(const float a){_round_fl_impl(a, unsigned short)} 
    int round_fl_to_int(const float a){_round_fl_impl(a, int)} 
    uint round_fl_to_uint(const float a){_round_fl_impl(a, uint)}

signed char round_db_to_char(const double a){_round_db_impl(a, signed char)} 
    unsigned char round_db_to_uchar(const double a){_round_db_impl(a, unsigned char)} 
    short round_db_to_short(const double a){_round_db_impl(a, short)} 
    unsigned short round_db_to_ushort(const double a){_round_db_impl(a, unsigned short)} 
    int round_db_to_int(const double a){_round_db_impl(a, int)} 
    uint round_db_to_uint(const double a)
{
  _round_db_impl(a, uint)
}

#undef _round_fl_impl
#undef _round_db_impl

signed char round_fl_to_char_clamp(const float a){
    _round_clamp_fl_impl(a, signed char, SCHAR_MIN, SCHAR_MAX)} 
    unsigned char round_fl_to_uchar_clamp(const float a){
        _round_clamp_fl_impl(a, unsigned char, 0, UCHAR_MAX)} 
    short round_fl_to_short_clamp(const float a){
        _round_clamp_fl_impl(a, short, SHRT_MIN, SHRT_MAX)} 
    unsigned short round_fl_to_ushort_clamp(const float a){
        _round_clamp_fl_impl(a, unsigned short, 0, USHRT_MAX)} 
    int round_fl_to_int_clamp(const float a){_round_clamp_fl_impl(a, int, INT_MIN, INT_MAX)} 
    uint round_fl_to_uint_clamp(const float a){
        _round_clamp_fl_impl(a, uint, 0, UINT_MAX)}

signed char round_db_to_char_clamp(const double a){
    _round_clamp_db_impl(a, signed char, SCHAR_MIN, SCHAR_MAX)} 
    unsigned char round_db_to_uchar_clamp(const double a){
        _round_clamp_db_impl(a, unsigned char, 0, UCHAR_MAX)} 
    short round_db_to_short_clamp(const double a){
        _round_clamp_db_impl(a, short, SHRT_MIN, SHRT_MAX)} 
    unsigned short round_db_to_ushort_clamp(const double a){
        _round_clamp_db_impl(a, unsigned short, 0, USHRT_MAX)} 
    int round_db_to_int_clamp(const double a){_round_clamp_db_impl(a, int, INT_MIN, INT_MAX)} 
    uint round_db_to_uint_clamp(const double a)
{
  _round_clamp_db_impl(a, uint, 0, UINT_MAX)
}

#undef _round_clamp_fl_impl
#undef _round_clamp_db_impl

/* integer division that rounds 0.5 up, particularly useful for color blending
 * with integers, to avoid gradual darkening when rounding down */
int divide_round_i(const int a, const int b)
{
  return (2 * a + b) / (2 * b);
}

/**
 * Integer division that floors negative result.
 * \note This works like Python's int division.
 */
int divide_floor_i(const int a, const int b)
{
  int d = a / b;
  int r = a % b; /* Optimizes into a single division. */
  return r ? d - ((a < 0) ^ (b < 0)) : d;
}

/**
 * Integer division that returns the ceiling, instead of flooring like normal C division.
 */
uint divide_ceil_u(const uint a, const uint b)
{
  return (a + b - 1) / b;
}

/**
 * modulo that handles negative numbers, works the same as Python's.
 */
int mod_i(const int i, const int n)
{
  return (i % n + n) % n;
}

float fractf(const float a)
{
  return a - floorf(a);
}

/* Adapted from godot-engine math_funcs.h. */
float wrapf(const float value, const float max, const float min)
{
  float range = max - min;
  return (range != 0.0f) ? value - (range * floorf((value - min) / range)) : min;
}

float pingpongf(const float value, const float scale)
{
  if (scale == 0.0f) {
    return 0.0f;
  }
  return fabsf(fractf((value - scale) / (scale * 2.0f)) * scale * 2.0f - scale);
}

// Square.

int square_s(const short a)
{
  return a * a;
}

int square_i(const int a)
{
  return a * a;
}

uint square_uint(const uint a)
{
  return a * a;
}

int square_uchar(const unsigned char a)
{
  return a * a;
}

__host__ __device__ float square_f(const float a)
{
  return a * a;
}

double square_d(const double a)
{
  return a * a;
}

// Cube.

int cube_s(const short a)
{
  return a * a * a;
}

int cube_i(const int a)
{
  return a * a * a;
}

uint cube_uint(const uint a)
{
  return a * a * a;
}

int cube_uchar(const unsigned char a)
{
  return a * a * a;
}

float cube_f(const float a)
{
  return a * a * a;
}

double cube_d(const double a)
{
  return a * a * a;
}

// Min/max

__host__ __device__ float min_ff(const float a, const float b)
{
  return (a < b) ? a : b;
}
__host__ __device__ float max_ff(const float a, const float b)
{
  return (a > b) ? a : b;
}
/* See: https://www.iquilezles.org/www/articles/smin/smin.htm. */
__host__ __device__ float smoothminf(const float a, const float b, const float c)
{
  if (c != 0.0f) {
    float h = max_ff(c - fabsf(a - b), 0.0f) / c;
    return min_ff(a, b) - h * h * h * c * (1.0f / 6.0f);
  }
  else {
    return min_ff(a, b);
  }
}

__host__ __device__ double min_dd(const double a, const double b)
{
  return (a < b) ? a : b;
}

__host__ __device__ double max_dd(const double a, const double b)
{
  return (a > b) ? a : b;
}

__host__ __device__ int min_ii(const int a, const int b)
{
  return (a < b) ? a : b;
}

__host__ __device__ int max_ii(const int a, const int b)
{
  return (b < a) ? a : b;
}

__host__ __device__ float min_fff(const float a, const float b, const float c)
{
  return min_ff(min_ff(a, b), c);
}
__host__ __device__ float max_fff(const float a, const float b, const float c)
{
  return max_ff(max_ff(a, b), c);
}

__host__ __device__ int min_iii(const int a, const int b, const int c)
{
  return min_ii(min_ii(a, b), c);
}
__host__ __device__ int max_iii(const int a, const int b, const int c)
{
  return max_ii(max_ii(a, b), c);
}

__host__ __device__ float min_ffff(const float a, const float b, const float c, const float d)
{
  return min_ff(min_fff(a, b, c), d);
}
__host__ __device__ float max_ffff(const float a, const float b, const float c, const float d)
{
  return max_ff(max_fff(a, b, c), d);
}

__host__ __device__ int min_iiii(const int a, const int b, const int c, const int d)
{
  return min_ii(min_iii(a, b, c), d);
}
__host__ __device__ int max_iiii(const int a, const int b, const int c, const int d)
{
  return max_ii(max_iii(a, b, c), d);
}

__host__ __device__ size_t min_zz(const size_t a, const size_t b)
{
  return (a < b) ? a : b;
}
__host__ __device__ size_t max_zz(const size_t a, const size_t b)
{
  return (b < a) ? a : b;
}

char min_cc(const char a, const char b)
{
  return (a < b) ? a : b;
}
char max_cc(const char a, const char b)
{
  return (b < a) ? a : b;
}

int clamp_i(const int value, const int min, const int max)
{
  return min_ii(max_ii(value, min), max);
}

float clamp_f(const float value, const float min, const float max)
{
  if (value > max) {
    return max;
  }
  else if (value < min) {
    return min;
  }
  return value;
}

size_t clamp_z(const size_t value, const size_t min, const size_t max)
{
  return min_zz(max_zz(value, min), max);
}

/**
 * Almost-equal for IEEE floats, using absolute difference method.
 *
 * \param max_diff: the maximum absolute difference.
 */
int compare_ff(const float a, const float b, const float max_diff)
{
  return fabsf(a - b) <= max_diff;
}

/**
 * Almost-equal for IEEE floats, using their integer representation
 * (mixing ULP and absolute difference methods).
 *
 * \param max_diff: is the maximum absolute difference (allows to take care of the near-zero area,
 * where relative difference methods cannot really work).
 * \param max_ulps: is the 'maximum number of floats + 1'
 * allowed between \a a and \a b to consider them equal.
 *
 * \see https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
 */
int compare_ff_relative(const float a, const float b, const float max_diff, const int max_ulps)
{
  union {
    float f;
    int i;
  } ua, ub;

  BLI_assert(sizeof(float) == sizeof(int));
  BLI_assert(max_ulps < (1 << 22));

  if (fabsf(a - b) <= max_diff) {
    return 1;
  }

  ua.f = a;
  ub.f = b;

  /* Important to compare sign from integers, since (-0.0f < 0) is false
   * (though this shall not be an issue in common cases)... */
  return ((ua.i < 0) != (ub.i < 0)) ? 0 : (abs(ua.i - ub.i) <= max_ulps) ? 1 : 0;
}

float signf(const float f)
{
  return (f < 0.0f) ? -1.0f : 1.0f;
}

float compatible_signf(const float f)
{
  if (f > 0.0f) {
    return 1.0f;
  }
  if (f < 0.0f) {
    return -1.0f;
  }
  else {
    return 0.0f;
  }
}

int signum_i_ex(const float a, const float eps)
{
  if (a > eps) {
    return 1;
  }
  if (a < -eps) {
    return -1;
  }
  else {
    return 0;
  }
}

int signum_i(const float a)
{
  if (a > 0.0f) {
    return 1;
  }
  if (a < 0.0f) {
    return -1;
  }
  else {
    return 0;
  }
}

/**
 * Returns number of (base ten) *significant* digits of integer part of given float
 * (negative in case of decimal-only floats, 0.01 returns -1 e.g.).
 */
int integer_digits_f(const float f)
{
  return (f == 0.0f) ? 0 : (int)floor(log10(fabs(f))) + 1;
}

/**
 * Returns number of (base ten) *significant* digits of integer part of given double
 * (negative in case of decimal-only floats, 0.01 returns -1 e.g.).
 */
int integer_digits_d(const double d)
{
  return (d == 0.0) ? 0 : (int)floor(log10(fabs(d))) + 1;
}

int integer_digits_i(const int i)
{
  return (int)log10((double)i) + 1;
}

/* Internal helpers for SSE2 implementation.
 *
 * NOTE: Are to be called ONLY from inside `#ifdef BLI_HAVE_SSE2` !!!
 */

#ifdef BLI_HAVE_SSE2

/* Calculate initial guess for arg^exp based on float representation
 * This method gives a constant bias, which can be easily compensated by
 * multiplying with bias_coeff.
 * Gives better results for exponents near 1 (e. g. 4/5).
 * exp = exponent, encoded as uint32_t
 * e2coeff = 2^(127/exponent - 127) * bias_coeff^(1/exponent), encoded as
 * uint32_t
 *
 * We hope that exp and e2coeff gets properly inlined
 */
MALWAYS_INLINE __m128 _bli_math_fastpow(const int exp, const int e2coeff, const __m128 arg)
{
  __m128 ret;
  ret = _mm_mul_ps(arg, _mm_castsi128_ps(_mm_set1_epi32(e2coeff)));
  ret = _mm_cvtepi32_ps(_mm_castps_si128(ret));
  ret = _mm_mul_ps(ret, _mm_castsi128_ps(_mm_set1_epi32(exp)));
  ret = _mm_castsi128_ps(_mm_cvtps_epi32(ret));
  return ret;
}

/* Improve x ^ 1.0f/5.0f solution with Newton-Raphson method */
MALWAYS_INLINE __m128 _bli_math_improve_5throot_solution(const __m128 old_result, const __m128 x)
{
  __m128 approx2 = _mm_mul_ps(old_result, old_result);
  __m128 approx4 = _mm_mul_ps(approx2, approx2);
  __m128 t = _mm_div_ps(x, approx4);
  __m128 summ = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(4.0f), old_result), t); /* FMA. */
  return _mm_mul_ps(summ, _mm_set1_ps(1.0f / 5.0f));
}

/* Calculate powf(x, 2.4). Working domain: 1e-10 < x < 1e+10 */
MALWAYS_INLINE __m128 _bli_math_fastpow24(const __m128 arg)
{
  /* max, avg and |avg| errors were calculated in gcc without FMA instructions
   * The final precision should be better than powf in glibc */

  /* Calculate x^4/5, coefficient 0.994 was constructed manually to minimize
   * avg error.
   */
  /* 0x3F4CCCCD = 4/5 */
  /* 0x4F55A7FB = 2^(127/(4/5) - 127) * 0.994^(1/(4/5)) */
  /* error max = 0.17, avg = 0.0018, |avg| = 0.05 */
  __m128 x = _bli_math_fastpow(0x3F4CCCCD, 0x4F55A7FB, arg);
  __m128 arg2 = _mm_mul_ps(arg, arg);
  __m128 arg4 = _mm_mul_ps(arg2, arg2);
  /* error max = 0.018        avg = 0.0031    |avg| = 0.0031  */
  x = _bli_math_improve_5throot_solution(x, arg4);
  /* error max = 0.00021    avg = 1.6e-05    |avg| = 1.6e-05 */
  x = _bli_math_improve_5throot_solution(x, arg4);
  /* error max = 6.1e-07    avg = 5.2e-08    |avg| = 1.1e-07 */
  x = _bli_math_improve_5throot_solution(x, arg4);
  return _mm_mul_ps(x, _mm_mul_ps(x, x));
}

/* Calculate powf(x, 1.0f / 2.4) */
MALWAYS_INLINE __m128 _bli_math_fastpow512(const __m128 arg)
{
  /* 5/12 is too small, so compute the 4th root of 20/12 instead.
   * 20/12 = 5/3 = 1 + 2/3 = 2 - 1/3. 2/3 is a suitable argument for fastpow.
   * weighting coefficient: a^-1/2 = 2 a; a = 2^-2/3
   */
  __m128 xf = _bli_math_fastpow(0x3f2aaaab, 0x5eb504f3, arg);
  __m128 xover = _mm_mul_ps(arg, xf);
  __m128 xfm1 = _mm_rsqrt_ps(xf);
  __m128 x2 = _mm_mul_ps(arg, arg);
  __m128 xunder = _mm_mul_ps(x2, xfm1);
  /* sqrt2 * over + 2 * sqrt2 * under */
  __m128 xavg = _mm_mul_ps(_mm_set1_ps(1.0f / (3.0f * 0.629960524947437f) * 0.999852f),
                           _mm_add_ps(xover, xunder));
  xavg = _mm_mul_ps(xavg, _mm_rsqrt_ps(xavg));
  xavg = _mm_mul_ps(xavg, _mm_rsqrt_ps(xavg));
  return xavg;
}

MALWAYS_INLINE __m128 _bli_math_blend_sse(const __m128 mask, const __m128 a, const __m128 b)
{
  return _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b));
}

#endif /* BLI_HAVE_SSE2 */

/* Low level conversion functions */
unsigned char unit_float_to_uchar_clamp(const float val)
{
  return (unsigned char)((
      (val <= 0.0f) ? 0 : ((val > (1.0f - 0.5f / 255.0f)) ? 255 : ((255.0f * val) + 0.5f))));
}
#define unit_float_to_uchar_clamp(val) \
  ((CHECK_TYPE_INLINE(val, float)), unit_float_to_uchar_clamp(val))

unsigned short unit_float_to_ushort_clamp(const float val)
{
  return (unsigned short)((val >= 1.0f - 0.5f / 65535) ?
                              65535 :
                              (val <= 0.0f) ? 0 : (val * 65535.0f + 0.5f));
}
#define unit_float_to_ushort_clamp(val) \
  ((CHECK_TYPE_INLINE(val, float)), unit_float_to_ushort_clamp(val))

unsigned char unit_ushort_to_uchar(const unsigned short val)
{
  return (unsigned char)(((val) >= 65535 - 128) ? 255 : ((val) + 128) >> 8);
}
#define unit_ushort_to_uchar(val) \
  ((CHECK_TYPE_INLINE(val, unsigned short)), unit_ushort_to_uchar(val))

#define unit_float_to_uchar_clamp_v3(v1, v2) \
  { \
    (v1)[0] = unit_float_to_uchar_clamp((v2[0])); \
    (v1)[1] = unit_float_to_uchar_clamp((v2[1])); \
    (v1)[2] = unit_float_to_uchar_clamp((v2[2])); \
  } \
  ((void)0)
#define unit_float_to_uchar_clamp_v4(v1, v2) \
  { \
    (v1)[0] = unit_float_to_uchar_clamp((v2[0])); \
    (v1)[1] = unit_float_to_uchar_clamp((v2[1])); \
    (v1)[2] = unit_float_to_uchar_clamp((v2[2])); \
    (v1)[3] = unit_float_to_uchar_clamp((v2[3])); \
  } \
  ((void)0)

#endif /* __MATH_BASE_INLINE_C__ */
