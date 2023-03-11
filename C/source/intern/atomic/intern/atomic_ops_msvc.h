#ifndef __ATOMIC_OPS_MSVC_H__
#define __ATOMIC_OPS_MSVC_H__

#define NOGDI
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN

#include <intrin.h>
#include <windows.h>

/* TODO(sergey): On x64 platform both read and write of a variable aligned to its type size is
 * atomic, so in theory it is possible to avoid memory barrier and gain performance. The downside
 * of that would be that it will impose requirement to value which is being operated on. */
#define __atomic_impl_load_generic(v) (MemoryBarrier(), *(v))
#define __atomic_impl_store_generic(p, v) \
  do { \
    *(p) = (v); \
    MemoryBarrier(); \
  } while (0)

/* 64-bit operations. */
/* Unsigned */
ATOMIC_INLINE uint64_t atomic_add_and_fetch_uint64(uint64_t *p, uint64_t x)
{
  return InterlockedExchangeAdd64((int64_t *)p, (int64_t)x) + x;
}

ATOMIC_INLINE uint64_t atomic_sub_and_fetch_uint64(uint64_t *p, uint64_t x)
{
  return InterlockedExchangeAdd64((int64_t *)p, -((int64_t)x)) - x;
}

ATOMIC_INLINE uint64_t atomic_cas_uint64(uint64_t *v, uint64_t old, uint64_t _new)
{
  return InterlockedCompareExchange64((int64_t *)v, _new, old);
}

ATOMIC_INLINE uint64_t atomic_load_uint64(const uint64_t *v)
{
  return __atomic_impl_load_generic(v);
}

ATOMIC_INLINE void atomic_store_uint64(uint64_t *p, uint64_t v)
{
  __atomic_impl_store_generic(p, v);
}

ATOMIC_INLINE uint64_t atomic_fetch_and_add_uint64(uint64_t *p, uint64_t x)
{
  return InterlockedExchangeAdd64((int64_t *)p, (int64_t)x);
}

ATOMIC_INLINE uint64_t atomic_fetch_and_sub_uint64(uint64_t *p, uint64_t x)
{
  return InterlockedExchangeAdd64((int64_t *)p, -((int64_t)x));
}

/* Signed */
ATOMIC_INLINE int64_t atomic_add_and_fetch_int64(int64_t *p, int64_t x)
{
  return InterlockedExchangeAdd64(p, x) + x;
}

ATOMIC_INLINE int64_t atomic_sub_and_fetch_int64(int64_t *p, int64_t x)
{
  return InterlockedExchangeAdd64(p, -x) - x;
}

ATOMIC_INLINE int64_t atomic_cas_int64(int64_t *v, int64_t old, int64_t _new)
{
  return InterlockedCompareExchange64(v, _new, old);
}

ATOMIC_INLINE int64_t atomic_load_int64(const int64_t *v)
{
  return __atomic_impl_load_generic(v);
}

ATOMIC_INLINE void atomic_store_int64(int64_t *p, int64_t v)
{
  __atomic_impl_store_generic(p, v);
}

ATOMIC_INLINE int64_t atomic_fetch_and_add_int64(int64_t *p, int64_t x)
{
  return InterlockedExchangeAdd64(p, x);
}

ATOMIC_INLINE int64_t atomic_fetch_and_sub_int64(int64_t *p, int64_t x)
{
  return InterlockedExchangeAdd64(p, -x);
}

/******************************************************************************/
/* 32-bit operations. */
/* Unsigned */
ATOMIC_INLINE uint32_t atomic_add_and_fetch_uint32(uint32_t *p, uint32_t x)
{
  return InterlockedExchangeAdd(p, x) + x;
}

ATOMIC_INLINE uint32_t atomic_sub_and_fetch_uint32(uint32_t *p, uint32_t x)
{
  return InterlockedExchangeAdd(p, -((int32_t)x)) - x;
}

ATOMIC_INLINE uint32_t atomic_cas_uint32(uint32_t *v, uint32_t old, uint32_t _new)
{
  return InterlockedCompareExchange((long *)v, _new, old);
}

ATOMIC_INLINE uint32_t atomic_load_uint32(const uint32_t *v)
{
  return __atomic_impl_load_generic(v);
}

ATOMIC_INLINE void atomic_store_uint32(uint32_t *p, uint32_t v)
{
  __atomic_impl_store_generic(p, v);
}

ATOMIC_INLINE uint32_t atomic_fetch_and_add_uint32(uint32_t *p, uint32_t x)
{
  return InterlockedExchangeAdd(p, x);
}

ATOMIC_INLINE uint32_t atomic_fetch_and_or_uint32(uint32_t *p, uint32_t x)
{
  return InterlockedOr((long *)p, x);
}

ATOMIC_INLINE uint32_t atomic_fetch_and_and_uint32(uint32_t *p, uint32_t x)
{
  return InterlockedAnd((long *)p, x);
}

/* Signed */
ATOMIC_INLINE int32_t atomic_add_and_fetch_int32(int32_t *p, int32_t x)
{
  return InterlockedExchangeAdd((long *)p, x) + x;
}

ATOMIC_INLINE int32_t atomic_sub_and_fetch_int32(int32_t *p, int32_t x)
{
  return InterlockedExchangeAdd((long *)p, -x) - x;
}

ATOMIC_INLINE int32_t atomic_cas_int32(int32_t *v, int32_t old, int32_t _new)
{
  return InterlockedCompareExchange((long *)v, _new, old);
}

ATOMIC_INLINE int32_t atomic_load_int32(const int32_t *v)
{
  return __atomic_impl_load_generic(v);
}

ATOMIC_INLINE void atomic_store_int32(int32_t *p, int32_t v)
{
  __atomic_impl_store_generic(p, v);
}

ATOMIC_INLINE int32_t atomic_fetch_and_add_int32(int32_t *p, int32_t x)
{
  return InterlockedExchangeAdd((long *)p, x);
}

ATOMIC_INLINE int32_t atomic_fetch_and_or_int32(int32_t *p, int32_t x)
{
  return InterlockedOr((long *)p, x);
}

ATOMIC_INLINE int32_t atomic_fetch_and_and_int32(int32_t *p, int32_t x)
{
  return InterlockedAnd((long *)p, x);
}

/******************************************************************************/
/* 16-bit operations. */

/* Signed */
ATOMIC_INLINE int16_t atomic_fetch_and_or_int16(int16_t *p, int16_t x)
{
  return InterlockedOr16((short *)p, x);
}

ATOMIC_INLINE int16_t atomic_fetch_and_and_int16(int16_t *p, int16_t x)
{
  return InterlockedAnd16((short *)p, x);
}

/******************************************************************************/
/* 8-bit operations. */

/* Unsigned */
#pragma intrinsic(_InterlockedAnd8)
ATOMIC_INLINE uint8_t atomic_fetch_and_and_uint8(uint8_t *p, uint8_t b)
{
#if (LG_SIZEOF_PTR == 8 || LG_SIZEOF_INT == 8)
  return InterlockedAnd8((char *)p, (char)b);
#else
  return _InterlockedAnd8((char *)p, (char)b);
#endif
}

#pragma intrinsic(_InterlockedOr8)
ATOMIC_INLINE uint8_t atomic_fetch_and_or_uint8(uint8_t *p, uint8_t b)
{
#if (LG_SIZEOF_PTR == 8 || LG_SIZEOF_INT == 8)
  return InterlockedOr8((char *)p, (char)b);
#else
  return _InterlockedOr8((char *)p, (char)b);
#endif
}

/* Signed */
#pragma intrinsic(_InterlockedAnd8)
ATOMIC_INLINE int8_t atomic_fetch_and_and_int8(int8_t *p, int8_t b)
{
#if (LG_SIZEOF_PTR == 8 || LG_SIZEOF_INT == 8)
  return InterlockedAnd8((char *)p, (char)b);
#else
  return _InterlockedAnd8((char *)p, (char)b);
#endif
}

#pragma intrinsic(_InterlockedOr8)
ATOMIC_INLINE int8_t atomic_fetch_and_or_int8(int8_t *p, int8_t b)
{
#if (LG_SIZEOF_PTR == 8 || LG_SIZEOF_INT == 8)
  return InterlockedOr8((char *)p, (char)b);
#else
  return _InterlockedOr8((char *)p, (char)b);
#endif
}

#undef __atomic_impl_load_generic
#undef __atomic_impl_store_generic

#if defined(__clang__)
#  pragma GCC diagnostic pop
#endif

#endif /* __ATOMIC_OPS_MSVC_H__ */
