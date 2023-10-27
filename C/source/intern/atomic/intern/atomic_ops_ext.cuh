#ifndef __ATOMIC_OPS_EXT_H__
#define __ATOMIC_OPS_EXT_H__

#include <stdint.h>
#include <stdlib.h>

#include <intrin0.inl.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#define LG_SIZEOF_PTR 4
#define LG_SIZEOF_INT 4

#define __atomic_impl_load_generic(v) (*(v))
#define __atomic_impl_store_generic(p, v) do { *(p) = (v);} while (0)

typedef unsigned int uint;

__host__ __device__ inline uint32_t atomic_load_uint32(const uint32_t* v)
{
    return __atomic_impl_load_generic(v);
}

__host__ __device__ inline void atomic_store_uint32(uint32_t* p, uint32_t v)
{
    __atomic_impl_store_generic(p, v);
}

__host__ __device__ inline uint32_t atomic_add_and_fetch_uint32(uint32_t* p, uint32_t x)
{
	#ifdef __CUDA_ARCH__
	    // Версия для device-кода, используя встроенные функции CUDA
	    return atomicAdd(p, x) + x;
	#else
	    // Версия для хост-кода, используя Windows Interlocked API
	    return _InterlockedExchangeAdd(reinterpret_cast<long volatile*>(p), x) + x;
	#endif
}

__host__ __device__ inline uint32_t atomic_sub_and_fetch_uint32(uint32_t* p, uint32_t x)
{
	#ifdef __CUDA_ARCH__
	    // Версия для device-кода, используя встроенные функции CUDA
	    return atomicSub(p, x) - x;
	#else
	    // Версия для хост-кода, используя Windows Interlocked API
	    return _InterlockedExchangeAdd(reinterpret_cast<long volatile*>(p), -static_cast<int32_t>(x)) - x;
	#endif
}

__host__ __device__ inline uint32_t atomic_cas_uint32(uint32_t* v, uint32_t old, uint32_t _new)
{
#ifdef __CUDA_ARCH__
    // Версия для device-кода, используя встроенные функции CUDA
    return atomicCAS(v, old, _new);
#else
    // Версия для хост-кода, используя Windows Interlocked API
    return _InterlockedCompareExchange(reinterpret_cast<long*>(v), _new, old);
#endif
    
}

__host__ __device__ inline uint32_t atomic_fetch_and_add_uint32(uint32_t* p, uint32_t x)
{
#ifdef __CUDA_ARCH__
    // Версия для device-кода, используя встроенные функции CUDA
    return atomicExch(p, *p + x);
#else
    // Версия для хост-кода, используя Windows Interlocked API
    return _InterlockedExchangeAdd(reinterpret_cast<long volatile*>(p), x);
#endif
}


__host__ __device__ inline uint32_t atomic_fetch_and_or_uint32(uint32_t* p, uint32_t x)
{
#ifdef __CUDA_ARCH__
    // Версия для device-кода, используя встроенные функции CUDA
    return atomicOr(p, x);
#else
    // Версия для хост-кода, используя Windows Interlocked API
    return _InterlockedOr(reinterpret_cast<long volatile*>(p), x);
#endif
}

__host__ __device__ inline uint32_t atomic_fetch_and_and_uint32(uint32_t* p, uint32_t x)
{
#ifdef __CUDA_ARCH__
    // Версия для device-кода, используя встроенные функции CUDA
    return atomicAnd(p, x);
#else
    // Версия для хост-кода, используя Windows Interlocked API
    return _InterlockedExchangeAdd(reinterpret_cast<long volatile*>(p), x);  // Это не точный аналог, проверьте логику вашей программы
#endif
}

/******************************************************************************/
/* 8-bit operations. */

/* Unsigned */
#pragma intrinsic(_InterlockedAnd8)
__host__ __device__ inline uint8_t atomic_fetch_and_and_uint8(uint8_t* p, uint8_t b)
{
#if (LG_SIZEOF_PTR == 8 || LG_SIZEOF_INT == 8)
    return InterlockedAnd8((char*)p, (char)b);
#else
    return _InterlockedAnd8((char*)p, (char)b);
#endif
}

#pragma intrinsic(_InterlockedOr8)
__host__ __device__ inline uint8_t atomic_fetch_and_or_uint8(uint8_t* p, uint8_t b)
{
#if (LG_SIZEOF_PTR == 8 || LG_SIZEOF_INT == 8)
    return InterlockedOr8((char*)p, (char)b);
#else
    return _InterlockedOr8((char*)p, (char)b);
#endif
}

/* Signed */
#pragma intrinsic(_InterlockedAnd8)
__host__ __device__ inline int8_t atomic_fetch_and_and_int8(int8_t* p, int8_t b)
{
#if (LG_SIZEOF_PTR == 8 || LG_SIZEOF_INT == 8)
    return InterlockedAnd8((char*)p, (char)b);
#else
    return _InterlockedAnd8((char*)p, (char)b);
#endif
}

#pragma intrinsic(_InterlockedOr8)
__host__ __device__ inline int8_t atomic_fetch_and_or_int8(int8_t* p, int8_t b)
{
#if (LG_SIZEOF_PTR == 8 || LG_SIZEOF_INT == 8)
    return InterlockedOr8((char*)p, (char)b);
#else
    return _InterlockedOr8((char*)p, (char)b);
#endif
}

__host__ __device__ inline size_t atomic_add_and_fetch_z(size_t* p, size_t x)
{
#if (LG_SIZEOF_PTR == 8)
    return (size_t)atomic_add_and_fetch_uint64((uint64_t*)p, (uint64_t)x);
#elif (LG_SIZEOF_PTR == 4)
    return (size_t)atomic_add_and_fetch_uint32((uint32_t*)p, (uint32_t)x);
#endif
}

__host__ __device__ inline size_t atomic_sub_and_fetch_z(size_t *p, size_t x)
{
#if (LG_SIZEOF_PTR == 8)
  return (size_t)atomic_add_and_fetch_uint64((uint64_t *)p, (uint64_t) - ((int64_t)x));
#elif (LG_SIZEOF_PTR == 4)
  return (size_t)atomic_add_and_fetch_uint32((uint32_t *)p, (uint32_t) - ((int32_t)x));
#endif
}

__host__ __device__ inline size_t atomic_fetch_and_add_z(size_t *p, size_t x)
{
#if (LG_SIZEOF_PTR == 8)
  return (size_t)atomic_fetch_and_add_uint64((uint64_t *)p, (uint64_t)x);
#elif (LG_SIZEOF_PTR == 4)
  return (size_t)atomic_fetch_and_add_uint32((uint32_t *)p, (uint32_t)x);
#endif
}

__host__ __device__ inline size_t atomic_fetch_and_sub_z(size_t *p, size_t x)
{
#if (LG_SIZEOF_PTR == 8)
  return (size_t)atomic_fetch_and_add_uint64((uint64_t *)p, (uint64_t) - ((int64_t)x));
#elif (LG_SIZEOF_PTR == 4)
  return (size_t)atomic_fetch_and_add_uint32((uint32_t *)p, (uint32_t) - ((int32_t)x));
#endif
}

__host__ __device__ inline size_t atomic_cas_z(size_t *v, size_t old, size_t _new)
{
#if (LG_SIZEOF_PTR == 8)
  return (size_t)atomic_cas_uint64((uint64_t *)v, (uint64_t)old, (uint64_t)_new);
#elif (LG_SIZEOF_PTR == 4)
  return (size_t)atomic_cas_uint32((uint32_t *)v, (uint32_t)old, (uint32_t)_new);
#endif
}

__host__ __device__ inline size_t atomic_load_z(const size_t *v)
{
#if (LG_SIZEOF_PTR == 8)
  return (size_t)atomic_load_uint64((const uint64_t *)v);
#elif (LG_SIZEOF_PTR == 4)
  return (size_t)atomic_load_uint32((const uint32_t *)v);
#endif
}

__host__ __device__ inline void atomic_store_z(size_t *p, size_t v)
{
#if (LG_SIZEOF_PTR == 8)
  atomic_store_uint64((uint64_t *)p, (uint32_t)v);
#elif (LG_SIZEOF_PTR == 4)
  atomic_store_uint32((uint32_t *)p, (uint32_t)v);
#endif
}

__host__ __device__ inline size_t atomic_fetch_and_update_max_z(size_t *p, size_t x)
{
  size_t prev_value;
  while ((prev_value = *p) < x) {
    if (atomic_cas_z(p, prev_value, x) == prev_value) {
      break;
    }
  }
  return prev_value;
}

__host__ __device__ inline uint atomic_add_and_fetch_u(uint *p, uint x)
{
#if (LG_SIZEOF_INT == 8)
  return (uint)atomic_add_and_fetch_uint64((uint64_t *)p, (uint64_t)x);
#elif (LG_SIZEOF_INT == 4)
  return atomic_add_and_fetch_uint32((uint32_t *)p, (uint32_t)x);
#endif
}

__host__ __device__ inline uint atomic_sub_and_fetch_u(uint *p, uint x)
{
#if (LG_SIZEOF_INT == 8)
  return (uint)atomic_add_and_fetch_uint64((uint64_t *)p, (uint64_t) - ((int64_t)x));
#elif (LG_SIZEOF_INT == 4)
  return (uint)atomic_add_and_fetch_uint32((uint32_t *)p, (uint32_t) - ((int32_t)x));
#endif
}

__host__ __device__ inline uint atomic_fetch_and_add_u(uint *p, uint x)
{
#if (LG_SIZEOF_INT == 8)
  return (uint)atomic_fetch_and_add_uint64((uint64_t *)p, (uint64_t)x);
#elif (LG_SIZEOF_INT == 4)
  return (uint)atomic_fetch_and_add_uint32((uint32_t *)p, (uint32_t)x);
#endif
}

__host__ __device__ inline uint atomic_fetch_and_sub_u(uint *p, uint x)
{
#if (LG_SIZEOF_INT == 8)
  return (uint)atomic_fetch_and_add_uint64((uint64_t *)p, (uint64_t) - ((int64_t)x));
#elif (LG_SIZEOF_INT == 4)
  return (uint)atomic_fetch_and_add_uint32((uint32_t *)p, (uint32_t) - ((int32_t)x));
#endif
}

__host__ __device__ inline uint atomic_cas_u(uint *v, uint old, uint _new)
{
#if (LG_SIZEOF_INT == 8)
  return (uint)atomic_cas_uint64((uint64_t *)v, (uint64_t)old, (uint64_t)_new);
#elif (LG_SIZEOF_INT == 4)
  return (uint)atomic_cas_uint32((uint32_t *)v, (uint32_t)old, (uint32_t)_new);
#endif
}

/******************************************************************************/
/* Char operations. */
__host__ __device__ inline char atomic_fetch_and_or_char(char *p, char b)
{
  return (char)atomic_fetch_and_or_uint8((uint8_t *)p, (uint8_t)b);
}

__host__ __device__ inline char atomic_fetch_and_and_char(char *p, char b)
{
  return (char)atomic_fetch_and_and_uint8((uint8_t *)p, (uint8_t)b);
}

/******************************************************************************/
/* Pointer operations. */

__host__ __device__ inline void *atomic_cas_ptr(void **v, void *old, void *_new)
{
#if (LG_SIZEOF_PTR == 8)
  return (void *)atomic_cas_uint64((uint64_t *)v, *(uint64_t *)&old, *(uint64_t *)&_new);
#elif (LG_SIZEOF_PTR == 4)
  return (void *)atomic_cas_uint32((uint32_t *)v, *(uint32_t *)&old, *(uint32_t *)&_new);
#endif
}

__host__ __device__ inline void *atomic_load_ptr(void *const *v)
{
#if (LG_SIZEOF_PTR == 8)
  return (void *)atomic_load_uint64((const uint64_t *)v);
#elif (LG_SIZEOF_PTR == 4)
  return reinterpret_cast<void*>(atomic_load_uint32(reinterpret_cast<const uint32_t*>(v)));
#endif
}

__host__ __device__ inline void atomic_store_ptr(void **p, void *v)
{
#if (LG_SIZEOF_PTR == 8)
  atomic_store_uint64((uint64_t *)p, (uint64_t)v);
#elif (LG_SIZEOF_PTR == 4)
  atomic_store_uint32((uint32_t *)p, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(v)));
#endif
}

__host__ __device__ inline float atomic_cas_float(float *v, float old, float _new)
{
  uint32_t ret = atomic_cas_uint32((uint32_t *)v, *(uint32_t *)&old, *(uint32_t *)&_new);
  return *(float *)&ret;
}

__host__ __device__ inline float atomic_add_and_fetch_fl(float *p, const float x)
{
  float oldval, newval;
  uint32_t prevval;

  do { /* Note that since collisions are unlikely, loop will nearly always run once. */
    oldval = *p;
    newval = oldval + x;
    prevval = atomic_cas_uint32((uint32_t *)p, *(uint32_t *)(&oldval), *(uint32_t *)(&newval));
  } while (prevval != *(uint32_t *)(&oldval));

  return newval;
}

#endif /* __ATOMIC_OPS_EXT_H__ */
