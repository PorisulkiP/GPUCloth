#pragma once

#include "cuda_runtime_api.h"

#ifndef __MALLOCN_INTERN_H__
#define __MALLOCN_INTERN_H__

#ifdef __GNUC__
#  define UNUSED(x) UNUSED_##x __attribute__((__unused__))
#else
#  define UNUSED(x) UNUSED_##x
#endif

#undef HAVE_MALLOC_STATS
#define USE_MALLOC_USABLE_SIZE /* internal, when we have malloc_usable_size() */

#if defined(__linux__) || (defined(__FreeBSD_kernel__) && !defined(__FreeBSD__)) || \
    defined(__GLIBC__)
#  include <malloc.h>
#  define HAVE_MALLOC_STATS
#elif defined(__FreeBSD__)
#  include <malloc_np.h>
#elif defined(__APPLE__)
#  include <malloc/malloc.h>
#  define malloc_usable_size malloc_size
#elif defined(WIN32) || defined(WIN64)
#  include <malloc.h>
#  define malloc_usable_size _msize
#elif defined(__HAIKU__)
#  include <malloc.h>
size_t malloc_usable_size(void *ptr);
#else
#  pragma message ("We don't know how to use malloc_usable_size on your platform")
#  undef USE_MALLOC_USABLE_SIZE
#endif

#define SIZET_FORMAT "%zu"
#define SIZET_ARG(a) ((size_t)(a))

#define SIZET_ALIGN_4(len) ((len + 3) & ~(size_t)3)

#ifdef __GNUC__
#  define LIKELY(x) __builtin_expect(!!(x), 1)
#  define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#  define LIKELY(x) (x)
#  define UNLIKELY(x) (x)
#endif

#if !defined(__APPLE__) && !defined(__FreeBSD__) && !defined(__NetBSD__)
// Needed for memalign on Linux and _aligned_alloc on Windows.

#  include <malloc.h>
#else
// Apple's malloc is 16-byte aligned, and does not have malloc.h, so include
// stdilb instead.
#  include <stdlib.h>
#endif

/* visual studio 2012 does not define inline for C */
#ifdef _MSC_VER
#  define MEM_INLINE static __inline
#else
#  define MEM_INLINE static inline
#endif

#define IS_POW2(a) (((a) & ((a)-1)) == 0)

/* Extra padding which needs to be applied on MemHead to make it aligned. */
#define MEMHEAD_ALIGN_PADDING(alignment)  ((size_t)alignment - (sizeof(MemHeadAligned) % (size_t)alignment))

/* Real pointer returned by the malloc or aligned_alloc. */
#define MEMHEAD_REAL_PTR(memh) ((char *)memh - MEMHEAD_ALIGN_PADDING(memh->alignment))

#define ALIGNED_MALLOC_MINIMUM_ALIGNMENT sizeof(void *)

__host__ __device__ void *aligned_malloc(size_t size, size_t alignment);
__host__ __device__ void aligned_free(void *ptr);

extern bool leak_detector_has_run;
extern char free_after_leak_detection_message[];

extern __device__ bool d_leak_detector_has_run;
extern __device__ char d_free_after_leak_detection_message[];

/* Prototypes for counted allocator functions */
__host__ __device__ size_t MEM_lockfree_allocN_len(const void *vmemh);
__host__ __device__ void MEM_lockfree_freeN(void *vmemh);
__host__ __device__ void *MEM_lockfree_dupallocN(const void *vmemh);
__host__ __device__ void *MEM_lockfree_reallocN_id(void *vmemh, size_t len, const char *str);
__host__ __device__ void *MEM_lockfree_recallocN_id(void *vmemh, size_t len, const char *str);
__host__ __device__ void *MEM_lockfree_callocN(size_t len, const char *str);
__host__ __device__ void *MEM_lockfree_calloc_arrayN(size_t len, size_t size, const char *str);
__host__ __device__ void *MEM_lockfree_mallocN(size_t len, const char *str);
__host__ __device__ void *MEM_lockfree_malloc_arrayN(size_t len, size_t size, const char *str);
__host__ __device__ void *MEM_lockfree_mallocN_aligned(size_t len, size_t alignment, const char *str);
__host__ __device__ void MEM_lockfree_printmemlist_pydict(void);
__host__ __device__ void MEM_lockfree_printmemlist(void);
__host__ __device__ void MEM_lockfree_callbackmemlist(void (*func)(void *));
__host__ __device__ void MEM_lockfree_printmemlist_stats(void);
__host__ __device__ void MEM_lockfree_set_error_callback(void (*func)(const char *));
__host__ __device__ bool MEM_lockfree_consistency_check(void);
__host__ __device__ void MEM_lockfree_set_memory_debug(void);
__host__ __device__ size_t MEM_lockfree_get_memory_in_use(void);
__host__ __device__ unsigned int MEM_lockfree_get_memory_blocks_in_use(void);
__host__ __device__ void MEM_lockfree_reset_peak_memory(void);
__host__ __device__ size_t MEM_lockfree_get_peak_memory(void);

#ifndef NDEBUG
__host__ __device__ const char *MEM_lockfree_name_ptr(void *vmemh);
#endif

/* Prototypes for fully guarded allocator functions */
__host__ __device__ size_t MEM_guarded_allocN_len(const void *vmemh);
__host__ __device__ void MEM_guarded_freeN(void *vmemh);
__host__ __device__ void *MEM_guarded_dupallocN(const void *vmemh);
__host__ __device__ void *MEM_guarded_reallocN_id(void *vmemh,
                              size_t len,
                              const char *str);
__host__ __device__ void *MEM_guarded_recallocN_id(void *vmemh,
                               size_t len,
                               const char *str);
__host__ __device__ void *MEM_guarded_callocN(size_t len, const char *str);
__host__ __device__ void *MEM_guarded_calloc_arrayN(size_t len,
                                size_t size,
                                const char *str);
__host__ __device__ void *MEM_guarded_mallocN(size_t len, const char *str);
__host__ __device__ void *MEM_guarded_malloc_arrayN(size_t len,
                                size_t size,
                                const char *str);
__host__ __device__ void *MEM_guarded_mallocN_aligned(size_t len,
                                  size_t alignment,
                                  const char *str);
__host__ __device__ void MEM_guarded_printmemlist_pydict(void);
__host__ __device__ void MEM_guarded_printmemlist(void);
__host__ __device__ void MEM_guarded_callbackmemlist(void (*func)(void *));
__host__ __device__ void MEM_guarded_printmemlist_stats(void);
__host__ __device__ void MEM_guarded_set_error_callback(void (*func)(const char *));
__host__ __device__ bool MEM_guarded_consistency_check(void);
__host__ __device__ void MEM_guarded_set_memory_debug(void);
__host__ __device__ size_t MEM_guarded_get_memory_in_use(void);
__host__ __device__ unsigned int MEM_guarded_get_memory_blocks_in_use(void);
__host__ __device__ void MEM_guarded_reset_peak_memory(void);
__host__ __device__ size_t MEM_guarded_get_peak_memory(void);
#ifndef NDEBUG
__host__ __device__ const char *MEM_guarded_name_ptr(void *vmemh);
#endif


#define BLI_array_alloca(arr, realsize) alloca(sizeof(*(arr)) * (realsize))
#define MEM_reallocN(vmemh, len) MEM_lockfree_reallocN_id(vmemh, len, __func__)
#define MEM_recallocN(vmemh, len) MEM_lockfree_recallocN_id(vmemh, len, __func__)


#include <new>
#include <type_traits>
#include <utility>

/**
 * Allocate new memory for and constructs an object of type #T.
 * #MEM_delete should be used to delete the object. Just calling #MEM_lockfree_freeN is not enough when #T
 * is not a trivial type.
 *
 * Note that when no arguments are passed, C++ will do recursive member-wise value initialization.
 * That is because C++ differentiates between creating an object with `T` (default initialization)
 * and `T()` (value initialization), whereby this function does the latter. Value initialization
 * rules are complex, but for C-style structs, memory will be zero-initialized. So this doesn't
 * match a `malloc()`, but a `calloc()` call in this case. See https://stackoverflow.com/a/4982720.
 */
template<typename T, typename... Args>
__host__ __device__  T* MEM_new(const char* allocation_name, Args &&...args)
{
    void* buffer = MEM_lockfree_mallocN(sizeof(T), allocation_name);
    return new (buffer) T(std::forward<Args>(args)...);
}

/**
 * Allocates zero-initialized memory for an object of type #T. The constructor of #T is not called,
 * therefor this should only used with trivial types (like all C types).
 * It's valid to call #MEM_lockfree_freeN on a pointer returned by this, because a destructor call is not
 * necessary, because the type is trivial.
 */
template<typename T>
__host__ __device__ T* MEM_cnew(const char* allocation_name)
{
    static_assert(std::is_trivial_v<T>, "For non-trivial types, MEM_new should be used.");
    return static_cast<T*>(MEM_lockfree_callocN(sizeof(T), allocation_name));
}

/* Allocation functions (for C++ only). */
#  define MEM_CXX_CLASS_ALLOC_FUNCS(_id) \
   public: \
    __host__ __device__ void *operator new(size_t num_bytes) \
    { \
      return MEM_lockfree_mallocN(num_bytes, _id); \
    } \
    __host__ __device__ void operator delete(void *mem) \
    { \
      if (mem) { \
        MEM_lockfree_freeN(mem); \
      } \
    } \
    __host__ __device__ void *operator new[](size_t num_bytes) \
    { \
      return MEM_lockfree_mallocN(num_bytes, _id "[]"); \
    } \
    __host__ __device__ void operator delete[](void *mem) \
    { \
      if (mem) { \
        MEM_lockfree_freeN(mem); \
      } \
    } \
    __host__ __device__ void *operator new(size_t /*count*/, void *ptr) \
    { \
      return ptr; \
    } \
    /* This is the matching delete operator to the placement-new operator above. Both parameters \
     * will have the same value. Without this, we get the warning C4291 on windows. */ \
    __host__ __device__ void operator delete(void * /*ptr_to_free*/, void * /*ptr*/) \
    { \
    }

template<typename T>
__host__ __device__ void MEM_SAFE_FREE(T*& ptr)
{
    if (ptr)
    {
        MEM_lockfree_freeN((void*)(ptr));
        ptr = nullptr;
    }
}

#endif /* __MALLOCN_INTERN_H__ */

