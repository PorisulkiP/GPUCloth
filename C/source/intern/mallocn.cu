#include "MEM_guardedalloc.cuh"
#include "mallocn_intern.cuh"
#include <cassert>

__host__ __device__ void *aligned_malloc(size_t size, size_t alignment)
{
#ifdef __CUDA_ARCH__
	// Версия для device-кода, используя встроенные функции CUDA
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size + alignment);
    if (err != cudaSuccess) {
        return nullptr;
    }
    size_t adjusted = ((size_t)ptr + alignment - 1) & ~(alignment - 1);
    return (void*)adjusted;
#else
    /* posix_memalign requires alignment to be a multiple of sizeof(void *). */
    assert(alignment >= ALIGNED_MALLOC_MINIMUM_ALIGNMENT);

	#ifdef _WIN32
	    return _aligned_malloc(size, alignment);
	#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__APPLE__)
	    void* result;

	    if (posix_memalign(&result, alignment, size)) {
	        /* non-zero means allocation error
	         * either no allocation or bad alignment value
	         */
	        return NULL;
	    }
	    return result;
	#else /* This is for Linux. */
	    return memalign(alignment, size);
	#endif
#endif
}

__host__ __device__ void aligned_free(void *ptr)
{
#ifdef __CUDA_ARCH__
	cudaFree(ptr);
#else
	#ifdef _WIN32
	  _aligned_free(ptr);
	#else
	  free(ptr);
	#endif
#endif
}

/* Perform assert checks on allocator type change.
 *
 * Helps catching issues (in debug build) caused by an unintended allocator type change when there
 * are allocation happened. */
//__host__ __device__ static void assert_for_allocator_change(void)
//{
//  /* NOTE: Assume that there is no "sticky" internal state which would make switching allocator
//   * type after all allocations are freed unsafe. In fact, it should be safe to change allocator
//   * type after all blocks has been freed: some regression tests do rely on this property of
//   * allocators. */
//  assert(MEM_lockfree_get_memory_blocks_in_use() == 0);
//}
//
//__host__ __device__ void MEM_use_lockfree_allocator(void)
//{
//  /* NOTE: Keep in sync with static initialization of the variables. */
//
//  /* TODO(sergey): Find a way to de-duplicate the logic. Maybe by requiring an explicit call
//   * to guarded allocator initialization at an application startup. */
//
//  assert_for_allocator_change();
//
//  //MEM_lockfree_allocN_len = MEM_lockfree_allocN_len;
//  MEM_lockfree_freeN = MEM_lockfree_freeN;
//  MEM_lockfree_dupallocN = MEM_lockfree_dupallocN;
//  MEM_lockfree_reallocN_id = MEM_lockfree_reallocN_id;
//  MEM_lockfree_recallocN_id = MEM_lockfree_recallocN_id;
//  MEM_lockfree_callocN = MEM_lockfree_callocN;
//  MEM_lockfree_calloc_arrayN = MEM_lockfree_calloc_arrayN;
//  MEM_lockfree_mallocN = MEM_lockfree_mallocN;
//  MEM_lockfree_malloc_arrayN = MEM_lockfree_malloc_arrayN;
//  MEM_lockfree_mallocN_aligned = MEM_lockfree_mallocN_aligned;
//  MEM_lockfree_printmemlist_pydict = MEM_lockfree_printmemlist_pydict;
//  MEM_lockfree_printmemlist = MEM_lockfree_printmemlist;
//  MEM_lockfree_callbackmemlist = MEM_lockfree_callbackmemlist;
//  MEM_lockfree_printmemlist_stats = MEM_lockfree_printmemlist_stats;
//  MEM_lockfree_set_error_callback = MEM_lockfree_set_error_callback;
//  MEM_lockfree_consistency_check = MEM_lockfree_consistency_check;
//  MEM_lockfree_set_memory_debug = MEM_lockfree_set_memory_debug;
//  MEM_lockfree_get_memory_in_use = MEM_lockfree_get_memory_in_use;
//  MEM_lockfree_get_memory_blocks_in_use = MEM_lockfree_get_memory_blocks_in_use;
//  MEM_lockfree_reset_peak_memory = MEM_lockfree_reset_peak_memory;
//  MEM_lockfree_get_peak_memory = MEM_lockfree_get_peak_memory;
//
//#ifndef NDEBUG
//  MEM_lockfree_name_ptr = MEM_lockfree_name_ptr;
//#endif
//}
//
//__host__ __device__ void MEM_use_guarded_allocator(void)
//{
//  assert_for_allocator_change();
//
//  //MEM_lockfree_allocN_len = MEM_guarded_allocN_len;
//  MEM_lockfree_freeN = MEM_guarded_freeN;
//  MEM_lockfree_dupallocN = MEM_guarded_dupallocN;
//  MEM_lockfree_reallocN_id = MEM_guarded_reallocN_id;
//  MEM_lockfree_recallocN_id = MEM_guarded_recallocN_id;
//  MEM_lockfree_callocN = MEM_guarded_callocN;
//  MEM_lockfree_calloc_arrayN = MEM_guarded_calloc_arrayN;
//  MEM_lockfree_mallocN = MEM_guarded_mallocN;
//  MEM_lockfree_malloc_arrayN = MEM_guarded_malloc_arrayN;
//  MEM_lockfree_mallocN_aligned = MEM_guarded_mallocN_aligned;
//  MEM_lockfree_printmemlist_pydict = MEM_guarded_printmemlist_pydict;
//  MEM_lockfree_printmemlist = MEM_guarded_printmemlist;
//  MEM_lockfree_callbackmemlist = MEM_guarded_callbackmemlist;
//  MEM_lockfree_printmemlist_stats = MEM_guarded_printmemlist_stats;
//  MEM_lockfree_set_error_callback = MEM_guarded_set_error_callback;
//  MEM_lockfree_consistency_check = MEM_guarded_consistency_check;
//  MEM_lockfree_set_memory_debug = MEM_guarded_set_memory_debug;
//  MEM_lockfree_get_memory_in_use = MEM_guarded_get_memory_in_use;
//  MEM_lockfree_get_memory_blocks_in_use = MEM_guarded_get_memory_blocks_in_use;
//  MEM_lockfree_reset_peak_memory = MEM_guarded_reset_peak_memory;
//  MEM_lockfree_get_peak_memory = MEM_guarded_get_peak_memory;
//
//#ifndef NDEBUG
//  MEM_lockfree_name_ptr = MEM_guarded_name_ptr;
//#endif
//}
