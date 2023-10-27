
#include "mallocn_intern.cuh"

#include <cstdarg>
#include <cstdio> /* printf */
#include <cstdlib>
#include <cstring> /* memcpy */
#include <cuda_runtime_api.h>

#include "MEM_guardedalloc.cuh"

#include "atomic_ops.cuh"
#include "mallocn_inline.cuh"

typedef struct MemHead {
  /* Length of allocated memory block. */
  size_t len;
} MemHead;

typedef struct MemHeadAligned {
  short alignment;
  size_t len;
} MemHeadAligned;

__device__ static uint d_totblock = 0;
__device__ static size_t d_mem_in_use = 0, d_peak_mem = 0;
__device__ static bool d_malloc_debug_memset = false;

__device__ static void (*d_error_callback)(const char*) = nullptr;

static uint totblock = 0;
static size_t mem_in_use = 0, peak_mem = 0;
static bool malloc_debug_memset = false;

static void (*error_callback)(const char *) = nullptr;

enum {
  MEMHEAD_ALIGN_FLAG = 1,
};

#define MEMHEAD_FROM_PTR(ptr) (((MemHead *)ptr) - 1)
#define PTR_FROM_MEMHEAD(memhead) (memhead + 1)
#define MEMHEAD_ALIGNED_FROM_PTR(ptr) (((MemHeadAligned *)ptr) - 1)
#define MEMHEAD_IS_ALIGNED(memhead) ((memhead)->len & (size_t)MEMHEAD_ALIGN_FLAG)
#define MEMHEAD_LEN(memhead) ((memhead)->len & ~((size_t)(MEMHEAD_ALIGN_FLAG)))

/* Uncomment this to have proper peak counter. */
#define USE_ATOMIC_MAX

__host__ __device__ void update_maximum(size_t *maximum_value, size_t value)
{
#ifdef USE_ATOMIC_MAX
  atomic_fetch_and_update_max_z(maximum_value, value);
#else
  *maximum_value = value > *maximum_value ? value : *maximum_value;
#endif
}

#ifdef __GNUC__
__attribute__((format(printf, 1, 2)))
#endif
__host__ __device__ static void
print_error(const char *str, ...)
{
  char buf[512];
  va_list ap;

  va_start(ap, str);
  vsnprintf(buf, sizeof(buf), str, ap);
  va_end(ap);
  buf[sizeof(buf) - 1] = '\0';
#ifdef __CUDA_ARCH__
  // Версия для device-кода
  if (d_error_callback) {
      d_error_callback(buf);
  }
#else
  // Версия для host-кода
  if (error_callback) {
      error_callback(buf);
  }
#endif
}

__host__ __device__ size_t MEM_lockfree_allocN_len(const void *vmemh)
{
  if (LIKELY(vmemh)) {
    return MEMHEAD_LEN(MEMHEAD_FROM_PTR(vmemh));
  }

  return 0;
}

__host__ __device__ void MEM_lockfree_freeN(void *vmemh)
{
#ifdef __CUDA_ARCH__
    // Версия для device-кода
    if (UNLIKELY(d_leak_detector_has_run)) {
        print_error("%s\n", d_free_after_leak_detection_message);
    }
#else
    // Версия для host-кода
    if (UNLIKELY(leak_detector_has_run)) {
        print_error("%s\n", free_after_leak_detection_message);
    }
#endif

  if (UNLIKELY(vmemh == NULL)) {
    print_error("Attempt to free NULL pointer\n");
#ifdef WITH_ASSERT_ABORT
    abort();
#endif
    return;
  }

  MemHead *memh = MEMHEAD_FROM_PTR(vmemh);
  size_t len = MEMHEAD_LEN(memh);

#ifdef __CUDA_ARCH__
  // Версия для device-кода
  atomic_sub_and_fetch_u(&d_totblock, 1);
  atomic_sub_and_fetch_z(&d_mem_in_use, len);

  if (UNLIKELY(d_malloc_debug_memset && len))
  {
      memset(memh + 1, 255, len);
  }
#else
  atomic_sub_and_fetch_u(&totblock, 1);
  atomic_sub_and_fetch_z(&mem_in_use, len);

  if (UNLIKELY(malloc_debug_memset && len))
  {
      memset(memh + 1, 255, len);
  }
#endif

  if (UNLIKELY(MEMHEAD_IS_ALIGNED(memh))) 
  {
    MemHeadAligned *memh_aligned = MEMHEAD_ALIGNED_FROM_PTR(vmemh);
    aligned_free(MEMHEAD_REAL_PTR(memh_aligned));
  }
  else {
    free(memh);
  }
}

__host__ __device__ void * MEM_lockfree_dupallocN(const void *vmemh)
{
  void *newp = nullptr;
  if (vmemh) 
  {
    MemHead *memh = MEMHEAD_FROM_PTR(vmemh);
    const size_t prev_size = MEM_lockfree_allocN_len(vmemh);
    if (UNLIKELY(MEMHEAD_IS_ALIGNED(memh))) 
    {
      MemHeadAligned *memh_aligned = MEMHEAD_ALIGNED_FROM_PTR(vmemh);
      newp = MEM_lockfree_mallocN_aligned(prev_size, (size_t)memh_aligned->alignment, "dupli_malloc");
    }
    else 
    {
      newp = MEM_lockfree_mallocN(prev_size, "dupli_malloc");
    }
    memcpy(newp, vmemh, prev_size);
  }
  return newp;
}

__host__ __device__ void *MEM_lockfree_reallocN_id(void *vmemh, size_t len, const char *str)
{
  void *newp = nullptr;

  if (vmemh) {
    MemHead *memh = MEMHEAD_FROM_PTR(vmemh);
    size_t old_len = MEM_lockfree_allocN_len(vmemh);

    if (LIKELY(!MEMHEAD_IS_ALIGNED(memh))) {
      newp = MEM_lockfree_mallocN(len, "realloc");
    }
    else {
      MemHeadAligned *memh_aligned = MEMHEAD_ALIGNED_FROM_PTR(vmemh);
      newp = MEM_lockfree_mallocN_aligned(len, (size_t)memh_aligned->alignment, "realloc");
    }

    if (newp) {
      if (len < old_len) {
        /* shrink */
        memcpy(newp, vmemh, len);
      }
      else {
        /* grow (or remain same size) */
        memcpy(newp, vmemh, old_len);
      }
    }

    MEM_lockfree_freeN(vmemh);
  }
  else {
    newp = MEM_lockfree_mallocN(len, str);
  }

  return newp;
}

__host__ __device__ void *MEM_lockfree_recallocN_id(void *vmemh, size_t len, const char *str)
{
  void *newp = nullptr;

  if (vmemh) {
    MemHead *memh = MEMHEAD_FROM_PTR(vmemh);
    size_t old_len = MEM_lockfree_allocN_len(vmemh);

    if (LIKELY(!MEMHEAD_IS_ALIGNED(memh))) {
      newp = MEM_lockfree_mallocN(len, "recalloc");
    }
    else {
      MemHeadAligned *memh_aligned = MEMHEAD_ALIGNED_FROM_PTR(vmemh);
      newp = MEM_lockfree_mallocN_aligned(len, (size_t)memh_aligned->alignment, "recalloc");
    }

    if (newp) {
      if (len < old_len) {
        /* shrink */
        memcpy(newp, vmemh, len);
      }
      else {
        memcpy(newp, vmemh, old_len);

        if (len > old_len) {
          /* grow */
          /* zero new bytes */
          memset(((char *)newp) + old_len, 0, len - old_len);
        }
      }
    }

    MEM_lockfree_freeN(vmemh);
  }
  else {
    newp = MEM_lockfree_callocN(len, str);
  }

  return newp;
}

__host__ __device__ void *MEM_lockfree_callocN(size_t len, const char *str)
{
	len = SIZET_ALIGN_4(len);

	auto* memh = static_cast<MemHead*>(calloc(1, len + sizeof(MemHead)));
#ifdef __CUDA_ARCH__
    // Версия для device-кода
    if (LIKELY(memh)) {
        memh->len = len;
        atomic_add_and_fetch_u(&d_totblock, 1);
        atomic_add_and_fetch_z(&d_mem_in_use, len);
        update_maximum(&d_peak_mem, d_mem_in_use);

        return PTR_FROM_MEMHEAD(memh);
    }
    print_error("Calloc returns null: len=" SIZET_FORMAT " in %s, total %u\n", SIZET_ARG(len), str, static_cast<uint>(d_mem_in_use));
#else
    // Версия для host-кода
    if (LIKELY(memh)) {
        memh->len = len;
        atomic_add_and_fetch_u(&totblock, 1);
        atomic_add_and_fetch_z(&mem_in_use, len);
        update_maximum(&peak_mem, mem_in_use);

        return PTR_FROM_MEMHEAD(memh);
    }
    print_error("Calloc returns null: len=" SIZET_FORMAT " in %s, total %u\n", SIZET_ARG(len), str, static_cast<uint>(mem_in_use));
#endif

	return nullptr;
}

__host__ __device__ void *MEM_lockfree_calloc_arrayN(size_t len, size_t size, const char *str)
{
  size_t total_size;
  if (UNLIKELY(!MEM_size_safe_multiply(len, size, &total_size))) {
#ifdef __CUDA_ARCH__
      // Версия для device-кода
      print_error("Calloc array aborted due to integer overflow: " "len=" SIZET_FORMAT "x" SIZET_FORMAT " in %s, total %u\n", SIZET_ARG(len), SIZET_ARG(size), str, (uint)d_mem_in_use);
#else
      // Версия для host-кода
      print_error("Calloc array aborted due to integer overflow: " "len=" SIZET_FORMAT "x" SIZET_FORMAT " in %s, total %u\n", SIZET_ARG(len), SIZET_ARG(size), str, (uint)mem_in_use);
#endif
    
    abort();
  }

  return MEM_lockfree_callocN(total_size, str);
}

__host__ __device__ void *MEM_lockfree_mallocN(size_t len, const char *str)
{
	len = SIZET_ALIGN_4(len);

	auto* memh = static_cast<MemHead*>(malloc(len + sizeof(MemHead)));

  if (LIKELY(memh)) 
  {
#ifdef __CUDA_ARCH__
      // Версия для device-кода
      if (UNLIKELY(d_malloc_debug_memset && len))
      {
          memset(memh + 1, 255, len);
      }

      memh->len = len;
      atomic_add_and_fetch_u(&d_totblock, 1);
      atomic_add_and_fetch_z(&d_mem_in_use, len);
      update_maximum(&d_peak_mem, d_mem_in_use);

      return PTR_FROM_MEMHEAD(memh);
  }
  print_error("Malloc returns null: len=" SIZET_FORMAT " in %s, total %u\n",
      SIZET_ARG(len),
      str,
      static_cast<uint>(d_mem_in_use));
#else
      // Версия для host-кода
      if (UNLIKELY(malloc_debug_memset && len))
      {
          memset(memh + 1, 255, len);
      }

      memh->len = len;
      atomic_add_and_fetch_u(&totblock, 1);
      atomic_add_and_fetch_z(&mem_in_use, len);
      update_maximum(&peak_mem, mem_in_use);

      return PTR_FROM_MEMHEAD(memh);
  }
  print_error("Malloc returns null: len=" SIZET_FORMAT " in %s, total %u\n",
      SIZET_ARG(len),
      str,
      static_cast<uint>(mem_in_use));
#endif
}

__host__ __device__ void *MEM_lockfree_malloc_arrayN(size_t len, size_t size, const char *str)
{
  size_t total_size;
  if (UNLIKELY(!MEM_size_safe_multiply(len, size, &total_size))) 
  {
#ifdef __CUDA_ARCH__
      // Версия для device-кода
      print_error("Malloc array aborted due to integer overflow: len=" SIZET_FORMAT "x" SIZET_FORMAT " in %s, total %u\n", SIZET_ARG(len), SIZET_ARG(size), str, (uint)d_mem_in_use);
#else
      // Версия для host-кода
      print_error("Malloc array aborted due to integer overflow: len=" SIZET_FORMAT "x" SIZET_FORMAT " in %s, total %u\n", SIZET_ARG(len), SIZET_ARG(size), str, (uint)mem_in_use);
#endif
    
    abort();
  }

  return MEM_lockfree_mallocN(total_size, str);
}

__host__ __device__ void *MEM_lockfree_mallocN_aligned(size_t len, size_t alignment, const char *str)
{
  /* Huge alignment values doesn't make sense and they wouldn't fit into 'short' used in the
   * MemHead. */
  assert(alignment < 1024);

  /* We only support alignments that are a power of two. */
  assert(IS_POW2(alignment));

  /* Some OS specific aligned allocators require a certain minimal alignment. */
  if (alignment < ALIGNED_MALLOC_MINIMUM_ALIGNMENT) {
    alignment = ALIGNED_MALLOC_MINIMUM_ALIGNMENT;
  }

  /* It's possible that MemHead's size is not properly aligned,
   * do extra padding to deal with this.
   *
   * We only support small alignments which fits into short in
   * order to save some bits in MemHead structure.
   */
  size_t extra_padding = MEMHEAD_ALIGN_PADDING(alignment);

  len = SIZET_ALIGN_4(len);

  MemHeadAligned *memh = (MemHeadAligned *)aligned_malloc(len + extra_padding + sizeof(MemHeadAligned), alignment);

  if (LIKELY(memh)) {
    /* We keep padding in the beginning of MemHead,
     * this way it's always possible to get MemHead
     * from the data pointer.
     */
    memh = (MemHeadAligned *)((char *)memh + extra_padding);

#ifdef __CUDA_ARCH__
    if (UNLIKELY(d_malloc_debug_memset && len)) {
      memset(memh + 1, 255, len);
    }

    memh->len = len | (size_t)MEMHEAD_ALIGN_FLAG;
    memh->alignment = (short)alignment;
    atomic_add_and_fetch_u(&d_totblock, 1);
    atomic_add_and_fetch_z(&d_mem_in_use, len);
    update_maximum(&d_peak_mem, d_mem_in_use);

    return PTR_FROM_MEMHEAD(memh);
  }
  // Версия для device-кода
  print_error("Malloc returns null: len=" SIZET_FORMAT " in %s, total %u\n", SIZET_ARG(len), str, (uint)d_mem_in_use);
#else
  // Версия для host-кода
    if (UNLIKELY(malloc_debug_memset && len)) {
        memset(memh + 1, 255, len);
    }

    memh->len = len | (size_t)MEMHEAD_ALIGN_FLAG;
    memh->alignment = (short)alignment;
    atomic_add_and_fetch_u(&totblock, 1);
    atomic_add_and_fetch_z(&mem_in_use, len);
    update_maximum(&peak_mem, mem_in_use);

    return PTR_FROM_MEMHEAD(memh);
  }
  print_error("Malloc returns null: len=" SIZET_FORMAT " in %s, total %u\n", SIZET_ARG(len), str, (uint)mem_in_use);
#endif
  
  return nullptr;
}

__host__ __device__ void MEM_lockfree_printmemlist_pydict(void)
{
}

__host__ __device__ void MEM_lockfree_printmemlist(void)
{
}

///* unused */
//__host__ __device__ void MEM_lockfree_callbackmemlist(void (*func)(void *))
//{
//  (void)func; /* Ignored. */
//}

__host__ __device__ void MEM_lockfree_printmemlist_stats(void)
{
#ifdef __CUDA_ARCH__
    // Версия для device-кода
    printf("\ntotal memory len: %.3f MB\n", (double)d_mem_in_use / (double)(1024 * 1024));
    printf("peak memory len: %.3f MB\n", (double)d_peak_mem / (double)(1024 * 1024));
    printf("\nFor more detailed per-block statistics run Blender with memory debugging command line \n");
#else
    // Версия для host-кода
    printf("\ntotal memory len: %.3f MB\n", (double)mem_in_use / (double)(1024 * 1024));
    printf("peak memory len: %.3f MB\n", (double)peak_mem / (double)(1024 * 1024));
    printf("\nFor more detailed per-block statistics run Blender with memory debugging command line \n");
#endif

#ifdef HAVE_MALLOC_STATS
  printf("System Statistics:\n");
  malloc_stats();
#endif
}

__host__ __device__ void MEM_lockfree_set_error_callback(void (*func)(const char *))
{
#ifdef __CUDA_ARCH__
    // Версия для device-кода
    d_error_callback = func;
#else
    // Версия для host-кода
    error_callback = func;
#endif
}

__host__ __device__ bool MEM_lockfree_consistency_check(void)
{
  return true;
}

__host__ __device__ void MEM_lockfree_set_memory_debug(void)
{
#ifdef __CUDA_ARCH__
    // Версия для device-кода
    d_malloc_debug_memset = true;
#else
    // Версия для host-кода
    malloc_debug_memset = true;
#endif
}

__host__ __device__ size_t MEM_lockfree_get_memory_in_use(void)
{
#ifdef __CUDA_ARCH__
    // Версия для device-кода
    return d_mem_in_use;
#else
    // Версия для host-кода
    return mem_in_use;
#endif
}

__host__ __device__ uint MEM_lockfree_get_memory_blocks_in_use(void)
{
#ifdef __CUDA_ARCH__
    // Версия для device-кода
    return d_totblock;
#else
    // Версия для host-кода
    return totblock;
#endif
}

/* dummy */
__host__ __device__ void MEM_lockfree_reset_peak_memory(void)
{
#ifdef __CUDA_ARCH__
    // Версия для device-кода
    d_peak_mem = d_mem_in_use;
#else
    // Версия для host-кода
    peak_mem = mem_in_use;
#endif
}

__host__ __device__ size_t MEM_lockfree_get_peak_memory(void)
{
#ifdef __CUDA_ARCH__
    // Версия для device-кода
    return d_peak_mem;
#else
    // Версия для host-кода
    return peak_mem;
#endif
}

#ifndef NDEBUG
__host__ __device__ const char *MEM_lockfree_name_ptr(void *vmemh)
{
  if (vmemh) {
    return "unknown block name ptr";
  }

  return "MEM_lockfree_name_ptr(NULL)";
}
#endif /* NDEBUG */
