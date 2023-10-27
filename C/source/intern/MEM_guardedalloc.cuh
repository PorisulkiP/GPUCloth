#pragma once

#ifndef __MEM_GUARDEDALLOC_H__
#define __MEM_GUARDEDALLOC_H__

/* Необходимо для uintptr_t и атрибутов, исключений, не используйте BLI в других частях MEM_* */
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <assert.h>
#include "mallocn_intern.cuh"

#pragma warning(disable: 4244)

#define BLI_array_alloca(arr, realsize) alloca(sizeof(*(arr)) * (realsize))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) std::exit(code);
    }
}

#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__device__ inline void cdpAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) assert(0);
    }
}

//
///** Возвращает размер выделенного блока памяти, на который указывает
// * vmemh. Если указатель не был ранее выделен этим
// * модулем, результат неопределен. */
////__host__ __device__ inline size_t(*MEM_lockfree_allocN_len)(const void* vmemh);
//
///**
// * Освобождает память, ранее выделенную этим модулем.
// */
//__host__ __device__ inline void (*MEM_lockfree_freeN)(void* vmemh);
//
///**
// * Дублирует блок памяти и возвращает указатель на
// * вновь выделенный блок. */
//__host__ __device__ inline void* (*MEM_lockfree_dupallocN)(const void* vmemh);
//
///**
// * Перевыделяет блок памяти и возвращает указатель на новый
// * выделенный блок, старый блок освобождается. Это не так оптимизировано
// * как системный realloc, но просто создает новое выделение и копирует
// * данные из существующей памяти. */
//__host__ __device__ inline void* (*MEM_lockfree_reallocN_id)(void* vmemh, size_t len, const char* str);
//
///**
// * Вариант realloc, который заполняет новые байты нулями.
// */
//__host__ __device__ inline void* (*MEM_lockfree_recallocN_id)(void* vmemh, size_t len, const char* str);

#define MEM_reallocN(vmemh, len) MEM_lockfree_reallocN_id(vmemh, len, __func__)
#define MEM_recallocN(vmemh, len) MEM_lockfree_recallocN_id(vmemh, len, __func__)
//
///**
// * Выделяет блок памяти размером len с тегом str. Память
// * очищается. Имя должно быть статическим, потому что хранится только указатель на него! */
//void* (*MEM_lockfree_callocN)(size_t len, const char* str);
//
///**
// * Выделяет блок памяти размером (len * size) с тегом
// * str, прерывая работу в случае переполнения целых чисел для предотвращения уязвимостей.
// * Память очищается. Имя должно быть статическим, потому что хранится только указатель на него! */
//void* (*MEM_lockfree_calloc_arrayN)(size_t len, size_t size, const char* str);
//
///**
// * Выделяет блок памяти размером len с тегом str. Имя
// * должно быть стат * должно быть статическим, потому что хранится только указатель на него!
// */
//void* (*MEM_lockfree_mallocN)(size_t len, const char* str);
//
///**
// * Выделяет блок памяти размером (len * size) с тегом str,
// * прерывая работу в случае переполнения целых чисел для предотвращения уязвимостей. Имя
// * должно быть статическим, потому что хранится только указатель на него!
// */
//void* (*MEM_lockfree_malloc_arrayN)(size_t len, size_t size, const char* str);
//
///**
// * Выделяет выровненный блок памяти размером len с тегом str. Имя
// * должно быть статическим, потому что хранится только указатель на него!
// */
//void* (*MEM_lockfree_mallocN_aligned)(size_t len, size_t alignment, const char* str);
//
///** Выводит список имен и размеров всех выделенных блоков памяти
// * в виде словаря Python для удобного исследования. */
//void (*MEM_lockfree_printmemlist_pydict)(void);
//
///** Выводит список имен и размеров всех выделенных блоков памяти. */
//void (*MEM_lockfree_printmemlist)(void);
//
///** Вызывает функцию для всех выделенных блоков памяти. */
//void (*MEM_lockfree_callbackmemlist)(void (*func)(void*));
//
///** Выводит статистику об использовании памяти. */
//void (*MEM_lockfree_printmemlist_stats)(void);
//
///** Устанавливает функцию обратного вызова для вывода ошибок. */
//void (*MEM_lockfree_set_error_callback)(void (*func)(const char*));
//
///**
// * Все ли начальные/конечные маркеры блоков корректны?
// *
// * \retval true для корректной памяти, false для поврежденной памяти. */
//bool (*MEM_lockfree_consistency_check)(void);
//
///** Пытается принудить OSX (или другие ОС) иметь ненулевой malloc и стек. */
//void (*MEM_lockfree_set_memory_debug)(void);
//
///** Статистика использования памяти. */
//size_t (*MEM_lockfree_get_memory_in_use)(void);
///** Получение количества используемых блоков памяти. */
//uint   (*MEM_lockfree_get_memory_blocks_in_use)(void);
//
///** Сброс статистики пика использования памяти до нуля. */
//void   (*MEM_lockfree_reset_peak_memory)(void);
//
///** Получение пика использования памяти в байтах, включая mmap-выделения. */
//size_t (*MEM_lockfree_get_peak_memory)(void);


/* накладные расходы для lockfree аллокатора (используйте, чтобы избежать лишнего пространства) */
#define MEM_SIZE_OVERHEAD sizeof(size_t)
#define MEM_SIZE_OPTIMAL(size) ((size)-MEM_SIZE_OVERHEAD)
//
//#ifndef NDEBUG
//const char* (*MEM_lockfree_name_ptr)(void* vmemh);
//#endif
//
///** Эту функцию следует вызывать как можно раньше в программе. После ее вызова при выходе будут выведены
// * сведения о утечках памяти. */
//__host__ __device__ void MEM_init_memleak_detection(void);
//
///**
// * Используйте это, если мы хотим вызвать #exit, например, во время разбора аргументов,
// * не освобождая все данные.
// */
//__host__ __device__ void MEM_use_memleak_detection(bool enabled);
//
///** Если после вызова этой функции будут обнаружены утечки памяти, процесс завершится с кодом ошибки,
// * указывающим на сбой. Это можно использовать при проверке утечек памяти с автоматическими
// * тестами. */
//__host__ __device__ void MEM_enable_fail_on_memleak(void);
//
///* Переключение аллокатора на быстрый режим с меньшим количеством отслеживания.
// *
// * Используйте в рабочем коде, где приоритетом является производительность, а точные сведения об
// * выделении памяти не важны. Этот аллокатор отслеживает количество выделений и количество
// * выделенных байт, но не отслеживает имена выделенных блоков.
// *
// * Заметка: Переключение между типами аллокаторов может происходить только до того, как произошло
// * любое выделение памяти. */
//__host__ __device__ void MEM_use_lockfree_allocator(void);
//
///* Переключение аллокатора на медленный полностью защищенный режим.
// *
// * Используйте для отладочных целей. Этот аллокатор содержит блокировку вокруг каждого вызова
// * аллокатора, что делает его медленным. Полученным преимуществом является возможность иметь список
// * выделенных блоков (помимо отслеживания количества выделений и объема выделенных байт).
// *
// * Заметка: Переключение между типами аллокаторов может происходить только до того, как произошло
// * любое выделение памяти. */
//__host__ __device__ void MEM_use_guarded_allocator(void);

///**
//* Returns the length of the allocated memory segment pointed at
//* by vmemh. If the pointer was not previously allocated by this
//* module, the result is undefined.
//*/
//size_t(*MEM_lockfree_allocN_len)(const void* vmemh);
//
///**
//* Release memory previously allocated by this module.
//*/
//void (*MEM_lockfree_freeN)(void* vmemh);
//
///**
//* Duplicates a block of memory, and returns a pointer to the
//* newly allocated block.
//* NULL-safe; will return NULL when receiving a NULL pointer. */
//void* (*MEM_lockfree_dupallocN)(const void* vmemh);
//
///**
//* Reallocates a block of memory, and returns pointer to the newly
//* allocated block, the old one is freed. this is not as optimized
//* as a system realloc but just makes a new allocation and copies
//* over from existing memory. */
//void* (*MEM_lockfree_reallocN_id)(void* vmemh, size_t len, const char* str);
//
///**
//* A variant of realloc which zeros new bytes
//*/
//void* (*MEM_lockfree_recallocN_id)(void* vmemh, size_t len, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(2);

#define MEM_reallocN(vmemh, len) MEM_lockfree_reallocN_id(vmemh, len, __func__)
#define MEM_recallocN(vmemh, len) MEM_lockfree_recallocN_id(vmemh, len, __func__)
//
///**
//* Allocate a block of memory of size len, with tag name str. The
//* memory is cleared. The name must be static, because only a
//* pointer to it is stored!
//*/
//void* (*MEM_lockfree_callocN)(size_t len, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(1) ATTR_NONNULL(2);
//
///**
//* Allocate a block of memory of size (len * size), with tag name
//* str, aborting in case of integer overflows to prevent vulnerabilities.
//* The memory is cleared. The name must be static, because only a
//* pointer to it is stored ! */
//void* (*MEM_lockfree_calloc_arrayN)(size_t len, size_t size, const char* str);
//
///**
//* Allocate a block of memory of size len, with tag name str. The
//* name must be a static, because only a pointer to it is stored !
//*/
//void* (*MEM_lockfree_mallocN)(size_t len, const char* str);
//
///**
//* Allocate a block of memory of size (len * size), with tag name str,
//* aborting in case of integer overflow to prevent vulnerabilities. The
//* name must be a static, because only a pointer to it is stored !
//*/
//void* (*MEM_lockfree_malloc_arrayN)(size_t len, size_t size, const char* str) ;
//
///**
//* Allocate an aligned block of memory of size len, with tag name str. The
//* name must be a static, because only a pointer to it is stored !
//*/
//void* (*MEM_lockfree_mallocN_aligned)(size_t len, size_t alignment, const char* str);
//
///**
//* Print a list of the names and sizes of all allocated memory
//* blocks. as a python dict for easy investigation.
//*/
//void (*MEM_lockfree_printmemlist_pydict)(void);
//
///**
//* Print a list of the names and sizes of all allocated memory blocks.
//*/
//void (*MEM_lockfree_printmemlist)(void);
//
///** calls the function on all allocated memory blocks. */
//void (*MEM_lockfree_callbackmemlist)(void (*func)(void*));
//
///** Print statistics about memory usage */
//void (*MEM_lockfree_printmemlist_stats)(void);
//
///** Set the callback function for error output. */
//void (*MEM_lockfree_set_error_callback)(void (*func)(const char*));
//
///**
//* Are the start/end block markers still correct ?
//*
//* \retval true for correct memory, false for corrupted memory.
//*/
//bool (*MEM_lockfree_consistency_check)(void);
//
///** Attempt to enforce OSX (or other OS's) to have malloc and stack nonzero */
//void (*MEM_lockfree_set_memory_debug)(void);
//
///** Memory usage stats. */
//size_t(*MEM_lockfree_get_memory_in_use)(void);
///** Get amount of memory blocks in use. */
//uint (*MEM_lockfree_get_memory_blocks_in_use)(void);
//
///** Reset the peak memory statistic to zero. */
//void (*MEM_lockfree_reset_peak_memory)(void);
//
///** Get the peak memory usage in bytes, including mmap allocations. */
//size_t(*MEM_lockfree_get_peak_memory)(void) ATTR_WARN_UNUSED_RESULT;

/* overhead for lockfree allocator (use to avoid slop-space) */
#define MEM_SIZE_OVERHEAD sizeof(size_t)
#define MEM_SIZE_OPTIMAL(size) ((size)-MEM_SIZE_OVERHEAD)

///**
//* This should be called as early as possible in the program. When it has been called, information
//* about memory leaks will be printed on exit.
//*/
//__host__ __device__ void MEM_init_memleak_detection(void);
//
///**
//* Use this if we want to call #exit during argument parsing for example,
//* without having to free all data.
//*/
//__host__ __device__ void MEM_use_memleak_detection(bool enabled);
//
///**
//* When this has been called and memory leaks have been detected, the process will have an exit
//* code that indicates failure. This can be used for when checking for memory leaks with automated
//* tests.
//*/
//__host__ __device__ void MEM_enable_fail_on_memleak(void);
//
///* Switch allocator to fast mode, with less tracking.
//*
//* Use in the production code where performance is the priority, and exact details about allocation
//* is not. This allocator keeps track of number of allocation and amount of allocated bytes, but it
//* does not track of names of allocated blocks.
//*
//* NOTE: The switch between allocator types can only happen before any allocation did happen. */
//__host__ __device__ void MEM_use_lockfree_allocator(void);
//
///* Switch allocator to slow fully guarded mode.
//*
//* Use for debug purposes. This allocator contains lock section around every allocator call, which
//* makes it slow. What is gained with this is the ability to have list of allocated blocks (in an
//* addition to the tracking of number of allocations and amount of allocated bytes).
//*
//* NOTE: The switch between allocator types can only happen before any allocation did happen. */
//__host__ __device__ void MEM_use_guarded_allocator(void);

#include <new>
#include <type_traits>
#include <utility>

//
///**
// * Allocate memory for an object of type #T and copy construct an object from `other`.
// * Only applicable for a trivial types.
// *
// * This function works around problem of copy-constructing DNA structs which contains deprecated
// * fields: some compilers will generate access deprecated field in implicitly defined copy
// * constructors.
// *
// * This is a better alternative to #MEM_lockfree_dupallocN.
// */
//template<typename T>
//__host__ __device__  T* MEM_cnew(const char* allocation_name, const T & other)
//{
//    static_assert(std::is_trivial_v<T>, "For non-trivial types, MEM_new should be used.");
//    T* new_object = static_cast<T*>(MEM_lockfree_mallocN(sizeof(T), allocation_name));
//    memcpy(new_object, &other, sizeof(T));
//    return new_object;
//}
//
///**
// * Destructs and deallocates an object previously allocated with any `MEM_*` function.
// * Passing in null does nothing.
// */
//template<typename T>
//__host__ __device__  void MEM_delete(const T * ptr)
//{
//    if (ptr == nullptr) {
//        /* Support #ptr being null, because C++ `delete` supports that as well. */
//        return;
//    }
//    /* C++ allows destruction of const objects, so the pointer is allowed to be const. */
//    ptr->~T();
//    MEM_lockfree_freeN(const_cast<T*>(ptr));
//}

///* Allocation functions (for C++ only). */
//#  define MEM_CXX_CLASS_ALLOC_FUNCS(_id) \
//   public: \
//    __host__ __device__ void *operator new(size_t num_bytes) \
//    { \
//      return MEM_lockfree_mallocN(num_bytes, _id); \
//    } \
//    __host__ __device__ void operator delete(void *mem) \
//    { \
//      if (mem) { \
//        MEM_lockfree_freeN(mem); \
//      } \
//    } \
//    __host__ __device__ void *operator new[](size_t num_bytes) \
//    { \
//      return MEM_lockfree_mallocN(num_bytes, _id "[]"); \
//    } \
//    __host__ __device__ void operator delete[](void *mem) \
//    { \
//      if (mem) { \
//        MEM_lockfree_freeN(mem); \
//      } \
//    } \
//    __host__ __device__ void *operator new(size_t /*count*/, void *ptr) \
//    { \
//      return ptr; \
//    } \
//    /* This is the matching delete operator to the placement-new operator above. Both parameters \
//     * will have the same value. Without this, we get the warning C4291 on windows. */ \
//    __host__ __device__ void operator delete(void * /*ptr_to_free*/, void * /*ptr*/) \
//    { \
//    }

#if __CUDA_ARCH__ > 100      // Atomics only used with > sm_10 architecture
	#include <sm_60_atomic_functions.h>
#endif

// This is the smallest amount of memory, per-thread, which is allowed.
// It is also the largest amount of space a single printf() can take up
const static int CUPRINTF_MAX_LEN = 256;

// This structure is used internally to track block/thread output restrictions.
typedef struct __align__(8) {
    int threadid;                           // CUPRINTF_UNRESTRICTED for unrestricted
    int blockid;                            // CUPRINTF_UNRESTRICTED for unrestricted
} cuPrintfRestriction;

// The main storage is in a global print buffer, which has a known
// start/end/length. These are atomically updated so it works as a
// circular buffer.
// Since the only control primitive that can be used is atomicAdd(),
// we cannot wrap the pointer as such. The actual address must be
// calculated from printfBufferPtr by mod-ing with printfBufferLength.
// For sm_10 architecture, we must subdivide the buffer per-thread
// since we do not even have an atomic primitive.
__constant__ static char* globalPrintfBuffer = NULL;         // Start of circular buffer (set up by host)
__constant__ static int printfBufferLength = 0;              // Size of circular buffer (set up by host)
__device__ static cuPrintfRestriction restrictRules;         // Output restrictions
__device__ volatile static char* printfBufferPtr = NULL;     // Current atomically-incremented non-wrapped offset

// This is the header preceeding all printf entries.
// NOTE: It *must* be size-aligned to the maximum entity size (size_t)
typedef struct __align__(8) {
    unsigned short magic;                   // Magic number says we're valid
    unsigned short fmtoffset;               // Offset of fmt string into buffer
    unsigned short blockid;                 // Block ID of author
    unsigned short threadid;                // Thread ID of author
} cuPrintfHeader;

// Special header for sm_10 architecture
#define CUPRINTF_SM10_MAGIC   0xC810        // Not a valid ascii character
typedef struct __align__(16) {
    unsigned short magic;                   // sm_10 specific magic number
    unsigned short unused;
    unsigned int thread_index;              // thread ID for this buffer
    unsigned int thread_buf_len;            // per-thread buffer length
    unsigned int offset;                    // most recent printf's offset
} cuPrintfHeaderSM10;


// Because we can't write an element which is not aligned to its bit-size,
// we have to align all sizes and variables on maximum-size boundaries.
// That means sizeof(double) in this case, but we'll use (long long) for
// better arch<1.3 support
#define CUPRINTF_ALIGN_SIZE      sizeof(long long)

// All our headers are prefixed with a magic number so we know they're ready
#define CUPRINTF_SM11_MAGIC  (unsigned short)0xC811        // Not a valid ascii character

#define CUPRINTF_UNRESTRICTED   -1

__device__ static char* getNextPrintfBufPtr()
{
    // Initialisation check
    if (!printfBufferPtr)
        return nullptr;

    // Conditional section, dependent on architecture
#ifdef __CUDA_ARCH__
    // Thread/block restriction check
    if ((restrictRules.blockid != CUPRINTF_UNRESTRICTED) && (restrictRules.blockid != (blockIdx.x + gridDim.x * blockIdx.y)))
        return NULL;
    if ((restrictRules.threadid != CUPRINTF_UNRESTRICTED) && (restrictRules.threadid != (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z)))
        return NULL;

        // Much easier with an atomic operation!
    size_t offset = atomicAdd((unsigned int*)&printfBufferPtr, CUPRINTF_MAX_LEN) - (size_t)globalPrintfBuffer;
    offset %= printfBufferLength;
    return globalPrintfBuffer + offset;
#else
    return nullptr;
#endif
}

__device__ static char* cuPrintfStrncpy(char* dest, const char* src, int n, char* end)
{
    // Initialisation and overflow check
    if (!dest || !src || (dest >= end))
        return NULL;

    // Prepare to write the length specifier. We're guaranteed to have
    // at least "CUPRINTF_ALIGN_SIZE" bytes left because we only write out in
    // chunks that size, and CUPRINTF_MAX_LEN is aligned with CUPRINTF_ALIGN_SIZE.
    int* lenptr = (int*)(void*)dest;
    int len = 0;
    dest += CUPRINTF_ALIGN_SIZE;

    // Now copy the string
    while (n--)
    {
        if (dest >= end)     // Overflow check
            break;

        len++;
        *dest++ = *src;
        if (*src++ == '\0')
            break;
    }

    // Now write out the padding bytes, and we have our length.
    while ((dest < end) && (((size_t)dest & (CUPRINTF_ALIGN_SIZE - 1)) != 0))
    {
        len++;
        *dest++ = 0;
    }
    *lenptr = len;
    return (dest < end) ? dest : NULL;        // Overflow means return NULL
}

__device__ static char* copyArg(char* ptr, const char* arg, char* end)
{
    // Initialisation check
    if (!ptr || !arg)
        return NULL;

    // strncpy does all our work. We just terminate.
    if ((ptr = cuPrintfStrncpy(ptr, arg, CUPRINTF_MAX_LEN, end)) != NULL)
        *ptr = 0;

    return ptr;
}

template <typename T>
__device__ static char* copyArg(char* ptr, T& arg, char* end)
{
    // Initisalisation and overflow check. Alignment rules mean that
    // we're at least CUPRINTF_ALIGN_SIZE away from "end", so we only need
    // to check that one offset.
    if (!ptr || ((ptr + CUPRINTF_ALIGN_SIZE) >= end))
        return NULL;

    // Write the length and argument
    *(int*)(void*)ptr = sizeof(arg);
    ptr += CUPRINTF_ALIGN_SIZE;
    *(T*)(void*)ptr = arg;
    ptr += CUPRINTF_ALIGN_SIZE;
    *ptr = 0;

    return ptr;
}

__device__ static void writePrintfHeader(char* ptr, char* fmtptr)
{
#ifdef __CUDA_ARCH__
    if (ptr)
    {
        cuPrintfHeader header;
        header.magic = CUPRINTF_SM11_MAGIC;
        header.fmtoffset = (unsigned short)(fmtptr - ptr);
        header.blockid = blockIdx.x + gridDim.x * blockIdx.y;
        header.threadid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
        *(cuPrintfHeader*)(void*)ptr = header;
    }
#endif
}

// All printf variants basically do the same thing, setting up the
// buffer, writing all arguments, then finalising the header. For
// clarity, we'll pack the code into some big macros.
#define CUPRINTF_PREAMBLE \
    char *start, *end, *bufptr, *fmtstart; \
    if((start = getNextPrintfBufPtr()) == NULL) return 0; \
    end = start + CUPRINTF_MAX_LEN; \
    bufptr = start + sizeof(cuPrintfHeader);

// Posting an argument is easy
#define CUPRINTF_ARG(argname) \
        bufptr = copyArg(bufptr, argname, end);

// After args are done, record start-of-fmt and write the fmt and header
#define CUPRINTF_POSTAMBLE \
    fmtstart = bufptr; \
    end = cuPrintfStrncpy(bufptr, fmt, CUPRINTF_MAX_LEN, end); \
    writePrintfHeader(start, end ? fmtstart : NULL); \
    return end ? (int)(end - start) : 0;

__device__ inline int cuPrintf(const char* fmt)
{
    CUPRINTF_PREAMBLE

    CUPRINTF_POSTAMBLE
}
template <typename T1> __device__ int cuPrintf(const char* fmt, T1 arg1)
{
    CUPRINTF_PREAMBLE;

    CUPRINTF_ARG(arg1);

    CUPRINTF_POSTAMBLE;
}

#endif /* __MEM_GUARDEDALLOC_H__ */