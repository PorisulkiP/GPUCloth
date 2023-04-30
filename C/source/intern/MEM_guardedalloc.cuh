#ifndef __MEM_GUARDEDALLOC_H__
#define __MEM_GUARDEDALLOC_H__

/* ���������� ��� uintptr_t � ���������, ����������, �� ����������� BLI � ������ ������ MEM_* */
#include "compiler_attrs.cuh"
#include "sys_types.cuh"

/** ���������� ������ ����������� ����� ������, �� ������� ���������
 * vmemh. ���� ��������� �� ��� ����� ������� ����
 * �������, ��������� �����������. */
extern size_t(*MEM_allocN_len)(const void* vmemh) ATTR_WARN_UNUSED_RESULT;

#pragma warning(disable: 4244)

#define BLI_array_alloca(arr, realsize) alloca(sizeof(*arr) * (realsize))

/**
 * ����������� ������, ����� ���������� ���� �������.
 */
extern void (*MEM_freeN)(void* vmemh);

/**
 * ��������� ���� ������ � ���������� ��������� ��
 * ����� ���������� ����. */
extern void* (*MEM_dupallocN)(const void* vmemh) ATTR_WARN_UNUSED_RESULT;

/**
 * ������������ ���� ������ � ���������� ��������� �� �����
 * ���������� ����, ������ ���� �������������. ��� �� ��� ��������������
 * ��� ��������� realloc, �� ������ ������� ����� ��������� � ��������
 * ������ �� ������������ ������. */
extern void* (*MEM_reallocN_id)(void* vmemh, size_t len, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(2);

/**
 * ������� realloc, ������� ��������� ����� ����� ������.
 */
extern void* (*MEM_recallocN_id)(void* vmemh, size_t len, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(2);

#define MEM_reallocN(vmemh, len) MEM_reallocN_id(vmemh, len, __func__)
#define MEM_recallocN(vmemh, len) MEM_recallocN_id(vmemh, len, __func__)

/**
 * �������� ���� ������ �������� len � ����� str. ������
 * ���������. ��� ������ ���� �����������, ������ ��� �������� ������ ��������� �� ����! */
extern void* (*MEM_callocN)(size_t len, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(1) ATTR_NONNULL(2);

/**
 * �������� ���� ������ �������� (len * size) � �����
 * str, �������� ������ � ������ ������������ ����� ����� ��� �������������� �����������.
 * ������ ���������. ��� ������ ���� �����������, ������ ��� �������� ������ ��������� �� ����! */
extern void* (*MEM_calloc_arrayN)(size_t len, size_t size, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(1, 2) ATTR_NONNULL(3);

/**
 * �������� ���� ������ �������� len � ����� str. ���
 * ������ ���� ���� * ������ ���� �����������, ������ ��� �������� ������ ��������� �� ����!
 */
extern void* (*MEM_mallocN)(size_t len, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(1) ATTR_NONNULL(2);

/**
 * �������� ���� ������ �������� (len * size) � ����� str,
 * �������� ������ � ������ ������������ ����� ����� ��� �������������� �����������. ���
 * ������ ���� �����������, ������ ��� �������� ������ ��������� �� ����!
 */
extern void* (*MEM_malloc_arrayN)(size_t len, size_t size, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(1, 2) ATTR_NONNULL(3);

/**
 * �������� ����������� ���� ������ �������� len � ����� str. ���
 * ������ ���� �����������, ������ ��� �������� ������ ��������� �� ����!
 */
extern void* (*MEM_mallocN_aligned)(size_t len, size_t alignment, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(1) ATTR_NONNULL(3);

/** ������� ������ ���� � �������� ���� ���������� ������ ������
 * � ���� ������� Python ��� �������� ������������. */
extern void (*MEM_printmemlist_pydict)(void);

/** ������� ������ ���� � �������� ���� ���������� ������ ������. */
extern void (*MEM_printmemlist)(void);

/** �������� ������� ��� ���� ���������� ������ ������. */
extern void (*MEM_callbackmemlist)(void (*func)(void*));

/** ������� ���������� �� ������������� ������. */
extern void (*MEM_printmemlist_stats)(void);

/** ������������� ������� ��������� ������ ��� ������ ������. */
extern void (*MEM_set_error_callback)(void (*func)(const char*));

/**
 * ��� �� ���������/�������� ������� ������ ���������?
 *
 * \retval true ��� ���������� ������, false ��� ������������ ������. */
extern bool (*MEM_consistency_check)(void);

/** �������� ��������� OSX (��� ������ ��) ����� ��������� malloc � ����. */
extern void (*MEM_set_memory_debug)(void);

/** ���������� ������������� ������. */
extern size_t (*MEM_get_memory_in_use)(void);
/** ��������� ���������� ������������ ������ ������. */
extern uint   (*MEM_get_memory_blocks_in_use)(void);

/** ����� ���������� ���� ������������� ������ �� ����. */
extern void   (*MEM_reset_peak_memory)(void);

/** ��������� ���� ������������� ������ � ������, ������� mmap-���������. */
extern size_t (*MEM_get_peak_memory)(void) ATTR_WARN_UNUSED_RESULT;

template<typename T>
void MEM_SAFE_FREE(T*& ptr)
{
    if (ptr)
    {
        MEM_freeN((void*)ptr);
        ptr = nullptr;
    }
}
/* ��������� ������� ��� lockfree ���������� (�����������, ����� �������� ������� ������������) */
#define MEM_SIZE_OVERHEAD sizeof(size_t)
#define MEM_SIZE_OPTIMAL(size) ((size)-MEM_SIZE_OVERHEAD)

#ifndef NDEBUG
    extern const char* (*MEM_name_ptr)(void* vmemh);
#endif

/** ��� ������� ������� �������� ��� ����� ������ � ���������. ����� �� ������ ��� ������ ����� ��������
 * �������� � ������� ������. */
void MEM_init_memleak_detection(void);

/**
 * ����������� ���, ���� �� ����� ������� #exit, ��������, �� ����� ������� ����������,
 * �� ���������� ��� ������.
 */
void MEM_use_memleak_detection(bool enabled);

/** ���� ����� ������ ���� ������� ����� ���������� ������ ������, ������� ���������� � ����� ������,
 * ����������� �� ����. ��� ����� ������������ ��� �������� ������ ������ � ���������������
 * �������. */
void MEM_enable_fail_on_memleak(void);

/* ������������ ���������� �� ������� ����� � ������� ����������� ������������.
 *
 * ����������� � ������� ����, ��� ����������� �������� ������������������, � ������ �������� ��
 * ��������� ������ �� �����. ���� ��������� ����������� ���������� ��������� � ����������
 * ���������� ����, �� �� ����������� ����� ���������� ������.
 *
 * �������: ������������ ����� ������ ����������� ����� ����������� ������ �� ����, ��� ���������
 * ����� ��������� ������. */
void MEM_use_lockfree_allocator(void);

/* ������������ ���������� �� ��������� ��������� ���������� �����.
 *
 * ����������� ��� ���������� �����. ���� ��������� �������� ���������� ������ ������� ������
 * ����������, ��� ������ ��� ���������. ���������� ������������� �������� ����������� ����� ������
 * ���������� ������ (������ ������������ ���������� ��������� � ������ ���������� ����).
 *
 * �������: ������������ ����� ������ ����������� ����� ����������� ������ �� ����, ��� ���������
 * ����� ��������� ������. */
void MEM_use_guarded_allocator(void);

/**
* Returns the length of the allocated memory segment pointed at
* by vmemh. If the pointer was not previously allocated by this
* module, the result is undefined.
*/
extern size_t(*MEM_allocN_len)(const void* vmemh) ATTR_WARN_UNUSED_RESULT;

/**
* Release memory previously allocated by this module.
*/
extern void (*MEM_freeN)(void* vmemh);

/**
* Duplicates a block of memory, and returns a pointer to the
* newly allocated block.
* NULL-safe; will return NULL when receiving a NULL pointer. */
extern void* (*MEM_dupallocN)(const void* vmemh) ATTR_WARN_UNUSED_RESULT;

/**
* Reallocates a block of memory, and returns pointer to the newly
* allocated block, the old one is freed. this is not as optimized
* as a system realloc but just makes a new allocation and copies
* over from existing memory. */
extern void* (*MEM_reallocN_id)(void* vmemh, size_t len, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(2);

/**
* A variant of realloc which zeros new bytes
*/
extern void* (*MEM_recallocN_id)(void* vmemh, size_t len, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(2);

#define MEM_reallocN(vmemh, len) MEM_reallocN_id(vmemh, len, __func__)
#define MEM_recallocN(vmemh, len) MEM_recallocN_id(vmemh, len, __func__)

/**
* Allocate a block of memory of size len, with tag name str. The
* memory is cleared. The name must be static, because only a
* pointer to it is stored!
*/
extern void* (*MEM_callocN)(size_t len, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(1) ATTR_NONNULL(2);

/**
* Allocate a block of memory of size (len * size), with tag name
* str, aborting in case of integer overflows to prevent vulnerabilities.
* The memory is cleared. The name must be static, because only a
* pointer to it is stored ! */
extern void* (*MEM_calloc_arrayN)(size_t len, size_t size, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(1, 2) ATTR_NONNULL(3);

/**
* Allocate a block of memory of size len, with tag name str. The
* name must be a static, because only a pointer to it is stored !
*/
extern void* (*MEM_mallocN)(size_t len, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(1) ATTR_NONNULL(2);

/**
* Allocate a block of memory of size (len * size), with tag name str,
* aborting in case of integer overflow to prevent vulnerabilities. The
* name must be a static, because only a pointer to it is stored !
*/
extern void* (*MEM_malloc_arrayN)(size_t len, size_t size, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(1, 2) ATTR_NONNULL(3);

/**
* Allocate an aligned block of memory of size len, with tag name str. The
* name must be a static, because only a pointer to it is stored !
*/
extern void* (*MEM_mallocN_aligned)(size_t len, size_t alignment, const char* str) ATTR_WARN_UNUSED_RESULT ATTR_ALLOC_SIZE(1) ATTR_NONNULL(3);

/**
* Print a list of the names and sizes of all allocated memory
* blocks. as a python dict for easy investigation.
*/
extern void (*MEM_printmemlist_pydict)(void);

/**
* Print a list of the names and sizes of all allocated memory blocks.
*/
extern void (*MEM_printmemlist)(void);

/** calls the function on all allocated memory blocks. */
extern void (*MEM_callbackmemlist)(void (*func)(void*));

/** Print statistics about memory usage */
extern void (*MEM_printmemlist_stats)(void);

/** Set the callback function for error output. */
extern void (*MEM_set_error_callback)(void (*func)(const char*));

/**
* Are the start/end block markers still correct ?
*
* \retval true for correct memory, false for corrupted memory.
*/
extern bool (*MEM_consistency_check)(void);

/** Attempt to enforce OSX (or other OS's) to have malloc and stack nonzero */
extern void (*MEM_set_memory_debug)(void);

/** Memory usage stats. */
extern size_t(*MEM_get_memory_in_use)(void);
/** Get amount of memory blocks in use. */
extern uint (*MEM_get_memory_blocks_in_use)(void);

/** Reset the peak memory statistic to zero. */
extern void (*MEM_reset_peak_memory)(void);

/** Get the peak memory usage in bytes, including mmap allocations. */
extern size_t(*MEM_get_peak_memory)(void) ATTR_WARN_UNUSED_RESULT;

/* overhead for lockfree allocator (use to avoid slop-space) */
#define MEM_SIZE_OVERHEAD sizeof(size_t)
#define MEM_SIZE_OPTIMAL(size) ((size)-MEM_SIZE_OVERHEAD)

/**
* This should be called as early as possible in the program. When it has been called, information
* about memory leaks will be printed on exit.
*/
void MEM_init_memleak_detection(void);

/**
* Use this if we want to call #exit during argument parsing for example,
* without having to free all data.
*/
void MEM_use_memleak_detection(bool enabled);

/**
* When this has been called and memory leaks have been detected, the process will have an exit
* code that indicates failure. This can be used for when checking for memory leaks with automated
* tests.
*/
void MEM_enable_fail_on_memleak(void);

/* Switch allocator to fast mode, with less tracking.
*
* Use in the production code where performance is the priority, and exact details about allocation
* is not. This allocator keeps track of number of allocation and amount of allocated bytes, but it
* does not track of names of allocated blocks.
*
* NOTE: The switch between allocator types can only happen before any allocation did happen. */
void MEM_use_lockfree_allocator(void);

/* Switch allocator to slow fully guarded mode.
*
* Use for debug purposes. This allocator contains lock section around every allocator call, which
* makes it slow. What is gained with this is the ability to have list of allocated blocks (in an
* addition to the tracking of number of allocations and amount of allocated bytes).
*
* NOTE: The switch between allocator types can only happen before any allocation did happen. */
void MEM_use_guarded_allocator(void);

#include <new>
#include <type_traits>
#include <utility>

/**
 * Allocate new memory for and constructs an object of type #T.
 * #MEM_delete should be used to delete the object. Just calling #MEM_freeN is not enough when #T
 * is not a trivial type.
 *
 * Note that when no arguments are passed, C++ will do recursive member-wise value initialization.
 * That is because C++ differentiates between creating an object with `T` (default initialization)
 * and `T()` (value initialization), whereby this function does the latter. Value initialization
 * rules are complex, but for C-style structs, memory will be zero-initialized. So this doesn't
 * match a `malloc()`, but a `calloc()` call in this case. See https://stackoverflow.com/a/4982720.
 */
template<typename T, typename... Args>
inline T* MEM_new(const char* allocation_name, Args &&...args)
{
    void* buffer = MEM_mallocN(sizeof(T), allocation_name);
    return new (buffer) T(std::forward<Args>(args)...);
}

/**
 * Allocates zero-initialized memory for an object of type #T. The constructor of #T is not called,
 * therefor this should only used with trivial types (like all C types).
 * It's valid to call #MEM_freeN on a pointer returned by this, because a destructor call is not
 * necessary, because the type is trivial.
 */
template<typename T> inline T* MEM_cnew(const char* allocation_name)
{
    static_assert(std::is_trivial_v<T>, "For non-trivial types, MEM_new should be used.");
    return static_cast<T*>(MEM_callocN(sizeof(T), allocation_name));
}

/**
 * Allocate memory for an object of type #T and copy construct an object from `other`.
 * Only applicable for a trivial types.
 *
 * This function works around problem of copy-constructing DNA structs which contains deprecated
 * fields: some compilers will generate access deprecated field in implicitly defined copy
 * constructors.
 *
 * This is a better alternative to #MEM_dupallocN.
 */
template<typename T> inline T* MEM_cnew(const char* allocation_name, const T & other)
{
    static_assert(std::is_trivial_v<T>, "For non-trivial types, MEM_new should be used.");
    T* new_object = static_cast<T*>(MEM_mallocN(sizeof(T), allocation_name));
    memcpy(new_object, &other, sizeof(T));
    return new_object;
}

/**
 * Destructs and deallocates an object previously allocated with any `MEM_*` function.
 * Passing in null does nothing.
 */
template<typename T> inline void MEM_delete(const T * ptr)
{
    if (ptr == nullptr) {
        /* Support #ptr being null, because C++ `delete` supports that as well. */
        return;
    }
    /* C++ allows destruction of const objects, so the pointer is allowed to be const. */
    ptr->~T();
    MEM_freeN(const_cast<T*>(ptr));
}

/* Allocation functions (for C++ only). */
#  define MEM_CXX_CLASS_ALLOC_FUNCS(_id) \
   public: \
    void *operator new(size_t num_bytes) \
    { \
      return MEM_mallocN(num_bytes, _id); \
    } \
    void operator delete(void *mem) \
    { \
      if (mem) { \
        MEM_freeN(mem); \
      } \
    } \
    void *operator new[](size_t num_bytes) \
    { \
      return MEM_mallocN(num_bytes, _id "[]"); \
    } \
    void operator delete[](void *mem) \
    { \
      if (mem) { \
        MEM_freeN(mem); \
      } \
    } \
    void *operator new(size_t /*count*/, void *ptr) \
    { \
      return ptr; \
    } \
    /* This is the matching delete operator to the placement-new operator above. Both parameters \
     * will have the same value. Without this, we get the warning C4291 on windows. */ \
    void operator delete(void * /*ptr_to_free*/, void * /*ptr*/) \
    { \
    }

#endif /* __MEM_GUARDEDALLOC_H__ */