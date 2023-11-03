#pragma once

#ifdef _MSC_VER
#  pragma warning(error : 4018) /* signed/unsigned mismatch */
#  pragma warning(error : 4244) /* conversion from 'type1' to 'type2', possible loss of data */
#  pragma warning(error : 4245) /* conversion from 'int' to 'uint' */
#  pragma warning(error : 4267) /* conversion from 'size_t' to 'type', possible loss of data */
#  pragma warning(error : 4305) /* truncation from 'type1' to 'type2' */
#  pragma warning(error : 4389) /* signed/unsigned mismatch */
#endif

#include <cstring>

typedef struct BLI_Buffer {
    void* data;
    const size_t elem_size;
    size_t count, alloc_count;
    int flag;
} BLI_Buffer;
