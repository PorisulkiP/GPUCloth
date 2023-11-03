#include <cstring>

#include "MEM_guardedalloc.cuh"
#include "compiler_attrs.cuh"

#include "ghash.cuh" /* own include */
#include "utildefines.h"

/* keep last */
#include "strict_flags.cuh"

typedef struct BLI_HashMurmur2A {
    uint32_t hash;
    uint32_t tail;
    uint32_t count;
    uint32_t size;
} BLI_HashMurmur2A;


/* Helpers. */
#define MM2A_M 0x5bd1e995

#define MM2A_MIX(h, k) \
  { \
    (k) *= MM2A_M; \
    (k) ^= (k) >> 24; \
    (k) *= MM2A_M; \
    (h) = ((h)*MM2A_M) ^ (k); \
  } \
  (void)0

#define MM2A_MIX_FINALIZE(h) \
  { \
    (h) ^= (h) >> 13; \
    (h) *= MM2A_M; \
    (h) ^= (h) >> 15; \
  } \
  (void)0

static void mm2a_mix_tail(BLI_HashMurmur2A* mm2, const unsigned char** data, size_t* len)
{
    while (*len && ((*len < 4) || mm2->count)) {
        mm2->tail |= (uint32_t)(**data) << (mm2->count * 8);

        mm2->count++;
        (*len)--;
        (*data)++;

        if (mm2->count == 4) {
            MM2A_MIX(mm2->hash, mm2->tail);
            mm2->tail = 0;
            mm2->count = 0;
        }
    }
}

void BLI_hash_mm2a_init(BLI_HashMurmur2A* mm2, uint32_t seed)
{
    mm2->hash = seed;
    mm2->tail = 0;
    mm2->count = 0;
    mm2->size = 0;
}

void BLI_hash_mm2a_add(BLI_HashMurmur2A* mm2, const unsigned char* data, size_t len)
{
    mm2->size += (uint32_t)len;

    mm2a_mix_tail(mm2, &data, &len);

    for (; len >= 4; data += 4, len -= 4) {
        uint32_t k = *(const uint32_t*)data;

        MM2A_MIX(mm2->hash, k);
    }

    mm2a_mix_tail(mm2, &data, &len);
}

void BLI_hash_mm2a_add_int(BLI_HashMurmur2A* mm2, int data)
{
    BLI_hash_mm2a_add(mm2, (const unsigned char*)&data, sizeof(data));
}

uint32_t BLI_hash_mm2a_end(BLI_HashMurmur2A* mm2)
{
    MM2A_MIX(mm2->hash, mm2->tail);
    MM2A_MIX(mm2->hash, mm2->size);

    MM2A_MIX_FINALIZE(mm2->hash);

    return mm2->hash;
}

/* Non-incremental version, quicker for small keys. */
uint32_t BLI_hash_mm2(const unsigned char* data, size_t len, uint32_t seed)
{
    /* Initialize the hash to a 'random' value */
    uint32_t h = uint32_t(seed ^ len);

    /* Mix 4 bytes at a time into the hash */
    for (; len >= 4; data += 4, len -= 4) {
        uint32_t k = *(uint32_t*)data;

        MM2A_MIX(h, k);
    }

    /* Handle the last few bytes of the input array */
    switch (len) {
    case 3:
        h ^= data[2] << 16;
        ATTR_FALLTHROUGH;
    case 2:
        h ^= data[1] << 8;
        ATTR_FALLTHROUGH;
    case 1:
        h ^= data[0];
        h *= MM2A_M;
    }

    /* Do a few final mixes of the hash to ensure the last few bytes are well-incorporated. */
    MM2A_MIX_FINALIZE(h);

    return h;
}



/* -------------------------------------------------------------------- */
/** \name Generic Key Hash & Comparison Functions
 * \{ */

#if 0
/* works but slower */
uint BLI_ghashutil_ptrhash(const void *key)
{
  return (uint)(intptr_t)key;
}
#else
/* Based Python3.7's pointer hashing function. */
uint BLI_ghashutil_ptrhash(const void *key)
{
  size_t y = (size_t)key;
  /* bottom 3 or 4 bits are likely to be 0; rotate y by 4 to avoid
   * excessive hash collisions for dicts and sets */

  /* Note: Unlike Python 'sizeof(uint)' is used instead of 'sizeof(void *)',
   * Otherwise casting to 'uint' ignores the upper bits on 64bit platforms. */
  return (uint)(y >> 4) | ((uint)y << (sizeof(uint[8]) - 4));
}
#endif
bool BLI_ghashutil_ptrcmp(const void *a, const void *b)
{
  return (a != b);
}

uint BLI_ghashutil_uinthash_v4(const uint key[4])
{
  uint hash;
  hash = key[0];
  hash *= 37;
  hash += key[1];
  hash *= 37;
  hash += key[2];
  hash *= 37;
  hash += key[3];
  return hash;
}

uint BLI_ghashutil_uinthash_v4_murmur(const uint key[4])
{
  return BLI_hash_mm2((const unsigned char *)key, sizeof(int[4]) /* sizeof(key) */, 0);
}

bool BLI_ghashutil_uinthash_v4_cmp(const void *a, const void *b)
{
  return (memcmp(a, b, sizeof(uint[4])) != 0);
}

uint BLI_ghashutil_uinthash(uint key)
{
  key += ~(key << 16);
  key ^= (key >> 5);
  key += (key << 3);
  key ^= (key >> 13);
  key += ~(key << 9);
  key ^= (key >> 17);

  return key;
}

uint BLI_ghashutil_inthash_p(const void *ptr)
{
  uintptr_t key = (uintptr_t)ptr;

  key += ~(key << 16);
  key ^= (key >> 5);
  key += (key << 3);
  key ^= (key >> 13);
  key += ~(key << 9);
  key ^= (key >> 17);

  return (uint)(key & 0xffffffff);
}

uint BLI_ghashutil_inthash_p_murmur(const void *ptr)
{
  uintptr_t key = (uintptr_t)ptr;

  return BLI_hash_mm2((const unsigned char *)&key, sizeof(key), 0);
}

uint BLI_ghashutil_inthash_p_simple(const void *ptr)
{
  return POINTER_AS_UINT(ptr);
}

bool BLI_ghashutil_intcmp(const void *a, const void *b)
{
  return (a != b);
}

size_t BLI_ghashutil_combine_hash(size_t hash_a, size_t hash_b)
{
  return hash_a ^ (hash_b + 0x9e3779b9 + (hash_a << 6) + (hash_a >> 2));
}

/**
 * This function implements the widely used "djb" hash apparently posted
 * by Daniel Bernstein to comp.lang.c some time ago.  The 32 bit
 * unsigned hash value starts at 5381 and for each byte 'c' in the
 * string, is updated: ``hash = hash * 33 + c``.  This
 * function uses the signed value of each byte.
 *
 * note: this is the same hash method that glib 2.34.0 uses.
 */
uint BLI_ghashutil_strhash_n(const char *key, size_t n)
{
  const signed char *p;
  uint h = 5381;

  for (p = (const signed char *)key; n-- && *p != '\0'; p++) {
    h = (uint)((h << 5) + h) + (uint)*p;
  }

  return h;
}
__host__ __device__ uint BLI_ghashutil_strhash_p(const void *ptr)
{
  const char *p;
  uint h = 5381;

  for (p = (char*)ptr; *p != '\0'; p++) {
    h = (uint)((h << 5) + h) + (uint)*p;
  }

  return h;
}
uint BLI_ghashutil_strhash_p_murmur(const void *ptr)
{
  const unsigned char *key = (unsigned char*)ptr;

  return BLI_hash_mm2(key, strlen((const char *)key) + 1, 0);
}
bool BLI_ghashutil_strcmp(const void *a, const void *b)
{
  return (a == b) ? false : !STREQ((const char*)a, (const char*)b);
}

GHashPair *BLI_ghashutil_pairalloc(const void *first, const void *second)
{
  GHashPair *pair = (GHashPair*)MEM_lockfree_mallocN(sizeof(GHashPair), "GHashPair");
  pair->first = first;
  pair->second = second;
  return pair;
}

uint BLI_ghashutil_pairhash(const void *ptr)
{
  const GHashPair *pair = (GHashPair*)ptr;
  uint hash = BLI_ghashutil_ptrhash(pair->first);
  return hash ^ BLI_ghashutil_ptrhash(pair->second);
}

bool BLI_ghashutil_paircmp(const void *a, const void *b)
{
  const GHashPair *A = (GHashPair*)a;
  const GHashPair *B = (GHashPair*)b;

  return ((A->first != B->first) || (A->second != B->second));
}

void BLI_ghashutil_pairfree(void *ptr)
{
  MEM_lockfree_freeN(ptr);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Convenience GHash Creation Functions
 * \{ */

GHash *BLI_ghash_ptr_new_ex(const char *info, const uint nentries_reserve)
{
  return BLI_ghash_new_ex(BLI_ghashutil_ptrhash, BLI_ghashutil_ptrcmp, info, nentries_reserve);
}
GHash *BLI_ghash_ptr_new(const char *info)
{
  return BLI_ghash_ptr_new_ex(info, 0);
}

GHash *BLI_ghash_str_new_ex(const char *info, const uint nentries_reserve)
{
  return BLI_ghash_new_ex(BLI_ghashutil_strhash_p, BLI_ghashutil_strcmp, info, nentries_reserve);
}
GHash *BLI_ghash_str_new(const char *info)
{
  return BLI_ghash_str_new_ex(info, 0);
}

GHash *BLI_ghash_int_new_ex(const char *info, const uint nentries_reserve)
{
  return BLI_ghash_new_ex(BLI_ghashutil_inthash_p, BLI_ghashutil_intcmp, info, nentries_reserve);
}
GHash *BLI_ghash_int_new(const char *info)
{
  return BLI_ghash_int_new_ex(info, 0);
}

GHash *BLI_ghash_pair_new_ex(const char *info, const uint nentries_reserve)
{
  return BLI_ghash_new_ex(BLI_ghashutil_pairhash, BLI_ghashutil_paircmp, info, nentries_reserve);
}
GHash *BLI_ghash_pair_new(const char *info)
{
  return BLI_ghash_pair_new_ex(info, 0);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Convenience GSet Creation Functions
 * \{ */

GSet *BLI_gset_ptr_new_ex(const char *info, const uint nentries_reserve)
{
  return BLI_gset_new_ex(BLI_ghashutil_ptrhash, BLI_ghashutil_ptrcmp, info, nentries_reserve);
}
GSet *BLI_gset_ptr_new(const char *info)
{
  return BLI_gset_ptr_new_ex(info, 0);
}

GSet *BLI_gset_str_new_ex(const char *info, const uint nentries_reserve)
{
  return BLI_gset_new_ex(BLI_ghashutil_strhash_p, BLI_ghashutil_strcmp, info, nentries_reserve);
}
GSet *BLI_gset_str_new(const char *info)
{
  return BLI_gset_str_new_ex(info, 0);
}

GSet *BLI_gset_pair_new_ex(const char *info, const uint nentries_reserve)
{
  return BLI_gset_new_ex(BLI_ghashutil_pairhash, BLI_ghashutil_paircmp, info, nentries_reserve);
}
GSet *BLI_gset_pair_new(const char *info)
{
  return BLI_gset_pair_new_ex(info, 0);
}

GSet *BLI_gset_int_new_ex(const char *info, const uint nentries_reserve)
{
  return BLI_gset_new_ex(BLI_ghashutil_inthash_p, BLI_ghashutil_intcmp, info, nentries_reserve);
}
GSet *BLI_gset_int_new(const char *info)
{
  return BLI_gset_int_new_ex(info, 0);
}

/** \} */
