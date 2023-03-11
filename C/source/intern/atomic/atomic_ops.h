#ifndef __ATOMIC_OPS_H__
#define __ATOMIC_OPS_H__

#include "intern/atomic_ops_utils.h"
#include "intern/atomic_ops_ext.h"

/******************************************************************************/
/* Function prototypes. */

//uint64_t atomic_add_and_fetch_uint64(uint64_t *p, uint64_t x);
//uint64_t atomic_sub_and_fetch_uint64(uint64_t *p, uint64_t x);
//uint64_t atomic_fetch_and_add_uint64(uint64_t *p, uint64_t x);
//uint64_t atomic_fetch_and_sub_uint64(uint64_t *p, uint64_t x);
//uint64_t atomic_cas_uint64(uint64_t *v, uint64_t old, uint64_t _new);
//uint64_t atomic_load_uint64(const uint64_t *v);
//void atomic_store_uint64(uint64_t *p, uint64_t v);
//
//int64_t atomic_add_and_fetch_int64(int64_t *p, int64_t x);
//int64_t atomic_sub_and_fetch_int64(int64_t *p, int64_t x);
//int64_t atomic_fetch_and_add_int64(int64_t *p, int64_t x);
//int64_t atomic_fetch_and_sub_int64(int64_t *p, int64_t x);
//int64_t atomic_cas_int64(int64_t *v, int64_t old, int64_t _new);
//int64_t atomic_load_int64(const int64_t *v);
//void atomic_store_int64(int64_t *p, int64_t v);

uint32_t atomic_add_and_fetch_uint32(uint32_t *p, uint32_t x);
uint32_t atomic_sub_and_fetch_uint32(uint32_t *p, uint32_t x);
uint32_t atomic_cas_uint32(uint32_t *v, uint32_t old, uint32_t _new);
uint32_t atomic_load_uint32(const uint32_t *v);
void atomic_store_uint32(uint32_t *p, uint32_t v);

uint32_t atomic_fetch_and_add_uint32(uint32_t *p, uint32_t x);
uint32_t atomic_fetch_and_or_uint32(uint32_t *p, uint32_t x);
uint32_t atomic_fetch_and_and_uint32(uint32_t *p, uint32_t x);

//int32_t atomic_add_and_fetch_int32(int32_t *p, int32_t x);
//int32_t atomic_sub_and_fetch_int32(int32_t *p, int32_t x);
//int32_t atomic_cas_int32(int32_t *v, int32_t old, int32_t _new);
//int32_t atomic_load_int32(const int32_t *v);
//void atomic_store_int32(int32_t *p, int32_t v);
//
//int32_t atomic_fetch_and_add_int32(int32_t *p, int32_t x);
//int32_t atomic_fetch_and_or_int32(int32_t *p, int32_t x);
//int32_t atomic_fetch_and_and_int32(int32_t *p, int32_t x);
//
//int16_t atomic_fetch_and_or_int16(int16_t *p, int16_t b);
//int16_t atomic_fetch_and_and_int16(int16_t *p, int16_t b);

uint8_t atomic_fetch_and_or_uint8(uint8_t *p, uint8_t b);
uint8_t atomic_fetch_and_and_uint8(uint8_t *p, uint8_t b);

int8_t atomic_fetch_and_or_int8(int8_t *p, int8_t b);
int8_t atomic_fetch_and_and_int8(int8_t *p, int8_t b);

char atomic_fetch_and_or_char(char *p, char b);
char atomic_fetch_and_and_char(char *p, char b);

size_t atomic_add_and_fetch_z(size_t *p, size_t x);
size_t atomic_sub_and_fetch_z(size_t *p, size_t x);
size_t atomic_fetch_and_add_z(size_t *p, size_t x);
size_t atomic_fetch_and_sub_z(size_t *p, size_t x);
size_t atomic_cas_z(size_t *v, size_t old, size_t _new);
size_t atomic_load_z(const size_t *v);
void atomic_store_z(size_t *p, size_t v);
/* Uses CAS loop, see warning below. */
size_t atomic_fetch_and_update_max_z(size_t *p, size_t x);

uint atomic_add_and_fetch_u(uint *p, uint x);
uint atomic_sub_and_fetch_u(uint *p, uint x);
uint atomic_fetch_and_add_u(uint *p, uint x);
uint atomic_fetch_and_sub_u(uint *p, uint x);
uint atomic_cas_u(uint *v, uint old, uint _new);

void *atomic_cas_ptr(void **v, void *old, void *_new);
void *atomic_load_ptr(void *const *v);
void atomic_store_ptr(void **p, void *v);

float atomic_cas_float(float *v, float old, float _new);

/* WARNING! Float 'atomics' are really faked ones, those are actually closer to some kind of
 * spinlock-sync'ed operation, which means they are only efficient if collisions are highly
 * unlikely (i.e. if probability of two threads working on the same pointer at the same time is
 * very low). */
float atomic_add_and_fetch_fl(float *p, const float x);

#endif /* __ATOMIC_OPS_H__ */
