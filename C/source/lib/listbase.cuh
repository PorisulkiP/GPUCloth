#pragma once

#ifndef __LISTBASE__
#define __LISTBASE__

#include "pointcache_types.cuh"
#include "cuda_runtime_api.h"

int BLI_findindex(const struct ListBase *listbase, const void *vlink);
int BLI_findstringindex(const struct ListBase *listbase, const char *id, const int offset);

/* find forwards */
void *BLI_findlink(const struct ListBase *listbase, int number);
void *BLI_findstring(const struct ListBase *listbase, const char *id, const int offset);
void *BLI_findstring_ptr(const struct ListBase *listbase, const char *id, const int offset);
void *BLI_findptr(const struct ListBase *listbase, const void *ptr, const int offset);
void *BLI_listbase_bytes_find(const ListBase *listbase, const void *bytes, const size_t bytes_size, const int offset);

/* find backwards */
void *BLI_rfindlink(const struct ListBase *listbase, int number);
void *BLI_rfindstring(const struct ListBase *listbase, const char *id, const int offset);
void *BLI_rfindstring_ptr(const struct ListBase *listbase,
                          const char *id,
                          const int offset);
void *BLI_rfindptr(const struct ListBase *listbase,
                   const void *ptr,
                   const int offset);
void *BLI_listbase_bytes_rfind(const ListBase *listbase,
                               const void *bytes,
                               const size_t bytes_size,
                               const int offset);

__host__ __device__ void BLI_freelistN(struct ListBase *listbase);
void BLI_addtail(struct ListBase *listbase, void *vlink);
void BLI_remlink(struct ListBase *listbase, void *vlink);
bool BLI_remlink_safe(struct ListBase *listbase, void *vlink);
void *BLI_pophead(ListBase *listbase);
void *BLI_poptail(ListBase *listbase);

void BLI_addhead(struct ListBase *listbase, void *vlink);
void BLI_insertlinkbefore(struct ListBase *listbase, void *vnextlink, void *vnewlink);
void BLI_insertlinkafter(struct ListBase *listbase, void *vprevlink, void *vnewlink);
void BLI_insertlinkreplace(ListBase *listbase, void *vreplacelink, void *vnewlink);
void BLI_listbase_sort(struct ListBase *listbase, int (*cmp)(const void *, const void *));
void BLI_listbase_sort_r(ListBase *listbase,
                         int (*cmp)(void *, const void *, const void *),
                         void *thunk) ;
bool BLI_listbase_link_move(ListBase *listbase, void *vlink, int step) ;
bool BLI_listbase_move_index(ListBase *listbase, int from, int to);
void BLI_freelist(struct ListBase* listbase);
int BLI_listbase_count_at_most(const struct ListBase *listbase, const int count_max);
__host__ __device__ int BLI_listbase_count(const struct ListBase *listbase);

void BLI_freelinkN(struct ListBase *listbase, void *vlink);

void BLI_listbase_swaplinks(struct ListBase* listbase, void* vlinka, void* vlinkb);
void BLI_listbases_swaplinks(struct ListBase *listbasea,
                             struct ListBase *listbaseb,
                             void *vlinka,
                             void *vlinkb);

void BLI_movelisttolist(struct ListBase *dst, struct ListBase *src);
void BLI_movelisttolist_reverse(struct ListBase *dst, struct ListBase *src);
__host__ __device__ void BLI_duplicatelist(struct ListBase *dst, const struct ListBase *src);
void BLI_listbase_reverse(struct ListBase *lb);
void BLI_listbase_rotate_first(struct ListBase *lb, void *vlink);
void BLI_listbase_rotate_last(struct ListBase *lb, void *vlink);

/**
 * Utility functions to avoid first/last references inline all over.
 */
__host__ __device__ inline bool BLI_listbase_is_single(const struct ListBase *lb)
{
  return (lb->first && lb->first == lb->last);
}

__host__ __device__ inline bool BLI_listbase_is_empty(const struct ListBase *lb)
{
  return (lb->first == static_cast<void*>(0));
}

__host__ __device__ inline void BLI_listbase_clear(struct ListBase *lb)
{
  lb->first = lb->last = static_cast<void*>(0);
}

/* create a generic list node containing link to provided data */
struct LinkData *BLI_genericNodeN(void *data);

/**
 * Does a full loop on the list, with any value acting as first
 * (handy for cycling items)
 *
 * \code{.c}
 *
 * LISTBASE_CIRCULAR_FORWARD_BEGIN(listbase, item, item_init)
 * {
 *     ...operate on marker...
 * }
 * LISTBASE_CIRCULAR_FORWARD_END (listbase, item, item_init);
 *
 * \endcode
 */
#define LISTBASE_CIRCULAR_FORWARD_BEGIN(lb, lb_iter, lb_init) \
  if ((lb)->first && ((lb_init) || ((lb_init) = (lb)->first))) { \
    (lb_iter) = lb_init; \
    do {
#define LISTBASE_CIRCULAR_FORWARD_END(lb, lb_iter, lb_init) \
  } \
  while (((lb_iter) = (lb_iter)->next ? (lb_iter)->next : (lb)->first), ((lb_iter) != (lb_init))) \
    ; \
  } \
  ((void)0)

#define LISTBASE_CIRCULAR_BACKWARD_BEGIN(lb, lb_iter, lb_init) \
  if ((lb)->last && ((lb_init) || ((lb_init) = (lb)->last))) { \
    (lb_iter) = lb_init; \
    do {
#define LISTBASE_CIRCULAR_BACKWARD_END(lb, lb_iter, lb_init) \
  } \
  while (((lb_iter) = (lb_iter)->prev ? (lb_iter)->prev : (lb)->last), ((lb_iter) != (lb_init))) \
    ; \
  } \
  ((void)0)

#define LISTBASE_FOREACH(type, var, list)   for (type var = (type)((list)->first); (var) != nullptr; (var) = (type)(((Link *)(var))->next))
#define LISTBASE_FOREACH_CUDA(type, var, list, count) for (int i = 0; i < (count); ++i) for(type var = (list)[i]; (var) != nullptr; (var) = (type)(((Link *)(var))->next))

/**
 * A version of #LISTBASE_FOREACH that supports incrementing an index variable at every step.
 * Including this in the macro helps prevent mistakes where "continue" mistakenly skips the
 * incrementation.
 */
#define LISTBASE_FOREACH_INDEX(type, var, list, index_var) \
  for (type var = (((void)((index_var) = 0)), (type)((list)->first)); (var) != NULL; \
       (var) = (type)(((Link *)(var))->next), (index_var)++)

#define LISTBASE_FOREACH_BACKWARD(type, var, list) \
  for (type var = (type)((list)->last); (var) != NULL; (var) = (type)(((Link *)(var))->prev))

/** A version of #LISTBASE_FOREACH that supports removing the item we're looping over. */
#define LISTBASE_FOREACH_MUTABLE(type, var, list) \
  for (type var = (type)((list)->first), *var##_iter_next; \
       (((var) != NULL) ? ((void)(var##_iter_next = (type)(((Link *)(var))->next)), 1) : 0); \
       (var) = var##_iter_next)

/** A version of #LISTBASE_FOREACH_BACKWARD that supports removing the item we're looping over. */
#define LISTBASE_FOREACH_BACKWARD_MUTABLE(type, var, list) \
  for (type var = (type)((list)->last), *var##_iter_prev; \
       (((var) != NULL) ? ((void)(var##_iter_prev = (type)(((Link *)(var))->prev)), 1) : 0); \
       (var) = var##_iter_prev)

#endif