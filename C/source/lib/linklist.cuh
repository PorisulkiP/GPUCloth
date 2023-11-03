#pragma once

#include "compiler_attrs.cuh"
#include "MEM_guardedalloc.cuh"

struct BLI_mempool;
struct MemArena;

typedef void (*LinkNodeFreeFP)(void *link);
typedef void (*LinkNodeApplyFP)(void *link, void *userdata);

typedef struct LinkNode {
  LinkNode *next;
  void *link;
} LinkNode;

/**
 * Use for append (single linked list, storing the last element).
 *
 * \note list manipulation functions don't operate on this struct.
 * This is only to be used while appending.
 */
typedef struct LinkNodePair {
  LinkNode *list, *last_node;
} LinkNodePair;

int BLI_linklist_count(const LinkNode *list) ATTR_WARN_UNUSED_RESULT;
int BLI_linklist_index(const LinkNode *list, void *ptr) ATTR_WARN_UNUSED_RESULT;

LinkNode *BLI_linklist_find(LinkNode *list, int index) ATTR_WARN_UNUSED_RESULT;
LinkNode *BLI_linklist_find_last(LinkNode *list) ATTR_WARN_UNUSED_RESULT;

void BLI_linklist_reverse(LinkNode **listp) ATTR_NONNULL(1);

void BLI_linklist_move_item(LinkNode **listp, int curr_index, int new_index) ATTR_NONNULL(1);

void BLI_linklist_prepend_nlink(LinkNode **listp, void *ptr, LinkNode *nlink) ATTR_NONNULL(1, 3);
void BLI_linklist_prepend(LinkNode **listp, void *ptr) ATTR_NONNULL(1);
void BLI_linklist_prepend_arena(LinkNode **listp, void *ptr, struct MemArena *ma)
    ATTR_NONNULL(1, 3);
void BLI_linklist_prepend_pool(LinkNode **listp, void *ptr, struct BLI_mempool *mempool)
    ATTR_NONNULL(1, 3);

/* use LinkNodePair to avoid full search */
void BLI_linklist_append_nlink(LinkNodePair *list_pair, void *ptr, LinkNode *nlink)
    ATTR_NONNULL(1, 3);
void BLI_linklist_append(LinkNodePair *list_pair, void *ptr) ATTR_NONNULL(1);
void BLI_linklist_append_arena(LinkNodePair *list_pair, void *ptr, struct MemArena *ma)
    ATTR_NONNULL(1, 3);
void BLI_linklist_append_pool(LinkNodePair *list_pair, void *ptr, struct BLI_mempool *mempool)
    ATTR_NONNULL(1, 3);

void *BLI_linklist_pop(LinkNode **listp) ATTR_WARN_UNUSED_RESULT ATTR_NONNULL(1);
void *BLI_linklist_pop_pool(LinkNode **listp, struct BLI_mempool *mempool) ATTR_WARN_UNUSED_RESULT
    ATTR_NONNULL(1, 2);
void BLI_linklist_insert_after(LinkNode **listp, void *ptr) ATTR_NONNULL(1);

void BLI_linklist_free(LinkNode *list, LinkNodeFreeFP freefunc);
void BLI_linklist_freeN(LinkNode *list);
void BLI_linklist_free_pool(LinkNode *list, LinkNodeFreeFP freefunc, struct BLI_mempool *mempool);
void BLI_linklist_apply(LinkNode *list, LinkNodeApplyFP applyfunc, void *userdata);
LinkNode *BLI_linklist_sort(LinkNode *list,
                            int (*cmp)(const void *, const void *)) ATTR_WARN_UNUSED_RESULT
    ATTR_NONNULL(2);
LinkNode *BLI_linklist_sort_r(LinkNode *list,
                              int (*cmp)(void *, const void *, const void *),
                              void *thunk) ATTR_WARN_UNUSED_RESULT ATTR_NONNULL(2);

#define BLI_linklist_prepend_alloca(listp, ptr) \
  BLI_linklist_prepend_nlink((LinkNode **)listp, ptr, (LinkNode *)alloca(sizeof(LinkNode)))
#define BLI_linklist_append_alloca(list_pair, ptr) \
  BLI_linklist_append_nlink((LinkNode **)list_pair, ptr, (LinkNode *)alloca(sizeof(LinkNode)))

inline LinkNode* copyListToGPU(const LinkNode* head, const size_t size)
{
    if (!head) return nullptr;

    LinkNode* gpu_head = nullptr, * gpu_current = nullptr, * gpu_prev = nullptr;
    const LinkNode* current = head;

    while (current) {
        // Выделяем память для текущей ноды на GPU
        gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&gpu_current), sizeof(LinkNode)))

        // Если это первая нода, сохраняем её как начало списка на GPU
        if (gpu_head == nullptr) {
            gpu_head = gpu_current;
        }

        // Копируем текущую ноду на GPU
        gpuErrchk(cudaMemcpy(gpu_current, current, sizeof(LinkNode), cudaMemcpyHostToDevice))

        // Копируем поле 'link', если оно не nullptr
        if (current->link) {
            void* gpu_link;
            // Здесь нужно знать размер данных, на которые указывает link
            gpuErrchk(cudaMalloc(&gpu_link, size))
            gpuErrchk(cudaMemcpy(gpu_link, current->link, size, cudaMemcpyHostToDevice))

            // Обновляем gpu_current->link на GPU
            gpuErrchk(cudaMemcpy(&(gpu_current->link), &gpu_link, sizeof(void*), cudaMemcpyHostToDevice))
        }

        // Обновляем указатель 'next' для предыдущей ноды на GPU, если это не первая нода
        if (gpu_prev) {
            gpuErrchk(cudaMemcpy(&(gpu_prev->next), &gpu_current, sizeof(LinkNode*), cudaMemcpyHostToDevice))
        }

        gpu_prev = gpu_current;
        current = current->next;
    }

    // Обнуляем указатель 'next' для последней ноды
    gpuErrchk(cudaMemset(&(gpu_current->next), 0, sizeof(LinkNode*)))

    return gpu_head;
}

inline void freeListFromGPU(LinkNode* gpu_head)
{
    LinkNode* gpu_current = gpu_head, * gpu_temp;

    while (gpu_current) {
        // Если поле 'link' не nullptr, освобождаем его
        void* gpu_link;
        gpuErrchk(cudaMemcpy(&gpu_link, &(gpu_current->link), sizeof(void*), cudaMemcpyDeviceToHost))

        if (gpu_link) {
            gpuErrchk(cudaFree(gpu_link));
        }

        // Сохраняем указатель на следующую ноду
        gpuErrchk(cudaMemcpy(&gpu_temp, &(gpu_current->next), sizeof(LinkNode*), cudaMemcpyDeviceToHost))

        // Освобождаем текущую ноду
        gpuErrchk(cudaFree(gpu_current))

        gpu_current = gpu_temp;
    }
}
