#pragma once
#include <stack>

#include "cloth_settings.cuh"
#include "cuda_runtime.h"
#include "kdopbvh.cuh"
#include "linklist.cuh"

inline LinkNode* copyClothSpringToGPU(const LinkNode* head, const size_t size)
{
	if (!head) return nullptr;

	LinkNode* gpu_head = nullptr, * gpu_current = nullptr, * gpu_prev = nullptr;
	const LinkNode* current = head;

	while (current)
	{
		// Выделяем память для текущей ноды на GPU
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&gpu_current), sizeof(LinkNode)))

		// Если это первая нода, сохраняем её как начало списка на GPU
		if (!gpu_head)
		{
			gpu_head = gpu_current;
		}

		// Копируем текущую ноду на GPU
		gpuErrchk(cudaMemcpy(gpu_current, current, sizeof(LinkNode), cudaMemcpyHostToDevice))

		// Копируем поле 'link'
		void* gpu_link;
		// Здесь нужно знать размер данных, на которые указывает link
		gpuErrchk(cudaMalloc(&gpu_link, size))
		gpuErrchk(cudaMemcpy(gpu_link, current->link, size, cudaMemcpyHostToDevice))

		// Теперь преобразуем LinkNode в ClothSpring и копируем все параметры
		const auto h_tmp = static_cast<ClothSpring*>(current->link);
		const auto d_tmp = static_cast<ClothSpring*>(gpu_link);

		int* gpu_pa, * gpu_pb;
		gpuErrchk(cudaMalloc(&gpu_pa, sizeof(int) * h_tmp->la))
		gpuErrchk(cudaMalloc(&gpu_pb, sizeof(int) * h_tmp->lb))

		gpuErrchk(cudaMemcpy(gpu_pa, h_tmp->pa, sizeof(int) * h_tmp->la, cudaMemcpyHostToDevice))
		gpuErrchk(cudaMemcpy(gpu_pb, h_tmp->pb, sizeof(int) * h_tmp->lb, cudaMemcpyHostToDevice))

		gpuErrchk(cudaMemcpy(&d_tmp->pa, &gpu_pa, sizeof(int*), cudaMemcpyHostToDevice))
		gpuErrchk(cudaMemcpy(&d_tmp->pb, &gpu_pb, sizeof(int*), cudaMemcpyHostToDevice))

		// Обновляем gpu_current->link на GPU
		gpuErrchk(cudaMemcpy(&gpu_current->link, &gpu_link, sizeof(void*), cudaMemcpyHostToDevice))

		// Обновляем указатель 'next' для предыдущей ноды на GPU, если это не первая нода
		if (gpu_prev)
		{
			gpuErrchk(cudaMemcpy(&gpu_prev->next, &gpu_current, sizeof(LinkNode*), cudaMemcpyHostToDevice))
		}

		gpu_prev = gpu_current;
		current = current->next;
	}

	// Обнуляем указатель 'next' для последней ноды
	gpuErrchk(cudaMemset(&gpu_current->next, 0, sizeof(LinkNode*)))

	return gpu_head;
}

// Recursive function to copy each BVHNode
// Рекурсивная функция для копирования BVHNode и его дочерних элементов
inline void recursiveCopyBVHNode(BVHNode* d_node, const BVHNode* h_node, const uchar axiesCount, const uint numNodes)
{
	gpuErrchk(cudaMemcpy(d_node, h_node, sizeof(BVHNode), cudaMemcpyHostToDevice))

	if (h_node->bv)
	{
		float* d_bv;
		gpuErrchk(cudaMalloc(&d_bv, axiesCount * numNodes * sizeof(float)))
		gpuErrchk(cudaMemcpy(d_bv, h_node->bv, axiesCount * numNodes * sizeof(float), cudaMemcpyHostToDevice))
		gpuErrchk(cudaMemcpy(&d_node->bv, &d_bv, sizeof(float*), cudaMemcpyHostToDevice))
	}

	if (h_node->children)
	{
		BVHNode** d_children;
		gpuErrchk(cudaMalloc(&d_children, h_node->totnode * sizeof(BVHNode*)))
		const auto h_children = std::make_unique_for_overwrite<BVHNode*[]>(h_node->totnode);

		for (int i = 0; i < h_node->totnode; ++i)
		{
			BVHNode* d_child;
			gpuErrchk(cudaMalloc(&d_child, sizeof(BVHNode)))
			recursiveCopyBVHNode(d_child, h_node->children[i], axiesCount, numNodes);
			h_children[i] = d_child;
		}

		gpuErrchk(cudaMemcpy(d_children, h_children.get(), h_node->totnode * sizeof(BVHNode*), cudaMemcpyHostToDevice))
		gpuErrchk(cudaMemcpy(&d_node->children, &d_children, sizeof(BVHNode**), cudaMemcpyHostToDevice))
	}
}

inline void copyBVHTreeToDevice(BVHTree* d_tree, const BVHTree* h_tree, const uint maxsize)
{
	constexpr uchar axiesCount = 26;
	// Копирование основной структуры BVHTree на устройство
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_tree), sizeof(BVHTree)))
	gpuErrchk(cudaMemcpy(d_tree, h_tree, sizeof(BVHTree), cudaMemcpyHostToDevice))

	const int numNodes = maxsize + max_ii(1, (maxsize + h_tree->tree_type - 3) / (h_tree->tree_type - 1)) + h_tree->tree_type;

	BVHNode** d_nodes;
	BVHNode* d_nodearray;
	BVHNode** d_nodechild;
	float* d_nodebv;

	// Копируем nodes
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_nodes), h_tree->totbranch * sizeof(BVHNode*)))
	const auto h_nodes = std::make_unique_for_overwrite<BVHNode * []>(h_tree->totbranch);

	for (int i = 0; i < h_tree->totbranch; ++i)
	{
		BVHNode* d_node;
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_node), sizeof(BVHNode)))
		recursiveCopyBVHNode(d_node, h_tree->nodes[i], axiesCount, numNodes);
		h_nodes[i] = d_node;
	}

	gpuErrchk(cudaMemcpy(d_nodes, h_nodes.get(), h_tree->totbranch * sizeof(BVHNode*), cudaMemcpyHostToDevice))
	gpuErrchk(cudaMemcpy(&d_tree->nodes, &d_nodes, sizeof(BVHNode**), cudaMemcpyHostToDevice))

	// Копируем nodearray
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_nodearray), h_tree->totbranch * sizeof(BVHNode*)))
	const auto h_nodearray = std::make_unique_for_overwrite<BVHNode * []>(h_tree->totbranch);

	for (int i = 0; i < h_tree->totbranch; ++i)
	{
		BVHNode* d_node;
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_node), sizeof(BVHNode)))
		recursiveCopyBVHNode(d_node, h_tree->nodes[i], axiesCount, numNodes);
		h_nodearray[i] = d_node;
	}
	gpuErrchk(cudaMemcpy(d_nodearray, h_nodearray.get(), h_tree->totbranch * sizeof(BVHNode*), cudaMemcpyHostToDevice))
	gpuErrchk(cudaMemcpy(&d_tree->nodearray, &d_nodearray, sizeof(BVHNode*), cudaMemcpyHostToDevice))

	// Копируем nodechild
	gpuErrchk(cudaMalloc(&d_nodechild, h_tree->totbranch * sizeof(BVHNode*)))
	const auto h_nodechild = std::make_unique_for_overwrite<BVHNode * []>(h_tree->totbranch);

	for (int i = 0; i < h_tree->totbranch; ++i)
	{
		BVHNode* d_node;
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_node), sizeof(BVHNode)))
		recursiveCopyBVHNode(d_node, h_tree->nodes[i], axiesCount, numNodes);
		h_nodechild[i] = d_node;
	}

	gpuErrchk(cudaMemcpy(d_nodechild, h_nodechild.get(), h_tree->totbranch * sizeof(BVHNode*), cudaMemcpyHostToDevice))
	gpuErrchk(cudaMemcpy(&d_tree->nodechild, &d_nodechild, sizeof(BVHNode**), cudaMemcpyHostToDevice))

	// Копируем nodebv
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_nodebv), h_tree->axis * axiesCount * sizeof(float)))
	gpuErrchk(cudaMemcpy(d_nodebv, h_tree->nodebv, h_tree->axis * numNodes * sizeof(float), cudaMemcpyHostToDevice))
	gpuErrchk(cudaMemcpy(&d_tree->nodebv, &d_nodebv, sizeof(float*), cudaMemcpyHostToDevice))
}

inline void freeBVHTreeFromDevice(BVHTree* d_tree)
{
	if (d_tree == nullptr) return;

	std::stack<BVHNode*> stack;
	stack.push(d_tree->nodes[0]);

	while (!stack.empty())
	{
		BVHNode* d_currNode = stack.top();
		stack.pop();

		if (d_currNode->bv)
		{
			gpuErrchk(cudaFree(d_currNode->bv))
		}

		for (int i = 0; i < d_currNode->totnode; ++i)
		{
			stack.push(d_currNode->children[i]);
		}

		gpuErrchk(cudaFree(d_currNode))
	}

	// Освобождение дополнительных массивов
	gpuErrchk(cudaFree(d_tree->nodes))
	gpuErrchk(cudaFree(d_tree->nodearray))
	gpuErrchk(cudaFree(d_tree->nodechild))
	gpuErrchk(cudaFree(d_tree->nodebv))

	gpuErrchk(cudaFree(d_tree))
}

inline void freeClothSpringOnGPU(LinkNode* gpu_head)
{
	LinkNode* gpu_current = gpu_head;
	LinkNode host_node;

	while (gpu_current)
	{
		// Копируем текущую ноду с GPU на хост
		gpuErrchk(cudaMemcpy(&host_node, gpu_current, sizeof(LinkNode), cudaMemcpyDeviceToHost))

		// Освобождаем память для поля 'link', если оно не nullptr
		if (host_node.link)
		{
			gpuErrchk(cudaFree(host_node.link))
		}

		// Копируем поля pa и pb
		ClothSpring host_spring;
		gpuErrchk(cudaMemcpy(&host_spring, gpu_current, sizeof(ClothSpring), cudaMemcpyDeviceToHost))

		// Освобождаем память для массивов
		if (host_spring.pa) gpuErrchk(cudaFree(host_spring.pa))
		if (host_spring.pb) gpuErrchk(cudaFree(host_spring.pb))

		// Сохраняем указатель на текущую ноду и двигаемся дальше
		LinkNode* to_delete = gpu_current;
		gpu_current = host_node.next;

		// Освобождаем память для текущей ноды
		gpuErrchk(cudaFree(to_delete))
	}
}