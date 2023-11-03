#pragma once

#include "cloth_settings.cuh"
#include "cuda_runtime.h"
#include "kdopbvh.cuh"
#include "linklist.cuh"
#include "kernel_config.h"
#include <device_launch_parameters.h>


namespace cudaMemUtils{
	inline LinkNode* copyClothSpringToGPU(const LinkNode* head, const size_t size)
	{
		if (!head) return nullptr;

		LinkNode* gpu_head = nullptr, * gpu_current = nullptr, * gpu_prev = nullptr;
		const LinkNode* current = head;

		while (current)
		{
			// �������� ������ ��� ������� ���� �� GPU
			gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&gpu_current), sizeof(LinkNode)))

			// ���� ��� ������ ����, ��������� � ��� ������ ������ �� GPU
			if (!gpu_head)
			{
				gpu_head = gpu_current;
			}

			// �������� ������� ���� �� GPU
			gpuErrchk(cudaMemcpy(gpu_current, current, sizeof(LinkNode), cudaMemcpyHostToDevice))

			// �������� ���� 'link'
			void* gpu_link;
			// ����� ����� ����� ������ ������, �� ������� ��������� link
			gpuErrchk(cudaMalloc(&gpu_link, size))
			gpuErrchk(cudaMemcpy(gpu_link, current->link, size, cudaMemcpyHostToDevice))

			// ������ ����������� LinkNode � ClothSpring � �������� ��� ���������
			const auto h_tmp = static_cast<ClothSpring*>(current->link);
			const auto d_tmp = static_cast<ClothSpring*>(gpu_link);

			int* gpu_pa, * gpu_pb;
			gpuErrchk(cudaMalloc(&gpu_pa, sizeof(int) * h_tmp->la))
			gpuErrchk(cudaMalloc(&gpu_pb, sizeof(int) * h_tmp->lb))

			gpuErrchk(cudaMemcpy(gpu_pa, h_tmp->pa, sizeof(int) * h_tmp->la, cudaMemcpyHostToDevice))
			gpuErrchk(cudaMemcpy(gpu_pb, h_tmp->pb, sizeof(int) * h_tmp->lb, cudaMemcpyHostToDevice))

			gpuErrchk(cudaMemcpy(&d_tmp->pa, &gpu_pa, sizeof(int*), cudaMemcpyHostToDevice))
			gpuErrchk(cudaMemcpy(&d_tmp->pb, &gpu_pb, sizeof(int*), cudaMemcpyHostToDevice))

			// ��������� gpu_current->link �� GPU
			gpuErrchk(cudaMemcpy(&gpu_current->link, &gpu_link, sizeof(void*), cudaMemcpyHostToDevice))

			// ��������� ��������� 'next' ��� ���������� ���� �� GPU, ���� ��� �� ������ ����
			if (gpu_prev)
			{
				gpuErrchk(cudaMemcpy(&gpu_prev->next, &gpu_current, sizeof(LinkNode*), cudaMemcpyHostToDevice))
			}

			gpu_prev = gpu_current;
			current = current->next;
		}

		// �������� ��������� 'next' ��� ��������� ����
		gpuErrchk(cudaMemset(&gpu_current->next, 0, sizeof(LinkNode*)))

		return gpu_head;
	}

	// ReSharper disable once CppNonInlineFunctionDefinitionInHeaderFile
	__global__ void updateNodePointers(BVHNode* d_nodes, float* d_nodebv, BVHNode** d_nodechild, const uint numNodes, const uchar axis, const uchar tree_type)
	{
		const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < numNodes) 
		{
			d_nodes[idx].bv = &d_nodebv[idx * axis];
			d_nodes[idx].children = &d_nodechild[idx * tree_type];
		}
	}

	inline void copyBVHNodeToDevice(BVHNode* d_node, const BVHNode* h_node, BVHNode*& d_nodearray, float*& d_nodebv, BVHNode**& d_nodechild) {
		// ����������� ������ ����
		BVHNode node = *h_node;
		node.bv = d_nodebv;
		node.children = d_nodechild;

		// ����������� ��������������� ������
		cudaMemcpy(d_nodebv, h_node->bv, sizeof(float) * 2 * 13, cudaMemcpyHostToDevice);
		d_nodebv += 2 * 13;

		// ����������� � ���������� ���������� �� �������� ��������
		for (int i = 0; i < h_node->totnode; ++i) {
			cudaMemcpy(&d_nodechild[i], &d_nodearray, sizeof(BVHNode*), cudaMemcpyHostToDevice);
			copyBVHNodeToDevice(d_nodearray, h_node->children[i], d_nodearray, d_nodebv, d_nodechild);
			d_nodearray++;
		}
		d_nodechild += h_node->totnode;

		// ����������� ����
		cudaMemcpy(d_node, &node, sizeof(BVHNode), cudaMemcpyHostToDevice);
	}

	inline BVHTree* copyBVHTreeToDevice(const BVHTree* h_tree, const int maxsize)
	{
		BVHTree* d_tree;
		BVHNode* d_nodes;
		BVHNode* d_nodearray;
		float* d_nodebv;
		BVHNode** d_nodechild;

		const int numNodes = maxsize + maxsize + max_ii(1, (maxsize + h_tree->tree_type - 3) / (h_tree->tree_type - 1)) + h_tree->tree_type;

		// Allocate device memory
		gpuErrchk(cudaMalloc(&d_tree, sizeof(BVHTree)))
		gpuErrchk(cudaMalloc(&d_nodes, numNodes * sizeof(BVHNode)))
		gpuErrchk(cudaMalloc(&d_nodearray, numNodes * sizeof(BVHNode)))
		gpuErrchk(cudaMalloc(&d_nodebv, h_tree->axis * numNodes * sizeof(float)))
		gpuErrchk(cudaMalloc(&d_nodechild, h_tree->tree_type * numNodes * sizeof(BVHNode*)))

		// ����������� ������
		float* d_nodebv_current = d_nodebv;
		BVHNode** d_nodechild_current = d_nodechild;
		copyBVHNodeToDevice(d_nodearray, h_tree->nodearray, d_nodearray, d_nodebv_current, d_nodechild_current);

		for (int i = 0; i < h_tree->totleaf; i++) 
		{
			gpuErrchk(cudaMemcpy(&d_nodes[i], &d_nodearray[i], sizeof(BVHNode*), cudaMemcpyHostToDevice))
		}
		for (int i = 0; i < h_tree->totbranch; i++) 
		{
			gpuErrchk(cudaMemcpy(&d_nodes[h_tree->totleaf + i], &d_nodearray[h_tree->totleaf + i], sizeof(BVHNode*), cudaMemcpyHostToDevice))
		}

		// ���������� ���������� � BVHTree
		gpuErrchk(cudaMemcpy(&d_tree->nodes, &d_nodes, sizeof(BVHNode**), cudaMemcpyHostToDevice))
		gpuErrchk(cudaMemcpy(&d_tree->nodearray, &d_nodearray, sizeof(BVHNode*), cudaMemcpyHostToDevice))
		gpuErrchk(cudaMemcpy(&d_tree->nodechild, &d_nodechild, sizeof(BVHNode**), cudaMemcpyHostToDevice))
		gpuErrchk(cudaMemcpy(&d_tree->nodebv, &d_nodebv, sizeof(float*), cudaMemcpyHostToDevice))

		return d_tree;
	}

	inline void freeBVHTreeFromDevice(BVHTree* d_tree)
	{
		if (d_tree == nullptr) return;

		// ������� ��������� ��������� BVHTree �� �����
		BVHTree h_tree;

		// �������� ������ ������ � ���������� �� ����
		gpuErrchk(cudaMemcpy(&h_tree, d_tree, sizeof(BVHTree), cudaMemcpyDeviceToHost))

		// ����������� ������, ���������� ��� �����, �������������� ������ � �������� �����
		cudaFree(h_tree.nodearray);
		cudaFree(h_tree.nodebv);
		cudaFree(h_tree.nodechild);

		// ����������� ������, ���������� ��� ������� �����
		cudaFree(h_tree.nodes);

		// ����������� ������, ���������� ��� ������ ������
		cudaFree(d_tree);
	}

	inline void freeClothSpringOnGPU(LinkNode* gpu_head)
	{
	    LinkNode* gpu_current = gpu_head;
	    LinkNode host_node;

	    while (gpu_current)
	    {
	        // �������� ������� ���� � GPU �� ����
	        gpuErrchk(cudaMemcpy(&host_node, gpu_current, sizeof(LinkNode), cudaMemcpyDeviceToHost))

	        // ����������� ������ ��� ���� 'link', ���� ��� �� nullptr
	        if (host_node.link)
	        {
	            ClothSpring host_spring;
	            gpuErrchk(cudaMemcpy(&host_spring, host_node.link, sizeof(ClothSpring), cudaMemcpyDeviceToHost))

	            // ����������� ������ ��� ��������
	            if (host_spring.pa) gpuErrchk(cudaFree(host_spring.pa))
	            if (host_spring.pb) gpuErrchk(cudaFree(host_spring.pb))

	            gpuErrchk(cudaFree(host_node.link))
	        }

	        // ��������� ��������� �� ������� ���� � ��������� ������
	        LinkNode* to_delete = gpu_current;
	        gpu_current = host_node.next;

	        // ����������� ������ ��� ������� ����
	        gpuErrchk(cudaFree(to_delete))
	    }
	}
}