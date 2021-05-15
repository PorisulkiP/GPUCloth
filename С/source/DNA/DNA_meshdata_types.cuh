#pragma once

/*
* Создаём цепочку связей
*/
typedef struct LinkNode {
	struct LinkNode* next;
	void* link;
} LinkNode;

/**
* Mesh Vertices triangles
*/
typedef struct MVertTri {
	unsigned int tri[3];
} MVertTri;