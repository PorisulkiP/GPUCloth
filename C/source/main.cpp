#include "main.h"

#include <rand.hh>

#include "CUDA_mem_utils.cuh"
#include "rand.h"
#include "depsgraph.cuh"
#include "edgehash.h"
#include "implicit.cuh"
#include "linklist.cuh"
#include "MEM_guardedalloc.cuh"
#include "meshdata_types.cuh"
#include "MOD_util.h"
#include "object_force_types.cuh"
#include "PIL_time_utildefines.cuh"
#include "../source/kernel/cloth.h"

using namespace cudaMemUtils;

#ifdef _WIN32
int main()
{
	Scene tmpScene;
	ToolSettings tmpToolSettings;
	memset(&tmpScene, 1, sizeof(Scene));
	memset(&tmpToolSettings, 1, sizeof(ToolSettings));
	tmpScene.physics_settings = {{0.f, 0.f, -9.8f}, 1};
	tmpScene.toolsettings = &tmpToolSettings;

	const Depsgraph depsgraph{nullptr, &tmpScene, nullptr, DAG_EVAL_VIEWPORT};

    Object ob;
	constexpr float imat[4][4] = {
			{1, 1, 1, 1},
			{1, 1, 1, 1},
			{1, 1, 1, 1},
			{1, 1, 1, 1}
		};
	std::memcpy(ob.imat, imat, sizeof(imat));
	std::memcpy(ob.obmat, imat, sizeof(imat));

	PartDeflect pd;
	memset(&pd, 1, sizeof(PartDeflect));
	ob.pd = &pd;

    Mesh mesh;

	// Allocate memory for vertices, edges, polys and loops
	mesh.mvert = static_cast<MVert*>(malloc(8 * sizeof(MVert)));
	mesh.medge = static_cast<MEdge*>(malloc(12 * sizeof(MEdge)));
	mesh.mpoly = static_cast<MPoly*>(malloc(6 * sizeof(MPoly)));
	mesh.mloop = static_cast<MLoop*>(malloc(24 * sizeof(MLoop)));

	// Set the total counts
	mesh.totvert = 8;
	mesh.totedge = 12;
	mesh.totpoly = 6;
	mesh.totloop = 24;

	// Define the vertices (coordinates)
	constexpr float coords[8][3] = {
		{1.0f, 1.0f, 1.0f},
		{1.0f, 1.0f, -1.0f},
		{1.0f, -1.0f, 1.0f},
		{1.0f, -1.0f, -1.0f},
		{-1.0f, 1.0f, 1.0f},
		{-1.0f, 1.0f, -1.0f},
		{-1.0f, -1.0f, 1.0f},
		{-1.0f, -1.0f, -1.0f}
	};
	for (int i = 0; i < mesh.totvert; ++i) {
		mesh.mvert[i].co[0] = coords[i][0];
		mesh.mvert[i].co[1] = coords[i][1];
		mesh.mvert[i].co[2] = coords[i][2];
	}

	// Define the edges (vertex indices)
	constexpr int edges[12][2] = {
		{5, 7}, {1, 5}, {0, 1}, {7, 6},
		{2, 3}, {4, 5}, {2, 6}, {0, 2},
		{7, 3}, {6, 4}, {4, 2}, {3, 1}
	};
	for (int i = 0; i < mesh.totedge; ++i) {
		mesh.medge[i].v1 = edges[i][0];
		mesh.medge[i].v2 = edges[i][1];
		mesh.medge[i].crease = 0;
		mesh.medge[i].bweight = 0;
	}

	// Define the polygons (start loop index and loop total)
	constexpr int polys[6][2] = {
		{0, 4}, {4, 4}, {8, 4}, {12, 4}, {16, 4}, {20, 4}
	};
	for (int i = 0; i < mesh.totpoly; ++i) {
		mesh.mpoly[i].loopstart = polys[i][0];
		mesh.mpoly[i].totloop = polys[i][1];
	}

	// Define the loops (vertex and edge indices)
	constexpr int loops[24][2] = {
		{0, 10}, {4, 9}, {6, 6}, {2, 7}, // Bottom face
		{3, 4}, {2, 6}, {6, 3}, {7, 8}, // Top face
		{7, 3}, {6, 9}, {4, 5}, {5, 0}, // Front face
		{5, 1}, {1, 11}, {3, 8}, {7, 0}, // Right face
		{1, 2}, {0, 7}, {2, 4}, {3, 11}, // Back face
		{5, 5}, {4, 10}, {0, 2}, {1, 1} // Left face
	};
	for (int i = 0; i < mesh.totloop; ++i) {
		mesh.mloop[i].v = loops[i][0];
		mesh.mloop[i].e = loops[i][1];
	}

	mesh.runtime = nullptr;

	auto clmd = static_cast<ClothModifierData*>(MEM_lockfree_callocN(sizeof(ClothModifierData), ""));

	MEMCPY_STRUCT_AFTER(clmd, DNA_struct_default_get(ClothModifierData), modifier);
	clmd->sim_parms = DNA_struct_default_alloc(ClothSimSettings);
	clmd->coll_parms = DNA_struct_default_alloc(ClothCollSettings);

	BuildClothSprings(clmd, &mesh);

	clmd->clothObject->implicit = SIM_mass_spring_solver_create(mesh.totvert, clmd->clothObject->numsprings);

    SIM_solver(&depsgraph, &ob, clmd, &mesh);

	MEM_lockfree_freeN(clmd);
	free(mesh.mvert);
	free(mesh.medge);
	free(mesh.mpoly);
	free(mesh.mloop);
}
#endif

struct RNG {
	blender::RandomNumberGenerator rng;

	MEM_CXX_CLASS_ALLOC_FUNCS("RNG")
};

// Сюда загружаются данные о ткани (Object, ClothModifierData)
int SIM_LoadClothOBJs()
{
    return 0;
}

// Сюда загружаются данные о объектах столкновений
// Depsgraph*, Object*, ClothModifierData*, float step, float dt
int SIM_SetCollisionOBJs() 
{
    //cloth_solve_collisions();
    return 0;
}

/* initialize simulation data if it didn't exist already */
bool BuildClothSprings(ClothModifierData* clmd, Mesh* mesh)
{
	if (!clmd->clothObject) 
	{
		if (!cloth_from_object(clmd, mesh))
		{
			return false;
		}

		if (!clmd->clothObject) 
		{
			return false;
		}

		SIM_cloth_solver_set_positions(clmd);

		const ClothSimSettings* parms = clmd->sim_parms;
		if (parms->flags & CLOTH_SIMSETTINGS_FLAG_PRESSURE && !(parms->flags & CLOTH_SIMSETTINGS_FLAG_PRESSURE_VOL)) 
		{
			SIM_cloth_solver_set_volume(clmd);
		}

		clmd->clothObject->last_frame = MINFRAME - 1;
		clmd->sim_parms->dt = 1.0f / static_cast<float>(clmd->sim_parms->stepsPerFrame);
	}

	return true;
}

// Запуск вычислений
// Depsgraph надо заменить на сцену
bool SIM_solver(const Depsgraph* depsgraph, const Object* ob, const ClothModifierData* clmd, const Mesh* mesh)
{
	/*
	 *		Mesh
	 */
	Mesh* d_mesh;
	MVert* d_mvert;
	MEdge* d_medge;
	MPoly* d_mpoly;
	MLoop* d_mloop;

	// Выделение памяти для структуры Mesh
	gpuErrchk(cudaMalloc(&d_mesh, sizeof(Mesh)))
		gpuErrchk(cudaMalloc(&d_mvert, sizeof(MVert) * mesh->totvert))
		gpuErrchk(cudaMalloc(&d_medge, sizeof(MEdge) * mesh->totedge))
		gpuErrchk(cudaMalloc(&d_mpoly, sizeof(MPoly) * mesh->totpoly))
		gpuErrchk(cudaMalloc(&d_mloop, sizeof(MLoop) * mesh->totloop))

		// Копирование данных в CUDA
		gpuErrchk(cudaMemcpy(d_mloop, mesh->mloop, sizeof(MLoop) * mesh->totloop, cudaMemcpyHostToDevice))
		gpuErrchk(cudaMemcpy(d_mpoly, mesh->mpoly, sizeof(MPoly) * mesh->totpoly, cudaMemcpyHostToDevice))
		gpuErrchk(cudaMemcpy(d_medge, mesh->medge, sizeof(MEdge) * mesh->totedge, cudaMemcpyHostToDevice))
		gpuErrchk(cudaMemcpy(d_mvert, mesh->mvert, sizeof(MVert) * mesh->totvert, cudaMemcpyHostToDevice))

	// Обновление указателей в d_mesh
	Mesh tempMesh = *mesh;
	tempMesh.mvert = d_mvert;
	tempMesh.medge = d_medge;
	tempMesh.mpoly = d_mpoly;
	tempMesh.mloop = d_mloop;

	// копирование примитивных переменных
	tempMesh.totvert = mesh->totvert;
	tempMesh.totedge = mesh->totedge;
	tempMesh.totpoly = mesh->totpoly;
	tempMesh.totloop = mesh->totloop;

	gpuErrchk(cudaMemcpy(d_mesh, &tempMesh, sizeof(Mesh), cudaMemcpyHostToDevice))

	/*
	 *		Depsgraph
	 */
	Depsgraph* d_depsgraph;
		Scene* d_depsgraph_scene;
			ToolSettings* d_toolsettings;
		Scene* d_depsgraph_scene_cow;
			ToolSettings* d_cow_toolsettings;
	gpuErrchk(cudaMalloc(&d_depsgraph, sizeof(Depsgraph)))
		gpuErrchk(cudaMalloc(&d_depsgraph_scene, sizeof(Scene)))
		gpuErrchk(cudaMalloc(&d_toolsettings, sizeof(ToolSettings)))
		gpuErrchk(cudaMalloc(&d_depsgraph_scene_cow, sizeof(Scene)))
		gpuErrchk(cudaMalloc(&d_cow_toolsettings, sizeof(ToolSettings)))

		gpuErrchk(cudaMemcpy(d_depsgraph_scene, depsgraph->scene, sizeof(Scene), cudaMemcpyHostToDevice))
		gpuErrchk(cudaMemcpy(d_toolsettings, depsgraph->scene->toolsettings, sizeof(ToolSettings), cudaMemcpyHostToDevice))
		gpuErrchk(cudaMemcpy(d_depsgraph_scene_cow, depsgraph->scene, sizeof(Scene), cudaMemcpyHostToDevice))
		gpuErrchk(cudaMemcpy(d_cow_toolsettings, depsgraph->scene->toolsettings, sizeof(ToolSettings), cudaMemcpyHostToDevice))

	// Обновление указателей в temp_depsgraph
	Depsgraph temp_depsgraph = *depsgraph;
	Scene temp_scene = *depsgraph->scene;
	ToolSettings temp_toolsettings = *depsgraph->scene->toolsettings;
	Scene temp_scene_cow = *depsgraph->scene;
	ToolSettings temp_cow_toolsettings = *depsgraph->scene->toolsettings;

	// Обновите временные объекты на хосте, чтобы они указывали на выделенные области памяти на устройстве
	temp_scene.toolsettings = d_toolsettings;
	temp_scene_cow.toolsettings = d_cow_toolsettings;

	// Скопируйте обновленные временные объекты обратно на устройство
	gpuErrchk(cudaMemcpy(d_toolsettings, &temp_toolsettings, sizeof(ToolSettings), cudaMemcpyHostToDevice))
	gpuErrchk(cudaMemcpy(d_cow_toolsettings, &temp_cow_toolsettings, sizeof(ToolSettings), cudaMemcpyHostToDevice))
	gpuErrchk(cudaMemcpy(d_depsgraph_scene, &temp_scene, sizeof(Scene), cudaMemcpyHostToDevice))
	gpuErrchk(cudaMemcpy(d_depsgraph_scene_cow, &temp_scene_cow, sizeof(Scene), cudaMemcpyHostToDevice))

	// Теперь обновите основной объект Depsgraph на хосте, чтобы он указывал на обновленные объекты Scene на устройстве
	temp_depsgraph.scene = d_depsgraph_scene;
	temp_depsgraph.scene_cow = d_depsgraph_scene_cow;

	// И наконец, скопируйте обновленный объект Depsgraph обратно на устройство
	gpuErrchk(cudaMemcpy(d_depsgraph, &temp_depsgraph, sizeof(Depsgraph), cudaMemcpyHostToDevice))

	/*
	 *		Object
	 */
	Object* d_ob;
		void* d_ob_data;
		PartDeflect* d_ob_PartDeflect;
			RNG* d_ob_rng;
	gpuErrchk(cudaMalloc(&d_ob, sizeof(Object)))
		gpuErrchk(cudaMalloc(&d_ob_data, sizeof(void*)))
		gpuErrchk(cudaMalloc(&d_ob_PartDeflect, sizeof(PartDeflect)))
		gpuErrchk(cudaMalloc(&d_ob_rng, sizeof(RNG)))

		gpuErrchk(cudaMemcpy(d_ob_PartDeflect, ob->pd, sizeof(PartDeflect), cudaMemcpyHostToDevice))

	// Обновление указателей в d_ob
	Object tempobject = *ob;
	PartDeflect temp_pd = *(ob->pd);
	temp_pd.rng = d_ob_rng;
	gpuErrchk(cudaMemcpy(d_ob_PartDeflect, &temp_pd, sizeof(PartDeflect), cudaMemcpyHostToDevice))

	tempobject.data = d_ob_data;
	tempobject.pd = d_ob_PartDeflect;

	gpuErrchk(cudaMemcpy(d_ob, &tempobject, sizeof(Object), cudaMemcpyHostToDevice))

	/*
	 *		ClothModifierData
	 */
	ClothModifierData* d_clmd;
	    ModifierData* d_modifier;
			ModifierData* d_nextModifier;
            ModifierData* d_prevModifier;
	    Cloth*  d_cloth;
			ClothVertex* d_clothVertex;
            LinkNode* d_springs;
            BVHTree* d_bvhtree;
			BVHTree* d_bvhselftree;
            MVertTri* d_tri;
            Implicit_Data* d_implicit;
            EdgeSet* d_edgeset;
				Edge* d_entries;
                int32_t* d_map;
            MEdge* d_edges;
            EdgeSet* d_sew_edge_graph;
				Edge* d_sew_entries;
                int32_t* d_sew_map;
        ClothSimSettings* d_sim_parms;
			EffectorWeights* d_effector_weights;
				//Collection* d_group;
        ClothCollSettings* d_coll_parms;
			LinkNode* d_collision_list;
	            LinkNode* d_collision_list_next;
	            void* d_collision_list_link;
        PointCache* d_point_cache; //  clmd->point_cache->totpoint
			PointCache* d_next_point_cache; //  clmd->point_cache->totpoint
            PointCache* d_prev_point_cache; //  clmd->point_cache->totpoint
		ListBase* d_ptcaches; //  clmd->point_cache->totpoint
	        void* d_ptcaches_first;//  clmd->point_cache->totpoint
	        void* d_ptcaches_last;//  clmd->point_cache->totpoint
        ClothSolverResult* d_solver_result;

		gpuErrchk(cudaMalloc(&d_clmd, sizeof(ClothModifierData)))
			gpuErrchk(cudaMalloc(&d_modifier, sizeof(ModifierData)))
			gpuErrchk(cudaMalloc(&d_nextModifier, sizeof(ModifierData)))
			gpuErrchk(cudaMalloc(&d_prevModifier, sizeof(ModifierData)))
			gpuErrchk(cudaMalloc(&d_cloth, sizeof(Cloth)))
			gpuErrchk(cudaMalloc(&d_clothVertex, sizeof(ClothVertex)* clmd->clothObject->mvert_num))
			gpuErrchk(cudaMalloc(&d_tri, sizeof(MVertTri)* clmd->clothObject->mvert_num))
			gpuErrchk(cudaMalloc(&d_edgeset, sizeof(EdgeSet)))
			gpuErrchk(cudaMalloc(&d_entries, sizeof(Edge)))
			gpuErrchk(cudaMalloc(&d_map, sizeof(int32_t)))
			gpuErrchk(cudaMalloc(&d_edges, sizeof(MEdge)))
			gpuErrchk(cudaMalloc(&d_sew_edge_graph, sizeof(EdgeSet)))
			gpuErrchk(cudaMalloc(&d_sew_entries, sizeof(Edge)))
			gpuErrchk(cudaMalloc(&d_sew_map, sizeof(int32_t)))
			gpuErrchk(cudaMalloc(&d_sim_parms, sizeof(ClothSimSettings)))
			gpuErrchk(cudaMalloc(&d_effector_weights, sizeof(EffectorWeights)))
			//gpuErrchk(cudaMalloc(&d_group, sizeof(Collection)))
			gpuErrchk(cudaMalloc(&d_coll_parms, sizeof(ClothCollSettings)))
			gpuErrchk(cudaMalloc(&d_collision_list, sizeof(LinkNode)))
			gpuErrchk(cudaMalloc(&(d_collision_list_next), sizeof(LinkNode)))
			gpuErrchk(cudaMalloc(&(d_collision_list_link), sizeof(void*)))
			gpuErrchk(cudaMalloc(&d_point_cache, sizeof(PointCache))) // * clmd->point_cache->totpoint
			gpuErrchk(cudaMalloc(&d_next_point_cache, sizeof(PointCache))) // * clmd->point_cache->totpoint
			gpuErrchk(cudaMalloc(&d_prev_point_cache, sizeof(PointCache))) //  * clmd->point_cache->totpoint
			gpuErrchk(cudaMalloc(&d_ptcaches, sizeof(ListBase)))
			gpuErrchk(cudaMalloc(&(d_ptcaches_first), sizeof(void*))) //  * clmd->point_cache->totpoint
			gpuErrchk(cudaMalloc(&(d_ptcaches_last), sizeof(void*))) //  * clmd->point_cache->totpoint
			gpuErrchk(cudaMalloc(&d_solver_result, sizeof(ClothSolverResult)))

	ClothSimSettings temp_sim_parms = *clmd->sim_parms;
	EffectorWeights temp_effector_weights; //  = *clmd->sim_parms->effector_weights
	// Копирование tempSewEdgeGraph в память GPU
	gpuErrchk(cudaMemcpy(d_effector_weights, &temp_effector_weights, sizeof(EffectorWeights), cudaMemcpyHostToDevice))
	// Обновление указателя sew_edge_graph в tempCloth
	temp_sim_parms.effector_weights = d_effector_weights;
	gpuErrchk(cudaMemcpy(d_effector_weights, &temp_effector_weights, sizeof(EffectorWeights), cudaMemcpyHostToDevice))

	temp_sim_parms.effector_weights = d_effector_weights;
	gpuErrchk(cudaMemcpy(d_sim_parms, &temp_sim_parms, sizeof(ClothSimSettings), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_coll_parms, clmd->coll_parms, sizeof(ClothCollSettings), cudaMemcpyHostToDevice))

	// Обновление указателей в d_clmd
	ClothModifierData tempclmd = *clmd;
	Cloth tempCloth = *(clmd->clothObject);

	tempclmd.modifier.next = d_nextModifier;
	tempclmd.modifier.prev = d_prevModifier;

	// Обновление полей в tempCloth
	tempCloth.verts = d_clothVertex;

	d_springs = copyClothSpringToGPU(clmd->clothObject->springs, sizeof(ClothSpring));
	tempCloth.springs = d_springs;

	// Обновление bvhtree
	d_bvhtree = copyBVHTreeToDevice(clmd->clothObject->bvhtree, clmd->clothObject->primitive_num);
	tempCloth.bvhtree = d_bvhtree;

	// Обновление bvhselftree
	d_bvhselftree = copyBVHTreeToDevice(clmd->clothObject->bvhselftree, clmd->clothObject->primitive_num);
	tempCloth.bvhselftree = d_bvhselftree;

	// Обновление edgeset
	EdgeSet tempEdgeSet = *(tempCloth.edgeset);
	tempEdgeSet.entries = d_entries;
	tempEdgeSet.map = d_map;
	gpuErrchk(cudaMemcpy(d_edgeset, &tempEdgeSet, sizeof(EdgeSet), cudaMemcpyHostToDevice))
	tempCloth.edgeset = d_edgeset;

	// Обновление sew_edge_graph
	if (tempCloth.sew_edge_graph) 
	{
		// Если sew_edge_graph уже существует, просто обновите его поля
		EdgeSet tempSewEdgeGraph = *(tempCloth.sew_edge_graph);
		tempSewEdgeGraph.entries = d_sew_entries;
		tempSewEdgeGraph.map = d_sew_map;
		gpuErrchk(cudaMemcpy(d_sew_edge_graph, &tempSewEdgeGraph, sizeof(EdgeSet), cudaMemcpyHostToDevice))
		tempCloth.sew_edge_graph = d_sew_edge_graph;
	}
	else 
	{
		// Создание временного объекта EdgeSet
		EdgeSet tempSewEdgeGraph;
		// Назначение полей tempSewEdgeGraph
		tempSewEdgeGraph.entries = d_sew_entries;
		tempSewEdgeGraph.map = d_sew_map;
		// Копирование tempSewEdgeGraph в память GPU
		gpuErrchk(cudaMemcpy(d_sew_edge_graph, &tempSewEdgeGraph, sizeof(EdgeSet), cudaMemcpyHostToDevice))
		// Обновление указателя sew_edge_graph в tempCloth
		tempCloth.sew_edge_graph = d_sew_edge_graph;
	}

	// Обновление implicit
	d_implicit = copyImplicit_DataToDevice(clmd->clothObject->implicit, clmd->clothObject->mvert_num);
	tempCloth.implicit = d_implicit;
	gpuErrchk(cudaMemcpy(d_tri, &clmd->clothObject->tri, sizeof(MVertTri), cudaMemcpyHostToDevice))
	for (uint i = 0; i < clmd->clothObject->mvert_num; ++i)
	{
		gpuErrchk(cudaMemcpy(&d_tri[i], &clmd->clothObject->tri[i], sizeof(MVertTri), cudaMemcpyHostToDevice))
	}
	tempCloth.tri = d_tri;

	// Копирование обновленной структуры tempCloth обратно на устройство
	gpuErrchk(cudaMemcpy(d_cloth, &tempCloth, sizeof(Cloth), cudaMemcpyHostToDevice))
	tempclmd.clothObject = d_cloth;

	ClothSimSettings tmp_ClothSettings;
	tmp_ClothSettings.effector_weights = d_effector_weights;

	tempclmd.sim_parms = &tmp_ClothSettings;
	tempclmd.sim_parms->effector_weights = tmp_ClothSettings.effector_weights;
	tempclmd.sim_parms = d_sim_parms;

	ClothCollSettings tmp_coll_settings = *clmd->coll_parms;
	LinkNode tmp_coll_settingsLinkNode;

	// Копируем данные в эти временные структуры
	tmp_coll_settingsLinkNode.next = d_collision_list_next;
	tmp_coll_settingsLinkNode.link = d_collision_list_link;
	tmp_coll_settings.collision_list = d_collision_list;

	// Копируем временные структуры обратно в GPU-память
	gpuErrchk(cudaMemcpy(d_collision_list, &tmp_coll_settingsLinkNode, sizeof(LinkNode), cudaMemcpyHostToDevice))
	gpuErrchk(cudaMemcpy(d_coll_parms, &tmp_coll_settings, sizeof(ClothCollSettings), cudaMemcpyHostToDevice))

	// Обновляем tempclmd.coll_parms, чтобы указывать на d_coll_parms
	tempclmd.coll_parms = d_coll_parms;

	// 1. Создаём временные структуры на CPU
	PointCache tmp_point_cache; //  = *(clmd->point_cache)

	// 2. Обновляем временные структуры данными
	tmp_point_cache.next = d_next_point_cache;
	tmp_point_cache.prev = d_prev_point_cache;

	// 3. Скопируем временные структуры обратно на GPU
	gpuErrchk(cudaMemcpy(d_point_cache, &tmp_point_cache, sizeof(PointCache), cudaMemcpyHostToDevice))

	// Обновляем tempclmd.point_cache, чтобы указывать на d_point_cache
	tempclmd.point_cache = d_point_cache;

	//ListBase tmp_ptcaches;
	//// Обновляем поля tmp_ptcaches
	//tmp_ptcaches.first = d_ptcaches_first;
	//tmp_ptcaches.last = d_ptcaches_last;

	//// Скопируем tmp_ptcaches обратно на GPU
	//gpuErrchk(cudaMemcpy(d_ptcaches, &tmp_ptcaches, sizeof(ListBase), cudaMemcpyHostToDevice));

	//// Обновляем tempclmd.ptcaches, чтобы указывать на d_ptcaches
	//tempclmd.ptcaches = d_ptcaches;


	//tempclmd.ptcaches.first = d_ptcaches_first;
	//tempclmd.ptcaches.last = d_ptcaches_last;
	tempclmd.solver_result = d_solver_result;

	gpuErrchk(cudaMemcpy(d_clmd, &tempclmd, sizeof(ClothModifierData), cudaMemcpyHostToDevice))

	/*int blockSize = 256;
	int numBlocks = (clmd->clothObject->numsprings + blockSize - 1) / blockSize*/

	// Засечение времени симуляции
	CUDA_TIMEIT_START(cloth_step);
	gpuErrchk(cudaDeviceSynchronize())
	g_do_step_cloth<<<32, 1>>>(d_depsgraph, d_ob, d_clmd, d_mesh, d_mvert);
	gpuErrchk(cudaDeviceSynchronize())

	CUDA_TIMEIT_END(cloth_step);

	// Шаг 1: Создание временного массива
	auto original_coords = static_cast<float(*)[3]>(malloc(sizeof(float[3]) * mesh->totvert));

	// Шаг 2: Сохранение оригинальных координат вершин
	for (int i = 0; i < mesh->totvert; ++i) {
		copy_v3_v3(original_coords[i], mesh->mvert[i].co);
	}

	gpuErrchk(cudaMemcpy(mesh->mvert, d_mvert, sizeof(MVert) * mesh->totvert, cudaMemcpyDeviceToHost))

	// Шаг 4: Сравнение данных
	bool data_unchanged = true;
	for (int i = 0; i < mesh->totvert; ++i) {
		if (!equals_v3v3(original_coords[i], mesh->mvert[i].co)) {
			data_unchanged = false;
			break;
		}
	}

	// Шаг 5: Вывод сообщения, если данные не изменились
	if (data_unchanged) 
	{
		std::cout << "In main.cpp they are the same\n";
	}

	// Освобождение памяти
	free(original_coords);
	freeImplicitDataOnDevice(d_implicit);

	gpuErrchk(cudaFree(d_clmd))
		gpuErrchk(cudaFree(d_modifier))
		gpuErrchk(cudaFree(d_nextModifier))
		gpuErrchk(cudaFree(d_prevModifier))
		gpuErrchk(cudaFree(d_cloth))
		gpuErrchk(cudaFree(d_clothVertex))
		freeClothSpringOnGPU(d_springs);
		gpuErrchk(cudaFree(d_tri))
		gpuErrchk(cudaFree(d_edgeset))
		gpuErrchk(cudaFree(d_entries))
		gpuErrchk(cudaFree(d_map))
		gpuErrchk(cudaFree(d_edges))
		gpuErrchk(cudaFree(d_sew_edge_graph))
		gpuErrchk(cudaFree(d_sew_entries))
		gpuErrchk(cudaFree(d_sew_map))
		gpuErrchk(cudaFree(d_sim_parms))
		gpuErrchk(cudaFree(d_effector_weights))
		freeBVHTreeFromDevice(d_bvhtree);
		freeBVHTreeFromDevice(d_bvhselftree);
		gpuErrchk(cudaFree(d_coll_parms))
		gpuErrchk(cudaFree(d_collision_list))
		gpuErrchk(cudaFree(d_collision_list_next))
		gpuErrchk(cudaFree(d_collision_list_link))
		gpuErrchk(cudaFree(d_point_cache))
		gpuErrchk(cudaFree(d_next_point_cache))
		gpuErrchk(cudaFree(d_prev_point_cache))
		gpuErrchk(cudaFree(d_ptcaches))
		gpuErrchk(cudaFree(d_ptcaches_first))
		gpuErrchk(cudaFree(d_ptcaches_last))
		gpuErrchk(cudaFree(d_solver_result))

	gpuErrchk(cudaFree(d_depsgraph))
		gpuErrchk(cudaFree(d_depsgraph_scene))
		gpuErrchk(cudaFree(d_toolsettings))
		gpuErrchk(cudaFree(d_depsgraph_scene_cow))
		gpuErrchk(cudaFree(d_cow_toolsettings))

	gpuErrchk(cudaFree(d_ob))
		gpuErrchk(cudaFree(d_ob_data))
		gpuErrchk(cudaFree(d_ob_PartDeflect))
		gpuErrchk(cudaFree(d_ob_rng))

	gpuErrchk(cudaFree(d_mesh))
	    gpuErrchk(cudaFree(d_mvert))
	    gpuErrchk(cudaFree(d_medge))
	    gpuErrchk(cudaFree(d_mpoly))
	    gpuErrchk(cudaFree(d_mloop))

    return true;
}

// Вывод в консоль сообщения о том, что всё работает
void print_work_info()
{
    //printf_s("\nDLL file is work!!!\n\n");

    system("chcp 1251"); //Подключаем русский язык

    cudaDeviceProp prop{};
    int count;

    cudaGetDeviceCount(&count);    
    for (int i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        printf_s("\n-----Общая информация об устройстве %d-----\n", i);
        printf_s("Кол-во устройств: %d\n", count);
        printf_s("Имя: %s\n", prop.name);
        printf_s("Вычислительные возможности: %d.%d\n", prop.major, prop.minor);
        printf_s("Тактовая частота: %d\n", prop.clockRate);
        printf_s("Количество асинхронных двигателей: %d\n", prop.asyncEngineCount);

        printf_s("Тайм-аут выполнения ядра: ");
        if (prop.kernelExecTimeoutEnabled) { printf_s("Включён \n"); }
        else { printf_s("Выключен\n"); }

        printf_s("\n---Информация о памяти устройства %d---\n", i);
        printf_s("Всего глобальной памяти: %zd\n", prop.totalGlobalMem);
        printf_s("Всего константной памяти: %zd\n", prop.totalConstMem);
        printf_s("Максимальный шаг: %zd\n", prop.memPitch);
        printf_s("Выравнивание текстур: %zd\n", prop.textureAlignment);

        printf_s("\n---Информация о сультипроцессорах устройства %d---\n", i);
        printf_s("Количество мультипроцессоров: %d\n", prop.multiProcessorCount);
        printf_s("Разделяемая память на один МП: %zd\n", prop.sharedMemPerBlock);
        printf_s("Регистров на один МП: %d\n", prop.regsPerBlock);
        printf_s("Нитей в варпе: %d\n", prop.warpSize);
        printf_s("Макс. колличество нитей в блоке: %d\n", prop.maxThreadsPerBlock);
        printf_s("Макс.количество нитей по измерениям:(%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf_s("Макс. размер сетки:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}