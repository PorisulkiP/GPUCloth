#pragma once

#include <stdlib.h>

#include "MEM_guardedalloc.cuh"
#include "ID.h" /* for ID_Type and INDEX_ID_MAX */
#include "collision.h"
#include "DEG_depsgraph.cuh"
#include "BLI_map.cuh"
#include "mallocn_intern.cuh"
#include "scene.h"

struct ID;
struct Scene;
struct ViewLayer;

struct IDNode;
struct Node;
struct OperationNode;
struct Relation;
struct TimeSourceNode;

inline int BKEtype_get_factory_idtype_idcode_to_index(const short idcode)
{
#define CASE_IDINDEX(_id) \
  case ID_##_id: \
    return INDEX_ID_##_id

	switch ((ID_Type)idcode) {
		CASE_IDINDEX(AC);
		CASE_IDINDEX(AR);
		CASE_IDINDEX(BR);
		CASE_IDINDEX(CA);
		CASE_IDINDEX(CF);
		CASE_IDINDEX(CU_LEGACY);
		CASE_IDINDEX(GD);
		CASE_IDINDEX(GR);
		CASE_IDINDEX(CV);
		CASE_IDINDEX(IM);
		CASE_IDINDEX(IP);
		CASE_IDINDEX(KE);
		CASE_IDINDEX(LA);
		CASE_IDINDEX(LI);
		CASE_IDINDEX(LS);
		CASE_IDINDEX(LT);
		CASE_IDINDEX(MA);
		CASE_IDINDEX(MB);
		CASE_IDINDEX(MC);
		CASE_IDINDEX(ME);
		CASE_IDINDEX(MSK);
		CASE_IDINDEX(NT);
		CASE_IDINDEX(OB);
		CASE_IDINDEX(PA);
		CASE_IDINDEX(PAL);
		CASE_IDINDEX(PC);
		CASE_IDINDEX(PT);
		CASE_IDINDEX(LP);
		CASE_IDINDEX(SCE);
		CASE_IDINDEX(SCR);
		CASE_IDINDEX(SIM);
		CASE_IDINDEX(SPK);
		CASE_IDINDEX(SO);
		CASE_IDINDEX(TE);
		CASE_IDINDEX(TXT);
		CASE_IDINDEX(VF);
		CASE_IDINDEX(VO);
		CASE_IDINDEX(WM);
		CASE_IDINDEX(WO);
		CASE_IDINDEX(WS);
	}

	/* Special naughty boy... */
	if (idcode == ID_LINK_PLACEHOLDER) {
		return INDEX_ID_NULL;
	}

	return -1;

#undef CASE_IDINDEX
}

struct TimeSourceNode : public Node {
	bool tagged_for_update = false;

	virtual void tag_update(Depsgraph* graph, eUpdateSource source)
	{
		tagged_for_update = true;
	}

	void flush_update_tag(Depsgraph* graph)
	{
		if (!tagged_for_update) 
		{
			return;
		}
		tag_update(graph, DEG_UPDATE_SOURCE_TIME);
		//for (Relation* rel : outlinks) 
		//{
		//	Node* node = rel->to;
		//	tag_update(graph, DEG_UPDATE_SOURCE_TIME);
		//}
	}
};

struct DepsNodeFactory {
	__host__ __device__ virtual NodeType type() const = 0;
	__host__ __device__ virtual const char* type_name() const = 0;

	__host__ __device__ virtual int id_recalc_tag() const = 0;

	__host__ __device__ virtual Node* create_node(const ID* id, const char* subdata, const char* name) const = 0;
};

//template<class ModeObjectType> struct DepsNodeFactoryImpl : public DepsNodeFactory {
//	virtual NodeType type() const override;
//	virtual const char* type_name() const override;
//
//	virtual int id_recalc_tag() const override;
//
//	virtual Node* create_node(const ID* id, const char* subdata, const char* name) const override;
//};

static DepsNodeFactory* node_typeinfo_registry[static_cast<int>(NodeType::NUM_TYPES)] = { nullptr };

// Объявление переменной в памяти устройства
__device__ static DepsNodeFactory* d_node_typeinfo_registry[static_cast<int>(NodeType::NUM_TYPES)];

// Функция для копирования данных в память устройства
__host__ void copyRegistryToDevice() {
	cudaMemcpyToSymbol(d_node_typeinfo_registry, node_typeinfo_registry, sizeof(DepsNodeFactory*) * static_cast<int>(NodeType::NUM_TYPES), 0, cudaMemcpyHostToDevice);
}
__host__ __device__ inline DepsNodeFactory* type_get_factory(NodeType type) {
#ifdef __CUDA_ARCH__
	// Версия для device-кода
	return d_node_typeinfo_registry[(int)type];
#else
	// Версия для host-кода
	return node_typeinfo_registry[(int)type];
#endif
}

/* Register typeinfo */
//void register_node_typeinfo(DepsNodeFactory* factory)
//{
//	BLI_assert(factory != nullptr);
//	const int type_as_int = static_cast<int>(factory->type());
//	node_typeinfo_registry[type_as_int] = factory;
//}

__host__ __device__ inline bool check_datablock_expanded(const ID* id_cow)
{
	return (id_cow->name[0] != '\0');
}

__host__ __device__ inline bool deg_copy_on_write_is_expanded(const ID* id_cow)
{
	return check_datablock_expanded(id_cow);
}

/* Dependency Graph object */
/* Объект графика зависимостей */
struct Depsgraph {
	typedef blender::Vector<OperationNode*> OperationNodes;
	typedef blender::Vector<IDNode*> IDDepsNodes;

	Depsgraph(Main* bmain, Scene* scene, ViewLayer* view_layer, eEvaluationMode mode): scene(scene), mode(), ctime(0), scene_cow(nullptr)
	//: time_source(nullptr),
	//has_animated_visibility(false),
	//need_update_relations(true),
	//need_update_nodes_visibility(true),
	//need_tag_id_on_graph_visibility_update(true),
	//need_tag_id_on_graph_visibility_time_update(false),
	//bmain(bmain),
	//scene(scene),
	//view_layer(view_layer),
	//mode(mode),
	//frame(scene->r.cfra + scene->r.subframe),
	//ctime((scene->r.cfra + scene->r.subframe) * scene->r.framelen),
	//scene_cow(nullptr)//,
	//is_active(false),
	//is_evaluating(false),
	//is_render_pipeline_depsgraph(false),
	//use_editors_update(false)
	{
		BLI_spin_init(&lock);
		//memset(id_type_updated, 0, sizeof(id_type_updated));
		//memset(id_type_exist, 0, sizeof(id_type_exist));
		//memset(physics_relations, 0, sizeof(physics_relations));

		//add_time_source();
	}

	~Depsgraph()
	{
		clear_id_nodes();
		//delete time_source;
		BLI_spin_end(&lock);
	}

	//TimeSourceNode* add_time_source()
	//{
	//	if (time_source == nullptr) {
	//		DepsNodeFactory* factory = type_get_factory(NodeType::TIMESOURCE);
	//		time_source = (TimeSourceNode*)factory->create_node(nullptr, "", "Time Source");
	//	}
	//	return time_source;
	//}
	//TimeSourceNode* find_time_source() const
	//{
	//	return time_source;
	//}
	//void tag_time_source()
	//{
	//	time_source->tag_update(this, DEG_UPDATE_SOURCE_TIME);
	//}

	__host__ __device__ static void BKE_effector_relations_free(ListBase* lb)
	{
		if (lb) {
			BLI_freelistN(lb);
			MEM_lockfree_freeN(lb);
		}
	}
	__host__ __device__ void clear_physics_relations(Depsgraph* graph);

	__host__ __device__ IDNode* find_id_node(const ID* id) const
	{
		return id_hash.lookup_default(id, nullptr);
	}

	__host__ __device__ IDNode* add_id_node(ID* id, ID* id_cow_hint = nullptr)
	{
		BLI_assert((id->tag & LIB_TAG_COPIED_ON_WRITE) == 0);
		IDNode* id_node = find_id_node(id);
		if (!id_node) {
			DepsNodeFactory* factory = type_get_factory(NodeType::ID_REF);
			id_node = (IDNode*)factory->create_node(id, "", id->name);
			//id_node->init_copy_on_write(id_cow_hint);
			/* Register node in ID hash.
			 *
			 * NOTE: We address ID nodes by the original ID pointer they are
			 * referencing to. */
			id_hash.add_new(id, id_node);
			//id_nodes.append(id_node);

			//id_type_exist[BKE_idtype_idcode_to_index(GS(id->name))] = 1;
		}
		return id_node;
	}
	__host__ __device__ void clear_id_nodes();

	///** Add new relationship between two nodes. */
	//Relation* add_new_relation(Node* from, Node* to, const char* description, int flags = 0);

	///* Check whether two nodes are connected by relation with given
	// * description. Description might be nullptr to check ANY relation between
	// * given nodes. */
	//Relation* check_nodes_connected(const Node* from, const Node* to, const char* description);

	///* Tag a specific node as needing updates. */
	//void add_entry_tag(OperationNode* node);

	///* Clear storage used by all nodes. */
	//void clear_all_nodes();

	/* Copy-on-Write Functionality ........ */

	/* For given original ID get ID which is created by CoW system. */
	//ID* get_cow_id(const ID* id_orig) const;

	/* Top-level time source node. */
	/* Узел источника времени верхнего уровня. */
	//TimeSourceNode* time_source;

	/* The graph contains data-blocks whose visibility depends on evaluation (driven or animated). */
	/* График содержит блоки данных, видимость которых зависит от оценки (управляемой или анимированной). */
	//bool has_animated_visibility;

	/* Indicates whether relations needs to be updated. */
	/* Указывает, необходимо ли обновлять отношения. */
	//bool need_update_relations;

	/* Indicates whether indirect effect of nodes on a directly visible ones needs to be updated. */
	/* Указывает, необходимо ли обновлять косвенное влияние узлов на непосредственно видимые узлы. */
	//bool need_update_nodes_visibility;

	/* Indicated whether IDs in this graph are to be tagged as if they first appear visible, with
	 * an optional tag for their animation (time) update. */
	 /* Указано, следует ли помечать идентификаторы на этом графике так, как если бы они впервые появились видимыми, с
	  * необязательным тегом для обновления их анимации (времени). */
	//bool need_tag_id_on_graph_visibility_update;
	//bool need_tag_id_on_graph_visibility_time_update;

	/* Indicates which ID types were updated. */
	/* Указывает, какие типы идентификаторов были обновлены. */
	//char id_type_updated[INDEX_ID_MAX];

	/* Indicates type of IDs present in the depsgraph. */
	//char id_type_exist[INDEX_ID_MAX];
	/* Main, scene, layer, mode this dependency graph is built for. */
	//Main* bmain;
	Scene* scene;
	//ViewLayer* view_layer;
	eEvaluationMode mode;

	/* Time at which dependency graph is being or was last evaluated.
	 * frame is the value before, and ctime the value after time remapping. */
	 /* Время, в которое выполняется или был выполнен последний расчет графика зависимостей.
	  * * кадр - это значение до, а время - значение после переназначения времени. */
	//float frame;
	float ctime;

	/* Evaluated version of datablocks we access a lot.
	 * Stored here to save us form doing hash lookup. */
	/* Оцененная версия блоков данных, к которым мы часто обращаемся.
	 * Хранится здесь, чтобы избавить нас от выполнения хэш-поиска. */
	Scene* scene_cow;

	/* Active dependency graph is a dependency graph which is used by the
	 * currently active window. When dependency graph is active, it is allowed
	 * for evaluation functions to write animation f-curve result, drivers
	 * result and other selective things (object matrix?) to original object.
	 *
	 * This way we simplify operators, which don't need to worry about where
	 * to read stuff from. */
	//bool is_active;

	//bool is_evaluating;

	/* Is set to truth for dependency graph which are used for post-processing (compositor and
	 * sequencer).
	 * Such dependency graph needs all view layers (so render pipeline can access names), but it
	 * does not need any bases. */
	//bool is_render_pipeline_depsgraph;

	/* Notify editors about changes to IDs in this depsgraph. */
	//bool use_editors_update;

	//blender::Map<const ID*, ListBase*>* physics_relations[DEG_PHYSICS_RELATIONS_NUM];
	blender::Map<const ID*, IDNode*> id_hash{};
	//IDDepsNodes id_nodes;
	SpinLock lock;
};

inline void print_Depsgraph(const Depsgraph *deps)
{
	printf("Depsgraph:");
	print_Scene(deps->scene);
	printf("\n\tctime: %f", deps->ctime);
	printf_s("\n\tsize of id_hash: %d", deps->id_hash.size(), 255);
	printf("\n\tspinLock: %d", deps->lock);
}