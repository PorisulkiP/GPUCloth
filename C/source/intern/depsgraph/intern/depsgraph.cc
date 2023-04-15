#include "depsgraph.h" /* own include */

#include <algorithm>

#include "MEM_guardedalloc.cuh"
#include "anim_types.h"
#include "utildefines.h"

#include "scene.h"

#include "DEG_depsgraph.h"
#include "intern/eval/deg_eval_copy_on_write.h"

#include "intern/node/deg_node.h"
#include "intern/node/deg_node_component.h"
#include "intern/node/deg_node_factory.h"
#include "intern/node/deg_node_id.h"
#include "intern/node/deg_node_operation.h"
#include "intern/node/deg_node_time.h"


template<typename FilterFunc>
static void clear_id_nodes_conditional(Depsgraph::IDDepsNodes* id_nodes, const FilterFunc& filter)
{
	for (IDNode* id_node : *id_nodes)
	{
		if (id_node->id_cow == nullptr)
		{
			/* This means builder "stole" ownership of the copy-on-written datablock for her own dirty needs. */
			/* Это означает, что builder "украл" право собственности на блок данных copy-on-written для своих собственных нужд. */
			continue;
		}
		if (id_node->id_cow == id_node->id_orig)
		{
			/* Copy-on-write version is not needed for this ID type.
			*
			* NOTE: Is important to not de-reference the original datablock here because it might be
			* freed already (happens during main database free when some IDs are freed prior to a
			* scene). */
			/* Версия копирования при записи не требуется для этого типа идентификатора.
			*
			* ПРИМЕЧАНИЕ: Здесь важно не разыменовывать исходный блок данных, потому что он может
			* быть уже освобожден (происходит во время освобождения основной базы данных,
			* когда некоторые идентификаторы освобождаются перед сценой). */
			continue;
		}
		if (!deg_copy_on_write_is_expanded(id_node->id_cow))
		{
			continue;
		}
		if (filter(GS(id_node->id_cow->name)))
		{
			id_node->destroy();
		}
	}
}

void Depsgraph::clear_id_nodes()
{
	/* Free memory used by ID nodes. */
	/* Stupid workaround to ensure we free IDs in a proper order. */
	//clear_id_nodes_conditional(&id_nodes, [](ID_Type id_type) { return id_type == ID_SCE; });
	//clear_id_nodes_conditional(&id_nodes, [](ID_Type id_type) { return id_type != ID_PA; });

	//for (IDNode* id_node : id_nodes) 
	//{
	//	delete id_node;
	//}
	/* Clear containers. */
	id_hash.clear();
	//id_nodes.clear();
	/* Clear physics relation caches. */
	//clear_physics_relations(this);
}

void Depsgraph::clear_physics_relations(Depsgraph* graph)
{
	//for (int i = 0; i < DEG_PHYSICS_RELATIONS_NUM; i++) {
	//	blender::Map<const ID*, ListBase*>* hash = graph->physics_relations[i];
	//	if (hash) 
	//	{
	//		const auto type = (ePhysicsRelationType)i;

	//		switch (type) {
	//		case DEG_PHYSICS_EFFECTOR:
	//			for (ListBase* list : hash->values()) {
	//				BKE_effector_relations_free(list);
	//			}
	//			break;
	//		case DEG_PHYSICS_COLLISION:
	//		case DEG_PHYSICS_SMOKE_COLLISION:
	//		case DEG_PHYSICS_DYNAMIC_BRUSH:
	//			for (ListBase* list : hash->values())
	//			{
	//				BKE_collision_relations_free(list);
	//			}
	//			break;
	//		case DEG_PHYSICS_RELATIONS_NUM:
	//			break;
	//		}
	//		delete hash;
	//		graph->physics_relations[i] = nullptr;
	//	}
	//}
}
