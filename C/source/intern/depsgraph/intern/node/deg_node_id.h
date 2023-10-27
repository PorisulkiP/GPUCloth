#pragma once

#include "ghash.cuh"
#include "sys_types.cuh"
#include "ID.h"
#include "node/deg_node.h"

namespace blender::deg {

	struct ComponentNode;

	typedef uint64_t IDComponentsMask;

	/* NOTE: We use max comparison to mark an id node that is linked more than once
	 * So keep this enum ordered accordingly. */
	enum eDepsNode_LinkedState_Type {
	  /* Generic indirectly linked id node. */
	  DEG_ID_LINKED_INDIRECTLY = 0,
	  /* Id node present in the set (background) only. */
	  DEG_ID_LINKED_VIA_SET = 1,
	  /* Id node directly linked via the SceneLayer. */
	  DEG_ID_LINKED_DIRECTLY = 2,
	};
	//const char *linkedStateAsString(eDepsNode_LinkedState_Type linked_state);
}  // namespace blender
