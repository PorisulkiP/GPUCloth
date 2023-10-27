#pragma once

#ifndef __COLLISION__
#define __COLLISION__

#include "modifier_types.cuh"

#ifdef __cplusplus
extern "C" {
#endif

struct CollisionModifierData;
struct BVHTree;
struct Collection;
struct Depsgraph;
struct MVert;
struct MVertTri;
struct Object;

////////////////////////////////////////

/////////////////////////////////////////////////
// forward declarations
/////////////////////////////////////////////////

/////////////////////////////////////////////////
// used in modifier.c from collision.c
/////////////////////////////////////////////////

__host__ __device__ BVHTree *bvhtree_build_from_mvert(const MVert *mvert,
                                  const MVertTri *tri,
                                  int tri_num,
                                  float epsilon);
__host__ __device__ void bvhtree_update_from_mvert(BVHTree *bvhtree,
                               const MVert *mvert,
                               const MVert *mvert_moving,
                               const MVertTri *tri,
                               int tri_num,
                               bool moving);

/////////////////////////////////////////////////

/* move Collision modifier object inter-frame with step = [0,1]
 * defined in collisions.c */
__host__ __device__ void collision_move_object(const CollisionModifierData* collmd, const float step, const float prevstep, const bool moving_bvh);

__host__ __device__ void collision_get_collider_velocity(float vel_old[3],
                                     float vel_new[3],
                                     const CollisionModifierData *collmd,
                                     const CollPair *collpair);

/* Collision relations for dependency graph build. */

typedef struct CollisionRelation {
	CollisionRelation *next, *prev;
	Object *ob;
} CollisionRelation;

//struct ListBase *BKE_collision_relations_create(struct Depsgraph *depsgraph,
//                                                struct Collection *collection,
//                                                uint modifier_type);

__host__ __device__ void BKE_collision_relations_free(ListBase* relations);

/* Collision object lists for physics simulation evaluation. */

__host__ __device__ Object **BKE_collision_objects_create(const Depsgraph *depsgraph,
                                      const Object *self, Collection *collection,
                                      uint *numcollobj,  uint modifier_type);

__host__ __device__ void BKE_collision_objects_free(Object **objects);

typedef struct ColliderCache {
	ColliderCache *next, *prev;
	Object *ob;
	CollisionModifierData *collmd;
} ColliderCache;

__host__ __device__ ListBase *BKE_collider_cache_create(const Depsgraph *depsgraph,
                                    const Object *self,
                                    Collection *collection);
__host__ __device__ void BKE_collider_cache_free(ListBase **colliders);

/////////////////////////////////////////////////

/////////////////////////////////////////////////
#ifdef __cplusplus
}
#endif

#endif