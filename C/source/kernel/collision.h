#pragma once

#include "modifier_types.cuh"
#include "listbase.h"
#include "mallocn_intern.h"

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

struct BVHTree *bvhtree_build_from_mvert(const struct MVert *mvert,
                                         const struct MVertTri *tri,
                                         int tri_num,
                                         float epsilon);
void bvhtree_update_from_mvert(struct BVHTree *bvhtree,
                               const struct MVert *mvert,
                               const struct MVert *mvert_moving,
                               const struct MVertTri *tri,
                               int tri_num,
                               bool moving);

/////////////////////////////////////////////////

/* move Collision modifier object inter-frame with step = [0,1]
 * defined in collisions.c */
void collision_move_object(CollisionModifierData* collmd, const float step, const float prevstep, const bool moving_bvh);
void collision_get_collider_velocity(float vel_old[3],
                                     float vel_new[3],
                                     struct CollisionModifierData *collmd,
                                     struct CollPair *collpair);

/* Collision relations for dependency graph build. */

typedef struct CollisionRelation {
  struct CollisionRelation *next, *prev;
  struct Object *ob;
} CollisionRelation;

//struct ListBase *BKE_collision_relations_create(struct Depsgraph *depsgraph,
//                                                struct Collection *collection,
//                                                uint modifier_type);

void BKE_collision_relations_free(struct ListBase* relations);

/* Collision object lists for physics simulation evaluation. */

struct Object **BKE_collision_objects_create(struct Depsgraph *depsgraph,
                                             struct Object *self, struct Collection *collection,
                                             uint *numcollobj,  uint modifier_type);
void BKE_collision_objects_free(struct Object **objects);

typedef struct ColliderCache {
  struct ColliderCache *next, *prev;
  struct Object *ob;
  struct CollisionModifierData *collmd;
} ColliderCache;

struct ListBase *BKE_collider_cache_create(struct Depsgraph *depsgraph,
                                           struct Object *self,
                                           struct Collection *collection);
void BKE_collider_cache_free(struct ListBase **colliders);

/////////////////////////////////////////////////

/////////////////////////////////////////////////
#ifdef __cplusplus
}
#endif