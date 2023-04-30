#pragma once

#include <float.h>
#include "pointcache.h"
#include "object.h"
#include "mesh_types.h"
#include "intern/atomic_ops_ext.h"

struct ClothModifierData;
struct CollisionModifierData;
struct Depsgraph;
struct GHash;
struct Object;
struct Scene;

#define DO_INLINE MALWAYS_INLINE

/* goal defines */
#define SOFTGOALSNAP 0.999f

/* This is approximately the smallest number that can be
 * represented by a float, given its precision. */
#define ALMOST_ZERO FLT_EPSILON

/* Bits to or into the ClothVertex.flags. */
typedef enum eClothVertexFlag {
  CLOTH_VERT_FLAG_PINNED = (1 << 0),
  CLOTH_VERT_FLAG_NOSELFCOLL = (1 << 1), /* vertex NOT used for self collisions */
  CLOTH_VERT_FLAG_NOOBJCOLL = (1 << 2),  /* vertex NOT used for object collisions */
} eClothVertexFlag;

typedef struct ClothHairData {
  float loc[3];
  float rot[3][3];
  float rest_target[3]; /* rest target direction for each segment */
  float radius;
  float bending_stiffness;
} ClothHairData;

typedef struct ClothSolverResult {
  int status;

  int max_iterations, min_iterations;
  float avg_iterations;
  float max_error, min_error, avg_error;
} ClothSolverResult;

// some macro enhancements for vector treatment
#define VECSUBADDSS(v1, v2, aS, v3, bS) \
  { \
    *(v1) -= *(v2)*aS + *(v3)*bS; \
    *(v1 + 1) -= *(v2 + 1) * aS + *(v3 + 1) * bS; \
    *(v1 + 2) -= *(v2 + 2) * aS + *(v3 + 2) * bS; \
  } \
  ((void)0)
#define VECADDSS(v1, v2, aS, v3, bS) \
  { \
    *(v1) = *(v2)*aS + *(v3)*bS; \
    *(v1 + 1) = *(v2 + 1) * aS + *(v3 + 1) * bS; \
    *(v1 + 2) = *(v2 + 2) * aS + *(v3 + 2) * bS; \
  } \
  ((void)0)
#define VECADDS(v1, v2, v3, bS) \
  { \
    *(v1) = *(v2) + *(v3)*bS; \
    *(v1 + 1) = *(v2 + 1) + *(v3 + 1) * bS; \
    *(v1 + 2) = *(v2 + 2) + *(v3 + 2) * bS; \
  } \
  ((void)0)
#define VECSUBMUL(v1, v2, aS) \
  { \
    *(v1) -= *(v2)*aS; \
    *(v1 + 1) -= *(v2 + 1) * aS; \
    *(v1 + 2) -= *(v2 + 2) * aS; \
  } \
  ((void)0)
#define VECSUBS(v1, v2, v3, bS) \
  { \
    *(v1) = *(v2) - *(v3)*bS; \
    *(v1 + 1) = *(v2 + 1) - *(v3 + 1) * bS; \
    *(v1 + 2) = *(v2 + 2) - *(v3 + 2) * bS; \
  } \
  ((void)0)
#define VECADDMUL(v1, v2, aS) \
  { \
    *(v1) += *(v2)*aS; \
    *(v1 + 1) += *(v2 + 1) * aS; \
    *(v1 + 2) += *(v2 + 2) * aS; \
  } \
  ((void)0)

/* Spring types as defined in the paper.*/
typedef enum {
  CLOTH_SPRING_TYPE_STRUCTURAL = (1 << 1),
  CLOTH_SPRING_TYPE_SHEAR = (1 << 2),
  CLOTH_SPRING_TYPE_BENDING = (1 << 3),
  CLOTH_SPRING_TYPE_GOAL = (1 << 4),
  CLOTH_SPRING_TYPE_SEWING = (1 << 5),
  CLOTH_SPRING_TYPE_BENDING_HAIR = (1 << 6),
  CLOTH_SPRING_TYPE_INTERNAL = (1 << 7),
} CLOTH_SPRING_TYPES;

/* SPRING FLAGS */
typedef enum {
  CLOTH_SPRING_FLAG_DEACTIVATE = (1 << 1),
  CLOTH_SPRING_FLAG_NEEDED = (1 << 2), /* Springs has values to be applied. */
} CLOTH_SPRINGS_FLAGS;

/////////////////////////////////////////////////
// collision.c
////////////////////////////////////////////////

struct CollPair;

typedef struct ColliderContacts {
  struct Object *ob;
  struct CollisionModifierData *collmd;

  struct CollPair *collisions;
  int totcollisions;
} ColliderContacts;

// needed for implicit.c
int cloth_bvh_collision(struct Depsgraph *depsgraph,
                        struct Object *ob,
                        struct ClothModifierData *clmd,
                        float step,
                        float dt);

int do_step_cloth(Depsgraph* depsgraph, Object* ob, ClothModifierData* clmd, Mesh* result, int framenr, int countOfObj);

////////////////////////////////////////////////

/////////////////////////////////////////////////
// cloth.c
////////////////////////////////////////////////

bool cloth_from_object(ClothModifierData* clmd, Mesh* mesh);
void cloth_from_mesh(ClothModifierData* clmd, Mesh* mesh);
void cloth_update_springs(ClothModifierData* clmd);
void cloth_update_verts(Object* ob, ClothModifierData* clmd, Mesh* mesh);
void cloth_update_spring_lengths(ClothModifierData* clmd, Mesh* mesh);
bool cloth_build_springs(ClothModifierData* clmd, Mesh* mesh);

// needed for modifier.c
void cloth_free_modifier_extern(struct ClothModifierData *clmd);
void cloth_free_modifier(struct ClothModifierData *clmd);
bool clothModifier_do(struct ClothModifierData *clmd, struct Depsgraph *depsgraph, struct Object *ob, struct Mesh *me);

int cloth_uses_vgroup(struct ClothModifierData *clmd);

// needed for collision.c
void bvhtree_update_from_cloth(struct ClothModifierData *clmd, bool moving, bool self);

// needed for button_object.c
void cloth_clear_cache(struct Object *ob, struct ClothModifierData *clmd, float framenr)
{
    PTCacheID pid;

    BKE_ptcache_id_from_cloth(&pid, ob, clmd);

    /* don't do anything as long as we're in editmode! */
    // потом сделаю возможность редактирования кэша в editmode
    //if (pid.cache->edit && ob->mode & OB_MODE_PARTICLE_EDIT) 
    //{
    //    return;
    //}

    BKE_ptcache_id_clear(&pid, 3, (uint)framenr); // PTCACHE_CLEAR_AFTER 3
}


void cloth_parallel_transport_hair_frame(float mat[3][3],
                                         const float dir_old[3],
                                         const float dir_new[3]);
