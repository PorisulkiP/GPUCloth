#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "MEM_guardedalloc.cuh"

#include "listbase.cuh"
#include "B_math.h"

#include "ID.h"
#include "meshdata_types.cuh"
#include "object_force_types.cuh"
#include "object_types.cuh"
#include "scene_types.cuh"

#include "B_collection.h"
#include "effect.h"
#include "DEG_depsgraph.cuh"
#include "DEG_depsgraph_query.cuh"


/* ************************************** */
/* Memory Management */

/* Freeing Methods --------------------- */

#ifdef WITH_BULLET
static void rigidbody_update_ob_array(RigidBodyWorld *rbw);

#else
static void RB_dworld_remove_constraint(void *UNUSED(world), void *UNUSED(con))
{
}
static void RB_dworld_remove_body(void *UNUSED(world), void *UNUSED(body))
{
}
static void RB_dworld_delete(void *UNUSED(world))
{
}
static void RB_body_delete(void *UNUSED(body))
{
}
static void RB_shape_delete(void *UNUSED(shape))
{
}
static void RB_constraint_delete(void *UNUSED(con))
{
}

#endif

/* Free RigidBody constraint and sim instance */
void BKE_rigidbody_free_constraint(Object *ob)
{
  //RigidBodyCon* rbc = NULL; //(ob) ? ob->rigidbody_constraint : NULL;

  ///* sanity check */
  //if (rbc == NULL) 
  //{
  //  return;
  //}

  /* free data itself */
  //MEM_lockfree_freeN(rbc);
  //ob->rigidbody_constraint = NULL;
}

bool BKE_rigidbody_is_affected_by_simulation(Object *ob)
{
  return true;
}

/* stubs */
#  if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#  endif

void BKE_rigidbody_object_copy(Main *bmain, Object *ob_dst, const Object *ob_src, const int flag)
{
}
void BKE_rigidbody_validate_sim_world(Scene *scene, RigidBodyWorld *rbw, bool rebuild)
{
}
void BKE_rigidbody_calc_volume(Object *ob, float *r_vol)
{
  if (r_vol) {
    *r_vol = 0.0f;
  }
}
void BKE_rigidbody_calc_center_of_mass(Object *ob, float r_center[3])
{
  zero_v3(r_center);
}
struct RigidBodyWorld *BKE_rigidbody_create_world(Scene *scene)
{
  return NULL;
}
struct RigidBodyWorld *BKE_rigidbody_world_copy(RigidBodyWorld *rbw, const int flag)
{
  return NULL;
}
void BKE_rigidbody_world_groups_relink(struct RigidBodyWorld *rbw)
{
}
struct RigidBodyOb *BKE_rigidbody_create_object(Scene *scene, Object *ob, short type)
{
  return NULL;
}
struct RigidBodyCon *BKE_rigidbody_create_constraint(Scene *scene, Object *ob, short type)
{
  return NULL;
}
struct RigidBodyWorld *BKE_rigidbody_get_world(Scene *scene)
{
  return NULL;
}

void BKE_rigidbody_ensure_local_object(Main *bmain, Object *ob)
{
}

void BKE_rigidbody_remove_object(struct Main *bmain, Scene *scene, Object *ob, const bool free_us)
{
}
void BKE_rigidbody_remove_constraint(Main *bmain, Scene *scene, Object *ob, const bool free_us)
{
}
void BKE_rigidbody_sync_transforms(RigidBodyWorld *rbw, Object *ob, float ctime)
{
}
void BKE_rigidbody_aftertrans_update(
    Object *ob, float loc[3], float rot[3], float quat[4], float rotAxis[3], float rotAngle)
{
}
bool BKE_rigidbody_check_sim_running(RigidBodyWorld *rbw, float ctime)
{
  return false;
}
void BKE_rigidbody_cache_reset(RigidBodyWorld *rbw)
{
}
void BKE_rigidbody_rebuild_world(Depsgraph *depsgraph, Scene *scene, float ctime)
{
}
void BKE_rigidbody_do_simulation(Depsgraph *depsgraph, Scene *scene, float ctime)
{
}
void BKE_rigidbody_objects_collection_validate(Scene *scene, RigidBodyWorld *rbw)
{
}
void BKE_rigidbody_constraints_collection_validate(Scene *scene, RigidBodyWorld *rbw)
{
}
void BKE_rigidbody_main_collection_object_add(Main *bmain, Collection *collection, Object *object)
{
}

/* -------------------- */
/* Depsgraph evaluation */

//void BKE_rigidbody_object_sync_transforms(Depsgraph *depsgraph, Scene *scene, Object *ob)
//{
//  RigidBodyWorld *rbw = scene->rigidbody_world;
//  float ctime = depsgraph->ctime;
//  //DEG_debug_print_eval_time(depsgraph, __func__, ob->id.name, ob, ctime);
//  /* read values pushed into RBO from sim/cache... */
//  BKE_rigidbody_sync_transforms(rbw, ob, ctime);
//}
