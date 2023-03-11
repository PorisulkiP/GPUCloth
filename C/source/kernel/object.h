#pragma once

#include "compiler_attrs.cuh"
#include "sys_types.cuh"
#include "object_types.cuh"
#include "lib_query.h"
#include "modifier_types.cuh"

struct Base;
struct BoundBox;
struct Depsgraph;
struct GpencilModifierData;
struct HookGpencilModifierData;
struct HookModifierData;
struct ID;
struct Main;
struct Mesh;
struct ModifierData;
struct MovieClip;
struct Object;
struct RegionView3D;
struct RigidBodyWorld;
struct Scene;
struct ShaderFxData;
struct View3D;
struct ViewLayer;

/* Active modifier. */
void BKE_object_modifier_set_active(struct Object *ob, struct ModifierData *md)
{
    if (md != nullptr) { md->flag |= eModifierFlag_Active; }
}

typedef enum eObjectVisibilityResult {
  OB_VISIBLE_SELF = 1,
  OB_VISIBLE_PARTICLES = 2,
  OB_VISIBLE_INSTANCES = 4,
  OB_VISIBLE_ALL = (OB_VISIBLE_SELF | OB_VISIBLE_PARTICLES | OB_VISIBLE_INSTANCES),
} eObjectVisibilityResult;

typedef struct ObjectTfmProtectedChannels {
  float loc[3], dloc[3];
  float scale[3], dscale[3];
  float rot[3], drot[3];
  float quat[4], dquat[4];
  float rotAxis[3], drotAxis[3];
  float rotAngle, drotAngle;
} ObjectTfmProtectedChannels;

struct Mesh *BKE_object_get_evaluated_mesh(struct Object *object)
{
    Mesh* mesh = nullptr; // BKE_object_get_evaluated_mesh_no_subsurf(object);
    if (!mesh) { return nullptr; }

    //if (object->data && GS(((const ID*)object->data)->name) == ID_ME) {
    //    mesh = BKE_mesh_wrapper_ensure_subdivision(mesh);
    //}

    return mesh;
}
typedef enum eObRelationTypes {
  OB_REL_NONE = 0,                      /* just the selection as is */
  OB_REL_PARENT = (1 << 0),             /* immediate parent */
  OB_REL_PARENT_RECURSIVE = (1 << 1),   /* parents up to root of selection tree*/
  OB_REL_CHILDREN = (1 << 2),           /* immediate children */
  OB_REL_CHILDREN_RECURSIVE = (1 << 3), /* All children */
  OB_REL_MOD_ARMATURE = (1 << 4),       /* Armatures related to the selected objects */
  /* OB_REL_SCENE_CAMERA = (1 << 5), */ /* you might want the scene camera too even if unselected?
                                         */
} eObRelationTypes;

typedef enum eObjectSet {
  OB_SET_SELECTED, /* Selected Objects */
  OB_SET_VISIBLE,  /* Visible Objects  */
  OB_SET_ALL,      /* All Objects      */
} eObjectSet;


void BKE_object_modifiers_lib_link_common(void *userData,
                                          struct Object *ob,
                                          struct ID **idpoin,
                                          int cb_flag)
{
    //BlendLibReader* reader = (BlendLibReader*)userData;
    //if (*idpoin != nullptr && (cb_flag & IDWALK_CB_USER) != 0) {
    //    id_us_plus_no_lib(*idpoin);
    //}
}