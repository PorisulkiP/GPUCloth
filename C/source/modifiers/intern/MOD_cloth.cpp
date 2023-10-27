#include "utildefines.h"

#include "listbase.cuh"

#include "BLT_translation.h"

#include "cloth_types.cuh"
#include "defaults.cuh"
#include "modifier_types.cuh"
#include "mesh_types.h"
#include "object_force_types.cuh"
#include "object_types.cuh"
#include "scene_types.cuh"
#include "DNA_screen_types.h"

#include "MEM_guardedalloc.cuh"

#include "cloth.h"
//#include "BKE_context.h"
//#include "BKE_effect.h"
//#include "global.h"
//#include "BKE_key.h"
#include "BKE_lib_id.h"
#include "lib_query.h"
//#include "BKE_mesh.h"
#include "modifier.h"
//#include "pointcache.cuh"
//#include "BKE_screen.h"

//#include "UI_interface.h"
//#include "UI_resources.h"

#include "RNA_access.h"
//#include "RNA_prototypes.h"

#include "DEG_depsgraph_physics.h"
#include "DEG_depsgraph_query.cuh"

//#include "MOD_ui_common.h"
#include "MOD_util.h"

PointCache* BKE_ptcache_add(ListBase* ptcaches)
{
    PointCache* cache;

    cache = static_cast<PointCache*>(MEM_lockfree_callocN(sizeof(PointCache), "PointCache"));
    cache->startframe = 1;
    cache->endframe = 250;
    cache->step = 1;
    cache->index = -1;

    BLI_addtail(ptcaches, cache);

    return cache;
}

EffectorWeights* BKE_effector_add_weights(Collection* collection)
{
	auto* weights = static_cast<EffectorWeights*>(MEM_lockfree_callocN(sizeof(EffectorWeights), "EffectorWeights"));
    for (int i = 0; i < NUM_PFIELD_TYPES; i++) {
        weights->weight[i] = 1.0f;
    }

    weights->global_gravity = 1.0f;

    //weights->group = collection;

    return weights;
}


static void initData(ModifierData *md)
{
	auto*clmd = static_cast<ClothModifierData*>(md);
    auto tmp = static_cast<const ClothModifierData*>(DNA_default_table[_SDNA_TYPE_ClothModifierData]);
    MEMCPY_STRUCT_AFTER(clmd, tmp, ClothModifierData::modifier);
    clmd->sim_parms = (ClothSimSettings*)_DNA_struct_default_alloc_impl(static_cast<const uint8_t*>(DNA_default_table[_SDNA_TYPE_ClothSimSettings]), sizeof(ClothSimSettings), "");
    clmd->coll_parms = (ClothCollSettings*)_DNA_struct_default_alloc_impl(static_cast<const uint8_t*>(DNA_default_table[_SDNA_TYPE_ClothCollSettings]), sizeof(ClothCollSettings), "");

    clmd->point_cache = BKE_ptcache_add(&clmd->ptcaches);

    /* check for alloc failing */
    if (!clmd->sim_parms || !clmd->coll_parms || !clmd->point_cache) 
    {
        return;
    }

    if (!clmd->sim_parms->effector_weights) 
    {
        clmd->sim_parms->effector_weights = BKE_effector_add_weights(nullptr);
    }

    if (clmd->point_cache) 
    {
        clmd->point_cache->step = 1;
    }
}

static void deformVerts(ModifierData *md,
                        const ModifierEvalContext *ctx,
                        Mesh *mesh,
                        float (*vertexCos)[3],
                        int verts_num)
{
  Mesh *mesh_src;
  auto*clmd = static_cast<ClothModifierData*>(md);
  Scene *scene = DEG_get_evaluated_scene(ctx->depsgraph);

  /* check for alloc failing */
  if (!clmd->sim_parms || !clmd->coll_parms) {
    initData(md);

    if (!clmd->sim_parms || !clmd->coll_parms) {
      return;
    }
  }

  //if (mesh == NULL) 
  //{
  //  mesh_src = MOD_deform_mesh_eval_get(ctx->object, NULL, NULL, NULL, verts_num, false, false);
  //}
  //else {
  //  /* Not possible to use get_mesh() in this case as we'll modify its vertices
  //   * and get_mesh() would return 'mesh' directly. */
  //  mesh_src = (Mesh *)BKE_id_copy_ex(NULL, (ID *)mesh, NULL, LIB_ID_COPY_LOCALIZE);
  //}

  ///* TODO(sergey): For now it actually duplicates logic from DerivedMesh.cc
  // * and needs some more generic solution. But starting experimenting with
  // * this so close to the release is not that nice..
  // *
  // * Also hopefully new cloth system will arrive soon..
  // */
  //if (mesh == NULL && clmd->sim_parms->shapekey_rest) 
  //{
  //  KeyBlock *kb = BKE_keyblock_from_key(BKE_key_from_object(ctx->object),
  //                                       clmd->sim_parms->shapekey_rest);
  //  if (kb && kb->data != NULL) {
  //    float(*layerorco)[3];
  //    if (!(layerorco = CustomData_get_layer(&mesh_src->vdata, CD_CLOTH_ORCO))) {
  //      layerorco = CustomData_add_layer(
  //          &mesh_src->vdata, CD_CLOTH_ORCO, CD_SET_DEFAULT, NULL, mesh_src->totvert);
  //    }

  //    memcpy(layerorco, kb->data, sizeof(float[3]) * verts_num);
  //  }
  //}

  //BKE_mesh_vert_coords_apply(mesh_src, vertexCos);

  //clothModifier_do(clmd, ctx->depsgraph, scene, ctx->object, mesh_src, vertexCos);

  BKE_id_free(nullptr, mesh_src);
}

static void updateDepsgraph(ModifierData *md, const ModifierUpdateDepsgraphContext *ctx)
{
  //ClothModifierData *clmd = (ClothModifierData *)md;
  //if (clmd != NULL) {
  //  if (clmd->coll_parms->flags & CLOTH_COLLSETTINGS_FLAG_ENABLED) {
  //    DEG_add_collision_relations(ctx->node,
  //                                ctx->object,
  //                                clmd->coll_parms->group,
  //                                eModifierType_Collision,
  //                                NULL,
  //                                "Cloth Collision");
  //  }
  //  DEG_add_forcefield_relations(ctx->node, ctx->object, clmd->sim_parms->effector_weights, true, 0, "Cloth Field");
  //}
  //DEG_add_modifier_to_transform_relation(ctx->node, "Cloth Modifier");
}

static void requiredDataMask(Object *UNUSED(ob),
                             ModifierData *md,
                             CustomData_MeshMasks *r_cddata_masks)
{
	auto*clmd = static_cast<ClothModifierData*>(md);

  if (cloth_uses_vgroup(clmd)) 
  {
    r_cddata_masks->vmask |= CD_MASK_MDEFORMVERT;
  }

  if (clmd->sim_parms->shapekey_rest != 0) 
  {
    r_cddata_masks->vmask |= CD_MASK_CLOTH_ORCO;
  }
}

static void copyData(const ModifierData *md, ModifierData *target, const int flag)
{
  const auto*clmd = static_cast<const ClothModifierData*>(md);
  auto*tclmd = static_cast<ClothModifierData*>(target);

  if (tclmd->sim_parms) 
  {
    if (tclmd->sim_parms->effector_weights) 
    {
      MEM_lockfree_freeN(tclmd->sim_parms->effector_weights);
    }
    MEM_lockfree_freeN(tclmd->sim_parms);
  }

  if (tclmd->coll_parms) {
    MEM_lockfree_freeN(tclmd->coll_parms);
  }

  //BKE_ptcache_free_list(&tclmd->ptcaches);
  if (flag & LIB_ID_COPY_SET_COPIED_ON_WRITE) 
  {
    /* Share the cache with the original object's modifier. */
    tclmd->modifier.flag |= eModifierFlag_SharedCaches;
    tclmd->ptcaches = clmd->ptcaches;
    tclmd->point_cache = clmd->point_cache;
  }
  else {
    const int clmd_point_cache_index = BLI_findindex(&clmd->ptcaches, clmd->point_cache);
    //BKE_ptcache_copy_list(&tclmd->ptcaches, &clmd->ptcaches, flag);
    tclmd->point_cache = static_cast<PointCache*>(BLI_findlink(&tclmd->ptcaches, clmd_point_cache_index));
  }

  tclmd->sim_parms = static_cast<ClothSimSettings*>(MEM_lockfree_dupallocN(clmd->sim_parms));
  if (clmd->sim_parms->effector_weights) 
  {
    tclmd->sim_parms->effector_weights = static_cast<EffectorWeights*>(MEM_lockfree_dupallocN(clmd->sim_parms->effector_weights));
  }
  tclmd->coll_parms = static_cast<ClothCollSettings*>(MEM_lockfree_dupallocN(clmd->coll_parms));
  tclmd->clothObject = nullptr;
  //tclmd->hairdata = NULL;
  tclmd->solver_result = nullptr;
}

static bool dependsOnTime(struct ModifierData* md)
{
  return true;
}

static void freeData(ModifierData *md)
{
	auto* clmd = static_cast<ClothModifierData*>(md);

	if (clmd) 
	{
	cloth_free_modifier_extern(clmd);

	if (clmd->sim_parms) 
	{
	  if (clmd->sim_parms->effector_weights) 
	  {
	    MEM_lockfree_freeN(clmd->sim_parms->effector_weights);
	  }
	  MEM_lockfree_freeN(clmd->sim_parms);
	}
	if (clmd->coll_parms) 
	{
	  MEM_lockfree_freeN(clmd->coll_parms);
	}

	if (md->flag & eModifierFlag_SharedCaches) 
	{
	  BLI_listbase_clear(&clmd->ptcaches);
	}
	else 
	{
	  //BKE_ptcache_free_list(&clmd->ptcaches);
	}
	clmd->point_cache = nullptr;

	//if (clmd->hairdata) 
	//{
	//  MEM_lockfree_freeN(clmd->hairdata);
	//}

	if (clmd->solver_result) 
	{
	  MEM_lockfree_freeN(clmd->solver_result);
	}
	}
}

static void foreachIDLink(ModifierData *md, Object *ob, IDWalkFunc walk, void *userData)
{
	const auto* clmd = reinterpret_cast<ClothModifierData*>(md);

	if (clmd->coll_parms) {
	walk(userData, ob, reinterpret_cast<ID**>(&clmd->coll_parms->group), IDWALK_CB_NOP);
	}

	//if (clmd->sim_parms && clmd->sim_parms->effector_weights) {
	//walk(userData, ob, reinterpret_cast<ID**>(&clmd->sim_parms->effector_weights->group), IDWALK_CB_USER);
	//}
}

static void panel_draw(const bContext *UNUSED(C), Panel *panel)
{
//  uiLayout *layout = panel->layout;
//
//  PointerRNA *ptr = modifier_panel_get_property_pointers(panel, NULL);
//
//  uiItemL(layout, TIP_("Settings are inside the Physics tab"), ICON_NONE);
//
//  modifier_panel_end(layout, ptr);
}

static void panelRegister(ARegionType *region_type)
{
  //modifier_panel_register(region_type, eModifierType_Cloth, panel_draw);
}

ModifierTypeInfo modifierType_Cloth = 
{
    /* name */ N_("Cloth"),
    /* structName */ "ClothModifierData",
    /* structSize */ sizeof(ClothModifierData),
    /* srna */ nullptr,
    /* type */ (ModifierTypeType)eModifierTypeType_OnlyDeform,
    /* flags */ static_cast<ModifierTypeFlag>(eModifierTypeFlag_AcceptsMesh | eModifierTypeFlag_UsesPointCache |
	    eModifierTypeFlag_Single),
    /* icon */ NULL,

    /* copyData */ copyData,

    /* deformVerts */ deformVerts,
    /* deformMatrices */ nullptr,
    /* deformVertsEM */ nullptr,
    /* deformMatricesEM */ nullptr,
    /* modifyMesh */ nullptr,
    /* modifyHair */ nullptr,
    /* modifyGeometrySet */ nullptr,
    /* modifyVolume */ nullptr,

    /* initData */ initData,
    /* requiredDataMask */ requiredDataMask,
    /* freeData */ freeData,
    /* isDisabled */ nullptr,
    /* updateDepsgraph */ updateDepsgraph,
    /* dependsOnTime */ dependsOnTime,
    /* dependsOnNormals */ nullptr,
    /* foreachIDLink */ foreachIDLink,
    /* foreachTexLink */ nullptr,
    /* freeRuntimeData */ nullptr,
    /* panelRegister */ panelRegister,
    /* blendWrite */ nullptr,
    /* blendRead */ nullptr,
};
