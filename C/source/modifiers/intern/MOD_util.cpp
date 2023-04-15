/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2005 Blender Foundation. All rights reserved. */

/** \file
 * \ingroup modifiers
 */

#include <string>
//
//#include "utildefines.h"

//#include "BLI_bitmap.h"
//#include "math_matrix.cuh"
//#include "math_vector.cuh"

//#include "image_types.h"
//#include "mesh_types.h"
#include "meshdata_types.cuh"
//#include "modifier_types.cuh"
//#include "object_types.cuh"
//#include "scene_types.cuh"

//#include "BKE_action.h" /* BKE_pose_channel_find_name */
//#include "BKE_deform.h"
//#include "BKE_editmesh.h"
//#include "BKE_image.h"
//#include "BKE_lattice.h"
//#include "BKE_lib_id.h"
//#include "BKE_mesh.h"
//#include "BKE_mesh_wrapper.h"
//#include "object.h"
//
#include "modifier.h"
//
//#include "DEG_depsgraph.h"
//#include "DEG_depsgraph_query.h"
//
#include "MOD_modifiertypes.h"
//#include "MOD_util.h"
//
//#include "MEM_guardedalloc.cuh"
//
//#include "bmesh.h"

//void MOD_init_texture(MappingInfoModifierData *dmd, const ModifierEvalContext *ctx)
//{
//  Tex *tex = dmd->texture;
//
//  if (tex == NULL) {
//    return;
//  }
//
//  if (tex->ima && BKE_image_is_animated(tex->ima)) {
//    BKE_image_user_frame_calc(tex->ima, &tex->iuser, DEG_get_ctime(ctx->depsgraph));
//  }
//}

/* TODO: to be renamed to get_texture_coords once we are done with moving modifiers to Mesh. */
//void MOD_get_texture_coords(MappingInfoModifierData *dmd,
//                            const ModifierEvalContext *UNUSED(ctx),
//                            Object *ob,
//                            Mesh *mesh,
//                            float (*cos)[3],
//                            float (*r_texco)[3])
//{
//  const int verts_num = mesh->totvert;
//  int i;
//  int texmapping = dmd->texmapping;
//  float mapref_imat[4][4];
//
//  if (texmapping == MOD_DISP_MAP_OBJECT) {
//    if (dmd->map_object != NULL) {
//      Object *map_object = dmd->map_object;
//      if (dmd->map_bone[0] != '\0') {
//        bPoseChannel *pchan = BKE_pose_channel_find_name(map_object->pose, dmd->map_bone);
//        if (pchan) {
//          float mat_bone_world[4][4];
//          mul_m4_m4m4(mat_bone_world, map_object->obmat, pchan->pose_mat);
//          invert_m4_m4(mapref_imat, mat_bone_world);
//        }
//        else {
//          invert_m4_m4(mapref_imat, map_object->obmat);
//        }
//      }
//      else {
//        invert_m4_m4(mapref_imat, map_object->obmat);
//      }
//    }
//    else { /* if there is no map object, default to local */
//      texmapping = MOD_DISP_MAP_LOCAL;
//    }
//  }
//
//  /* UVs need special handling, since they come from faces */
//  if (texmapping == MOD_DISP_MAP_UV) {
//    if (CustomData_has_layer(&mesh->ldata, CD_MLOOPUV)) {
//      MPoly *mpoly = mesh->mpoly;
//      MPoly *mp;
//      MLoop *mloop = mesh->mloop;
//      BLI_bitmap *done = BLI_BITMAP_NEW(verts_num, __func__);
//      const int polys_num = mesh->totpoly;
//      char uvname[MAX_CUSTOMDATA_LAYER_NAME];
//
//      CustomData_validate_layer_name(&mesh->ldata, CD_MLOOPUV, dmd->uvlayer_name, uvname);
//      const MLoopUV *mloop_uv = CustomData_get_layer_named(&mesh->ldata, CD_MLOOPUV, uvname);
//
//      /* verts are given the UV from the first face that uses them */
//      for (i = 0, mp = mpoly; i < polys_num; i++, mp++) {
//        uint fidx = mp->totloop - 1;
//
//        do {
//          uint lidx = mp->loopstart + fidx;
//          uint vidx = mloop[lidx].v;
//
//          if (!BLI_BITMAP_TEST(done, vidx)) {
//            /* remap UVs from [0, 1] to [-1, 1] */
//            r_texco[vidx][0] = (mloop_uv[lidx].uv[0] * 2.0f) - 1.0f;
//            r_texco[vidx][1] = (mloop_uv[lidx].uv[1] * 2.0f) - 1.0f;
//            BLI_BITMAP_ENABLE(done, vidx);
//          }
//
//        } while (fidx--);
//      }
//
//      MEM_freeN(done);
//      return;
//    }
//
//    /* if there are no UVs, default to local */
//    texmapping = MOD_DISP_MAP_LOCAL;
//  }
//
//  MVert *mv = mesh->mvert;
//  for (i = 0; i < verts_num; i++, mv++, r_texco++) {
//    switch (texmapping) {
//      case MOD_DISP_MAP_LOCAL:
//        copy_v3_v3(*r_texco, cos != NULL ? *cos : mv->co);
//        break;
//      case MOD_DISP_MAP_GLOBAL:
//        mul_v3_m4v3(*r_texco, ob->obmat, cos != NULL ? *cos : mv->co);
//        break;
//      case MOD_DISP_MAP_OBJECT:
//        mul_v3_m4v3(*r_texco, ob->obmat, cos != NULL ? *cos : mv->co);
//        mul_m4_v3(mapref_imat, *r_texco);
//        break;
//    }
//    if (cos != NULL) {
//      cos++;
//    }
//  }
//}

//void MOD_previous_vcos_store(ModifierData *md, const float (*vert_coords)[3])
//{
//  while ((md = md->next) && md->type == eModifierType_Armature) {
//    ArmatureModifierData *amd = (ArmatureModifierData *)md;
//    if (amd->multi && amd->vert_coords_prev == NULL) {
//      amd->vert_coords_prev = static_cast<float(*)[3]>(MEM_dupallocN(vert_coords));
//    }
//    else {
//      break;
//    }
//  }
//  /* lattice/mesh modifier too */
//}

//Mesh *MOD_deform_mesh_eval_get(Object *ob,
//                               struct BMEditMesh *em,
//                               Mesh *mesh,
//                               const float (*vertexCos)[3],
//                               const int verts_num,
//                               const bool use_normals,
//                               const bool use_orco)
//{
//  if (mesh != NULL) {
//    /* pass */
//  }
//  else if (ob->type == OB_MESH) {
//    if (em) {
//      mesh = BKE_mesh_wrapper_from_editmesh_with_coords(em, NULL, vertexCos, ob->data);
//    }
//    else {
//      /* TODO(sybren): after modifier conversion of DM to Mesh is done, check whether
//       * we really need a copy here. Maybe the CoW ob->data can be directly used. */
//      Mesh *mesh_prior_modifiers = BKE_object_get_pre_modified_mesh(ob);
//      mesh = (Mesh *)BKE_id_copy_ex(NULL,
//                                    &mesh_prior_modifiers->id,
//                                    NULL,
//                                    (LIB_ID_COPY_LOCALIZE | LIB_ID_COPY_CD_REFERENCE));
//      mesh->runtime->deformed_only = 1;
//    }
//
//    if (em != NULL) {
//      /* pass */
//    }
//    /* TODO(sybren): after modifier conversion of DM to Mesh is done, check whether
//     * we really need vertexCos here. */
//    else if (vertexCos) {
//      BKE_mesh_vert_coords_apply(mesh, vertexCos);
//    }
//
//    if (use_orco) {
//      BKE_mesh_orco_ensure(ob, mesh);
//    }
//  }
//  else if (ELEM(ob->type, OB_FONT, OB_CURVES_LEGACY, OB_SURF)) {
//    /* TODO(sybren): get evaluated mesh from depsgraph once
//     * that's properly generated for curves. */
//    mesh = BKE_mesh_new_nomain_from_curve(ob);
//
//    /* Currently, that may not be the case every time
//     * (texts e.g. tend to give issues,
//     * also when deforming curve points instead of generated curve geometry... ). */
//    if (mesh != NULL && mesh->totvert != verts_num) {
//      BKE_id_free(NULL, mesh);
//      mesh = NULL;
//    }
//  }
//
//  /* TODO: Remove this "use_normals" argument, since the caller should retrieve normals afterwards
//   * if necessary. */
//  if (use_normals) {
//    if (LIKELY(mesh)) {
//      BKE_mesh_vertex_normals_ensure(mesh);
//    }
//  }
//
//  if (mesh && mesh->runtime->wrapper_type == ME_WRAPPER_TYPE_MDATA) {
//    BLI_assert(mesh->totvert == verts_num);
//  }
//
//  return mesh;
//}

//void MOD_get_vgroup(Object *ob, struct Mesh *mesh, const char *name, MDeformVert **dvert, int *defgrp_index)
//{
//  if (mesh) {
//    *defgrp_index = BKE_id_defgroup_name_index(&mesh->id, name);
//    if (*defgrp_index != -1) {
//      *dvert = mesh->dvert;
//    }
//    else {
//      *dvert = NULL;
//    }
//  }
//  else {
//    *defgrp_index = BKE_object_defgroup_name_index(ob, name);
//    if (*defgrp_index != -1 && ob->type == OB_LATTICE) {
//      *dvert = BKE_lattice_deform_verts_get(ob);
//    }
//    else {
//      *dvert = NULL;
//    }
//  }
//}

//void MOD_depsgraph_update_object_bone_relation(struct DepsNodeHandle *node,
//                                               Object *object,
//                                               const char *bonename,
//                                               const char *description)
//{
//  if (object == NULL) {
//    return;
//  }
//  if (bonename[0] != '\0' && object->type == OB_ARMATURE) {
//    DEG_add_object_relation(node, object, DEG_OB_COMP_EVAL_POSE, description);
//  }
//  else {
//    DEG_add_object_relation(node, object, DEG_OB_COMP_TRANSFORM, description);
//  }
//}


void modifier_type_init(ModifierTypeInfo *types[])
{
	types[eModifierType_None] = &modifierType_None;
	types[eModifierType_Curve] = &modifierType_Curve;
	types[eModifierType_Lattice] = &modifierType_Lattice;
	types[eModifierType_Subsurf] = &modifierType_Subsurf;
	types[eModifierType_Build] = &modifierType_Build;
	types[eModifierType_Array] = &modifierType_Array;
	types[eModifierType_Mirror] = &modifierType_Mirror;
	types[eModifierType_EdgeSplit] = &modifierType_EdgeSplit;
	types[eModifierType_Bevel] = &modifierType_Bevel;
	types[eModifierType_Displace] = &modifierType_Displace;
	types[eModifierType_UVProject] = &modifierType_UVProject;
	types[eModifierType_Decimate] = &modifierType_Decimate;
	types[eModifierType_Smooth] = &modifierType_Smooth;
	types[eModifierType_Cast] = &modifierType_Cast;
	types[eModifierType_Wave] = &modifierType_Wave;
	types[eModifierType_Armature] = &modifierType_Armature;
	types[eModifierType_Softbody] = &modifierType_Softbody;
	types[eModifierType_Cloth] = &modifierType_Cloth;
	//types[eModifierType_None] = &modifierType_None;
	//types[eModifierType_None] = &modifierType_None;
	//types[eModifierType_None] = &modifierType_None;
	//types[eModifierType_None] = &modifierType_None;
	//types[eModifierType_None] = &modifierType_None;
	//types[eModifierType_None] = &modifierType_None;
	//types[eModifierType_None] = &modifierType_None;
	//types[eModifierType_None] = &modifierType_None;
	//types[eModifierType_None] = &modifierType_None;
	//types[eModifierType_None] = &modifierType_None;
  //INIT_TYPE(Collision);
  //INIT_TYPE(Boolean);
  //INIT_TYPE(MeshDeform);
  //INIT_TYPE(Ocean);
  //INIT_TYPE(ParticleSystem);
  //INIT_TYPE(ParticleInstance);
  //INIT_TYPE(Explode);
  //INIT_TYPE(Shrinkwrap);
  //INIT_TYPE(Mask);
  //INIT_TYPE(SimpleDeform);
  //INIT_TYPE(Multires);
  //INIT_TYPE(Surface);
  //INIT_TYPE(Fluid);
  //INIT_TYPE(ShapeKey);
  //INIT_TYPE(Solidify);
  //INIT_TYPE(Screw);
  //INIT_TYPE(Warp);
  //INIT_TYPE(WeightVGEdit);
  //INIT_TYPE(WeightVGMix);
  //INIT_TYPE(WeightVGProximity);
  //INIT_TYPE(DynamicPaint);
  //INIT_TYPE(Remesh);
  //INIT_TYPE(Skin);
  //INIT_TYPE(LaplacianSmooth);
  //INIT_TYPE(Triangulate);
  //INIT_TYPE(UVWarp);
  //INIT_TYPE(MeshCache);
  //INIT_TYPE(LaplacianDeform);
  //INIT_TYPE(Wireframe);
  //INIT_TYPE(Weld);
  //INIT_TYPE(DataTransfer);
  //INIT_TYPE(NormalEdit);
  //INIT_TYPE(CorrectiveSmooth);
  //INIT_TYPE(MeshSequenceCache);
  //INIT_TYPE(SurfaceDeform);
  //INIT_TYPE(WeightedNormal);
  //INIT_TYPE(MeshToVolume);
  //INIT_TYPE(VolumeDisplace);
  //INIT_TYPE(VolumeToMesh);
  //INIT_TYPE(Nodes);
#undef INIT_TYPE
}
