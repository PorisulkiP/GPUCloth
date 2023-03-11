#include "boid_types.h"
#include "cloth_types.cuh"
#include "material_types.h"
#include "mesh_types.h"
#include "meshdata_types.cuh"
#include "modifier_types.cuh"
#include "object_force_types.cuh"
#include "object_types.cuh"
#include "particle_types.h"
#include "scene_types.cuh"
#include "types.h"

#include "define.h"
#include "enum_types.h"

#include "BKE_mesh.h"

#include "listbase.h"

#include "internal.h"

#ifdef RNA_RUNTIME
static const EnumPropertyItem part_from_items[] = {
    {PART_FROM_VERT, "VERT", 0, "Vertices", ""},
    {PART_FROM_FACE, "FACE", 0, "Faces", ""},
    {PART_FROM_VOLUME, "VOLUME", 0, "Volume", ""},
    {0, NULL, 0, NULL, NULL},
};
#endif

#ifndef RNA_RUNTIME
static const EnumPropertyItem part_reactor_from_items[] = {
    {PART_FROM_VERT, "VERT", 0, "Vertices", ""},
    {PART_FROM_FACE, "FACE", 0, "Faces", ""},
    {PART_FROM_VOLUME, "VOLUME", 0, "Volume", ""},
    {0, NULL, 0, NULL, NULL},
};
#endif

static const EnumPropertyItem part_dist_items[] = {
    {PART_DISTR_JIT, "JIT", 0, "Jittered", ""},
    {PART_DISTR_RAND, "RAND", 0, "Random", ""},
    {PART_DISTR_GRID, "GRID", 0, "Grid", ""},
    {0, NULL, 0, NULL, NULL},
};

#ifdef RNA_RUNTIME
static const EnumPropertyItem part_hair_dist_items[] = {
    {PART_DISTR_JIT, "JIT", 0, "Jittered", ""},
    {PART_DISTR_RAND, "RAND", 0, "Random", ""},
    {0, NULL, 0, NULL, NULL},
};
#endif

static const EnumPropertyItem part_draw_as_items[] = {
    {PART_DRAW_NOT, "NONE", 0, "None", ""},
    {PART_DRAW_REND, "RENDER", 0, "Rendered", ""},
    {PART_DRAW_DOT, "DOT", 0, "Point", ""},
    {PART_DRAW_CIRC, "CIRC", 0, "Circle", ""},
    {PART_DRAW_CROSS, "CROSS", 0, "Cross", ""},
    {PART_DRAW_AXIS, "AXIS", 0, "Axis", ""},
    {0, NULL, 0, NULL, NULL},
};

#ifdef RNA_RUNTIME
static const EnumPropertyItem part_hair_draw_as_items[] = {
    {PART_DRAW_NOT, "NONE", 0, "None", ""},
    {PART_DRAW_REND, "RENDER", 0, "Rendered", ""},
    {PART_DRAW_PATH, "PATH", 0, "Path", ""},
    {0, NULL, 0, NULL, NULL},
};
#endif

static const EnumPropertyItem part_ren_as_items[] = {
    {PART_DRAW_NOT, "NONE", 0, "None", ""},
    {PART_DRAW_HALO, "HALO", 0, "Halo", ""},
    {PART_DRAW_LINE, "LINE", 0, "Line", ""},
    {PART_DRAW_PATH, "PATH", 0, "Path", ""},
    {PART_DRAW_OB, "OBJECT", 0, "Object", ""},
    {PART_DRAW_GR, "COLLECTION", 0, "Collection", ""},
    {0, NULL, 0, NULL, NULL},
};

#ifdef RNA_RUNTIME
static const EnumPropertyItem part_hair_ren_as_items[] = {
    {PART_DRAW_NOT, "NONE", 0, "None", ""},
    {PART_DRAW_PATH, "PATH", 0, "Path", ""},
    {PART_DRAW_OB, "OBJECT", 0, "Object", ""},
    {PART_DRAW_GR, "COLLECTION", 0, "Collection", ""},
    {0, NULL, 0, NULL, NULL},
};
#endif

static const EnumPropertyItem part_type_items[] = {
    {PART_EMITTER, "EMITTER", 0, "Emitter", ""},
    /*{PART_REACTOR, "REACTOR", 0, "Reactor", ""}, */
    {PART_HAIR, "HAIR", 0, "Hair", ""},
    {0, NULL, 0, NULL, NULL},
};

#ifdef RNA_RUNTIME
static const EnumPropertyItem part_fluid_type_items[] = {
    {PART_FLUID, "FLUID", 0, "Fluid", ""},
    {PART_FLUID_FLIP, "FLIP", 0, "Liquid", ""},
    {PART_FLUID_SPRAY, "SPRAY", 0, "Spray", ""},
    {PART_FLUID_BUBBLE, "BUBBLE", 0, "Bubble", ""},
    {PART_FLUID_FOAM, "FOAM", 0, "Foam", ""},
    {PART_FLUID_TRACER, "TRACER", 0, "Tracer", ""},
    {PART_FLUID_SPRAYFOAM, "SPRAYFOAM", 0, "Spray-Foam", ""},
    {PART_FLUID_SPRAYBUBBLE, "SPRAYBUBBLE", 0, "Spray-Bubble", ""},
    {PART_FLUID_FOAMBUBBLE, "FOAMBUBBLE", 0, "Foam-Bubble", ""},
    {PART_FLUID_SPRAYFOAMBUBBLE, "SPRAYFOAMBUBBLE", 0, "Spray-Foam-Bubble", ""},
    {0, NULL, 0, NULL, NULL},
};
#endif

#ifdef RNA_RUNTIME
//
//#  include "BLI_math.h"
//#  include "BLI_string_utils.h"
//
//#  include "boids.h"
//#  include "BKE_cloth.h"
//#  include "BKE_colortools.h"
//#  include "BKE_context.h"
//#  include "BKE_deform.h"
//#  include "BKE_effect.h"
//#  include "BKE_material.h"
//#  include "BKE_modifier.h"
//#  include "BKE_particle.h"
//#  include "BKE_pointcache.h"
//#  include "BKE_texture.h"
//
//#  include "DEG_depsgraph.h"
//#  include "DEG_depsgraph_build.h"
//
///* use for object space hair get/set */
//static void rna_ParticleHairKey_location_object_info(PointerRNA *ptr,
//                                                     ParticleSystemModifierData **psmd_pt,
//                                                     ParticleData **pa_pt)
//{
//  HairKey *hkey = (HairKey *)ptr->data;
//  Object *ob = (Object *)ptr->owner_id;
//  ModifierData *md;
//  ParticleSystemModifierData *psmd = NULL;
//  ParticleSystem *psys;
//  ParticleData *pa;
//  int i;
//
//  *psmd_pt = NULL;
//  *pa_pt = NULL;
//
//  /* given the pointer HairKey *hkey, we iterate over all particles in all
//   * particle systems in the object "ob" in order to find
//   * - the ParticleSystemData to which the HairKey (and hence the particle)
//   *   belongs (will be stored in psmd_pt)
//   * - the ParticleData to which the HairKey belongs (will be stored in pa_pt)
//   *
//   * not a very efficient way of getting hair key location data,
//   * but it's the best we've got at the present
//   *
//   * IDEAS: include additional information in PointerRNA beforehand,
//   * for example a pointer to the ParticleSystemModifierData to which the
//   * hair-key belongs.
//   */
//
//  for (md = ob->modifiers.first; md; md = md->next) {
//    if (md->type == eModifierType_ParticleSystem) {
//      psmd = (ParticleSystemModifierData *)md;
//      if (psmd && psmd->mesh_final && psmd->psys) {
//        psys = psmd->psys;
//        for (i = 0, pa = psys->particles; i < psys->totpart; i++, pa++) {
//          /* Hair-keys are stored sequentially in memory, so we can
//           * find if it's the same particle by comparing pointers,
//           * without having to iterate over them all. */
//          if ((hkey >= pa->hair) && (hkey < pa->hair + pa->totkey)) {
//            *psmd_pt = psmd;
//            *pa_pt = pa;
//            return;
//          }
//        }
//      }
//    }
//  }
//}
//
//static void rna_ParticleHairKey_location_object_get(PointerRNA *ptr, float *values)
//{
//  HairKey *hkey = (HairKey *)ptr->data;
//  Object *ob = (Object *)ptr->owner_id;
//  ParticleSystemModifierData *psmd;
//  ParticleData *pa;
//
//  rna_ParticleHairKey_location_object_info(ptr, &psmd, &pa);
//
//  if (pa) {
//    Mesh *hair_mesh = (psmd->psys->flag & PSYS_HAIR_DYNAMICS) ? psmd->psys->hair_out_mesh : NULL;
//
//    if (hair_mesh) {
//      MVert *mvert = &hair_mesh->mvert[pa->hair_index + (hkey - pa->hair)];
//      copy_v3_v3(values, mvert->co);
//    }
//    else {
//      float hairmat[4][4];
//      psys_mat_hair_to_object(ob, psmd->mesh_final, psmd->psys->part->from, pa, hairmat);
//      copy_v3_v3(values, hkey->co);
//      mul_m4_v3(hairmat, values);
//    }
//  }
//  else {
//    zero_v3(values);
//  }
//}
//
///* Helper function which returns index of the given hair_key in particle which owns it.
// * Works with cases when hair_key is coming from the particle which was passed here, and from the
// * original particle of the given one.
// *
// * Such trickery is needed to allow modification of hair keys in the original object using
// * evaluated particle and object to access proper hair matrix. */
//static int hair_key_index_get(const Object *object,
//                              /*const*/ HairKey *hair_key,
//                              /*const*/ ParticleSystemModifierData *modifier,
//                              /*const*/ ParticleData *particle)
//{
//  if (ARRAY_HAS_ITEM(hair_key, particle->hair, particle->totkey)) {
//    return hair_key - particle->hair;
//  }
//
//  const ParticleSystem *particle_system = modifier->psys;
//  const int particle_index = particle - particle_system->particles;
//
//  const ParticleSystemModifierData *original_modifier = (ParticleSystemModifierData *)
//      BKE_modifier_get_original(object, &modifier->modifier);
//  const ParticleSystem *original_particle_system = original_modifier->psys;
//  const ParticleData *original_particle = &original_particle_system->particles[particle_index];
//
//  if (ARRAY_HAS_ITEM(hair_key, original_particle->hair, original_particle->totkey)) {
//    return hair_key - original_particle->hair;
//  }
//
//  return -1;
//}
//
///* Set hair_key->co to the given coordinate in object space (the given coordinate will be
// * converted to the proper space).
// *
// * The hair_key can be coming from both original and evaluated object. Object, modifier and
// * particle are to be from evaluated object, so that all the data needed for hair matrix is
// * present. */
//static void hair_key_location_object_set(HairKey *hair_key,
//                                         Object *object,
//                                         ParticleSystemModifierData *modifier,
//                                         ParticleData *particle,
//                                         const float src_co[3])
//{
//  Mesh *hair_mesh = (modifier->psys->flag & PSYS_HAIR_DYNAMICS) ? modifier->psys->hair_out_mesh :
//                                                                  NULL;
//
//  if (hair_mesh != NULL) {
//    const int hair_key_index = hair_key_index_get(object, hair_key, modifier, particle);
//    if (hair_key_index == -1) {
//      return;
//    }
//
//    MVert *mvert = &hair_mesh->mvert[particle->hair_index + (hair_key_index)];
//    copy_v3_v3(mvert->co, src_co);
//    return;
//  }
//
//  float hairmat[4][4];
//  psys_mat_hair_to_object(
//      object, modifier->mesh_final, modifier->psys->part->from, particle, hairmat);
//
//  float imat[4][4];
//  invert_m4_m4(imat, hairmat);
//
//  copy_v3_v3(hair_key->co, src_co);
//  mul_m4_v3(imat, hair_key->co);
//}
//
//static void rna_ParticleHairKey_location_object_set(PointerRNA *ptr, const float *values)
//{
//  HairKey *hkey = (HairKey *)ptr->data;
//  Object *ob = (Object *)ptr->owner_id;
//
//  ParticleSystemModifierData *psmd;
//  ParticleData *pa;
//  rna_ParticleHairKey_location_object_info(ptr, &psmd, &pa);
//
//  if (pa == NULL) {
//    zero_v3(hkey->co);
//    return;
//  }
//
//  hair_key_location_object_set(hkey, ob, psmd, pa, values);
//}
//
//static void rna_ParticleHairKey_co_object(HairKey *hairkey,
//                                          Object *object,
//                                          ParticleSystemModifierData *modifier,
//                                          ParticleData *particle,
//                                          float n_co[3])
//{
//
//  Mesh *hair_mesh = (modifier->psys->flag & PSYS_HAIR_DYNAMICS) ? modifier->psys->hair_out_mesh :
//                                                                  NULL;
//  if (particle) {
//    if (hair_mesh) {
//      MVert *mvert = &hair_mesh->mvert[particle->hair_index + (hairkey - particle->hair)];
//      copy_v3_v3(n_co, mvert->co);
//    }
//    else {
//      float hairmat[4][4];
//      psys_mat_hair_to_object(
//          object, modifier->mesh_final, modifier->psys->part->from, particle, hairmat);
//      copy_v3_v3(n_co, hairkey->co);
//      mul_m4_v3(hairmat, n_co);
//    }
//  }
//  else {
//    zero_v3(n_co);
//  }
//}
//
//static void rna_ParticleHairKey_co_object_set(ID *id,
//                                              HairKey *hair_key,
//                                              Object *object,
//                                              ParticleSystemModifierData *modifier,
//                                              ParticleData *particle,
//                                              float co[3])
//{
//
//  if (particle == NULL) {
//    return;
//  }
//
//  /* Mark particle system as edited, so then particle_system_update() does not reset the hair
//   * keys from path. This behavior is similar to how particle edit mode sets flags. */
//  ParticleSystemModifierData *orig_modifier = (ParticleSystemModifierData *)
//      BKE_modifier_get_original(object, &modifier->modifier);
//  orig_modifier->psys->flag |= PSYS_EDITED;
//
//  hair_key_location_object_set(hair_key, object, modifier, particle, co);
//
//  /* Tag similar to brushes in particle edit mode, so the modifier stack is properly evaluated
//   * with the same particle system recalc flags as during combing. */
//  DEG_id_tag_update(id, ID_RECALC_GEOMETRY | ID_RECALC_PSYS_REDO);
//}
//
//static void rna_Particle_uv_on_emitter(ParticleData *particle,
//                                       ReportList *reports,
//                                       ParticleSystemModifierData *modifier,
//                                       float r_uv[2])
//{
//#  if 0
//  psys_particle_on_emitter(
//      psmd, part->from, pa->num, pa->num_dmcache, pa->fuv, pa->foffset, co, nor, 0, 0, sd.orco, 0);
//#  endif
//
//  if (modifier->mesh_final == NULL) {
//    BKE_report(reports, RPT_ERROR, "uv_on_emitter() requires a modifier from an evaluated object");
//    return;
//  }
//
//  /* get uvco & mcol */
//  int num = particle->num_dmcache;
//  int from = modifier->psys->part->from;
//
//  if (!CustomData_has_layer(&modifier->mesh_final->ldata, CD_MLOOPUV)) {
//    BKE_report(reports, RPT_ERROR, "Mesh has no UV data");
//    return;
//  }
//  BKE_mesh_tessface_ensure(modifier->mesh_final); /* BMESH - UNTIL MODIFIER IS UPDATED FOR MPoly */
//
//  if (ELEM(num, DMCACHE_NOTFOUND, DMCACHE_ISCHILD)) {
//    if (particle->num < modifier->mesh_final->totface) {
//      num = particle->num;
//    }
//  }
//
//  /* get uvco */
//  if (r_uv && ELEM(from, PART_FROM_FACE, PART_FROM_VOLUME) &&
//      !ELEM(num, DMCACHE_NOTFOUND, DMCACHE_ISCHILD)) {
//    MFace *mface;
//    MTFace *mtface;
//
//    mface = modifier->mesh_final->mface;
//    mtface = modifier->mesh_final->mtface;
//
//    if (mface && mtface) {
//      mtface += num;
//      psys_interpolate_uvs(mtface, mface->v4, particle->fuv, r_uv);
//      return;
//    }
//  }
//
//  r_uv[0] = 0.0f;
//  r_uv[1] = 0.0f;
//}
//
//static void rna_ParticleSystem_co_hair(
//    ParticleSystem *particlesystem, Object *object, int particle_no, int step, float n_co[3])
//{
//  ParticleSettings *part = NULL;
//  ParticleData *pars = NULL;
//  ParticleCacheKey *cache = NULL;
//  int totchild = 0;
//  int totpart;
//  int max_k = 0;
//
//  if (particlesystem == NULL) {
//    return;
//  }
//
//  part = particlesystem->part;
//  pars = particlesystem->particles;
//  totpart = particlesystem->totcached;
//  totchild = particlesystem->totchildcache;
//
//  if (part == NULL || pars == NULL) {
//    return;
//  }
//
//  if (ELEM(part->ren_as, PART_DRAW_OB, PART_DRAW_GR, PART_DRAW_NOT)) {
//    return;
//  }
//
//  /* can happen for disconnected/global hair */
//  if (part->type == PART_HAIR && !particlesystem->childcache) {
//    totchild = 0;
//  }
//
//  if (particle_no < totpart && particlesystem->pathcache) {
//    cache = particlesystem->pathcache[particle_no];
//    max_k = (int)cache->segments;
//  }
//  else if (particle_no < totpart + totchild && particlesystem->childcache) {
//    cache = particlesystem->childcache[particle_no - totpart];
//
//    if (cache->segments < 0) {
//      max_k = 0;
//    }
//    else {
//      max_k = (int)cache->segments;
//    }
//  }
//  else {
//    return;
//  }
//
//  /* Strands key loop data stored in cache + step->co. */
//  if (step >= 0 && step <= max_k) {
//    copy_v3_v3(n_co, (cache + step)->co);
//    mul_m4_v3(particlesystem->imat, n_co);
//    mul_m4_v3(object->obmat, n_co);
//  }
//}
//
//static const EnumPropertyItem *rna_Particle_Material_itemf(bContext *C,
//                                                           PointerRNA *UNUSED(ptr),
//                                                           PropertyRNA *UNUSED(prop),
//                                                           bool *r_free)
//{
//  Object *ob = CTX_data_pointer_get(C, "object").data;
//  Material *ma;
//  EnumPropertyItem *item = NULL;
//  EnumPropertyItem tmp = {0, "", 0, "", ""};
//  int totitem = 0;
//  int i;
//
//  if (ob && ob->totcol > 0) {
//    for (i = 1; i <= ob->totcol; i++) {
//      ma = BKE_object_material_get(ob, i);
//      tmp.value = i;
//      tmp.icon = ICON_MATERIAL_DATA;
//      if (ma) {
//        tmp.name = ma->id.name + 2;
//        tmp.identifier = tmp.name;
//      }
//      else {
//        tmp.name = "Default Material";
//        tmp.identifier = tmp.name;
//      }
//      RNA_enum_item_add(&item, &totitem, &tmp);
//    }
//  }
//  else {
//    tmp.value = 1;
//    tmp.icon = ICON_MATERIAL_DATA;
//    tmp.name = "Default Material";
//    tmp.identifier = tmp.name;
//    RNA_enum_item_add(&item, &totitem, &tmp);
//  }
//
//  RNA_enum_item_end(&item, &totitem);
//  *r_free = true;
//
//  return item;
//}
//
///* return < 0 means invalid (no matching tessellated face could be found). */
//static int rna_ParticleSystem_tessfaceidx_on_emitter(ParticleSystem *particlesystem,
//                                                     ParticleSystemModifierData *modifier,
//                                                     ParticleData *particle,
//                                                     int particle_no,
//                                                     float (**r_fuv)[4])
//{
//  ParticleSettings *part = NULL;
//  int totpart;
//  int totchild = 0;
//  int totface;
//  int totvert;
//  int num = -1;
//
//  BKE_mesh_tessface_ensure(modifier->mesh_final); /* BMESH - UNTIL MODIFIER IS UPDATED FOR MPoly */
//  totface = modifier->mesh_final->totface;
//  totvert = modifier->mesh_final->totvert;
//
//  /* 1. check that everything is ok & updated */
//  if (!particlesystem || !totface) {
//    return num;
//  }
//
//  part = particlesystem->part;
//  /* NOTE: only hair, keyed and baked particles may have cached items... */
//  totpart = particlesystem->totcached != 0 ? particlesystem->totcached : particlesystem->totpart;
//  totchild = particlesystem->totchildcache != 0 ? particlesystem->totchildcache :
//                                                  particlesystem->totchild;
//
//  /* can happen for disconnected/global hair */
//  if (part->type == PART_HAIR && !particlesystem->childcache) {
//    totchild = 0;
//  }
//
//  if (particle_no >= totpart + totchild) {
//    return num;
//  }
//
//  /* 2. get matching face index. */
//  if (particle_no < totpart) {
//    num = (ELEM(particle->num_dmcache, DMCACHE_ISCHILD, DMCACHE_NOTFOUND)) ? particle->num :
//                                                                             particle->num_dmcache;
//
//    if (ELEM(part->from, PART_FROM_FACE, PART_FROM_VOLUME)) {
//      if (num != DMCACHE_NOTFOUND && num < totface) {
//        *r_fuv = &particle->fuv;
//        return num;
//      }
//    }
//    else if (part->from == PART_FROM_VERT) {
//      if (num != DMCACHE_NOTFOUND && num < totvert) {
//        MFace *mface = modifier->mesh_final->mface;
//
//        *r_fuv = &particle->fuv;
//
//        /* This finds the first face to contain the emitting vertex,
//         * this is not ideal, but is mostly fine as UV seams generally
//         * map to equal-colored parts of a texture */
//        for (int i = 0; i < totface; i++, mface++) {
//          if (ELEM(num, mface->v1, mface->v2, mface->v3, mface->v4)) {
//            return i;
//          }
//        }
//      }
//    }
//  }
//  else {
//    ChildParticle *cpa = particlesystem->child + particle_no - totpart;
//    num = cpa->num;
//
//    if (part->childtype == PART_CHILD_FACES) {
//      if (ELEM(part->from, PART_FROM_FACE, PART_FROM_VOLUME, PART_FROM_VERT)) {
//        if (num != DMCACHE_NOTFOUND && num < totface) {
//          *r_fuv = &cpa->fuv;
//          return num;
//        }
//      }
//    }
//    else {
//      ParticleData *parent = particlesystem->particles + cpa->parent;
//      num = parent->num_dmcache;
//
//      if (num == DMCACHE_NOTFOUND) {
//        num = parent->num;
//      }
//
//      if (ELEM(part->from, PART_FROM_FACE, PART_FROM_VOLUME)) {
//        if (num != DMCACHE_NOTFOUND && num < totface) {
//          *r_fuv = &parent->fuv;
//          return num;
//        }
//      }
//      else if (part->from == PART_FROM_VERT) {
//        if (num != DMCACHE_NOTFOUND && num < totvert) {
//          MFace *mface = modifier->mesh_final->mface;
//
//          *r_fuv = &parent->fuv;
//
//          /* This finds the first face to contain the emitting vertex,
//           * this is not ideal, but is mostly fine as UV seams generally
//           * map to equal-colored parts of a texture */
//          for (int i = 0; i < totface; i++, mface++) {
//            if (ELEM(num, mface->v1, mface->v2, mface->v3, mface->v4)) {
//              return i;
//            }
//          }
//        }
//      }
//    }
//  }
//
//  return -1;
//}
//
//static void rna_ParticleSystem_uv_on_emitter(ParticleSystem *particlesystem,
//                                             ReportList *reports,
//                                             ParticleSystemModifierData *modifier,
//                                             ParticleData *particle,
//                                             int particle_no,
//                                             int uv_no,
//                                             float r_uv[2])
//{
//  if (modifier->mesh_final == NULL) {
//    BKE_report(reports, RPT_ERROR, "Object was not yet evaluated");
//    zero_v2(r_uv);
//    return;
//  }
//  if (!CustomData_has_layer(&modifier->mesh_final->ldata, CD_MLOOPUV)) {
//    BKE_report(reports, RPT_ERROR, "Mesh has no UV data");
//    zero_v2(r_uv);
//    return;
//  }
//
//  {
//    float(*fuv)[4];
//    /* Note all sanity checks are done in this helper func. */
//    const int num = rna_ParticleSystem_tessfaceidx_on_emitter(
//        particlesystem, modifier, particle, particle_no, &fuv);
//
//    if (num < 0) {
//      /* No matching face found. */
//      zero_v2(r_uv);
//    }
//    else {
//      MFace *mface = &modifier->mesh_final->mface[num];
//      const MTFace *mtface = (const MTFace *)CustomData_get_layer_n(
//          &modifier->mesh_final->fdata, CD_MTFACE, uv_no);
//
//      psys_interpolate_uvs(&mtface[num], mface->v4, *fuv, r_uv);
//    }
//  }
//}
//
//static void rna_ParticleSystem_mcol_on_emitter(ParticleSystem *particlesystem,
//                                               ReportList *reports,
//                                               ParticleSystemModifierData *modifier,
//                                               ParticleData *particle,
//                                               int particle_no,
//                                               int vcol_no,
//                                               float r_mcol[3])
//{
//  if (!CustomData_has_layer(&modifier->mesh_final->ldata, CD_PROP_BYTE_COLOR)) {
//    BKE_report(reports, RPT_ERROR, "Mesh has no VCol data");
//    zero_v3(r_mcol);
//    return;
//  }
//
//  {
//    float(*fuv)[4];
//    /* Note all sanity checks are done in this helper func. */
//    const int num = rna_ParticleSystem_tessfaceidx_on_emitter(
//        particlesystem, modifier, particle, particle_no, &fuv);
//
//    if (num < 0) {
//      /* No matching face found. */
//      zero_v3(r_mcol);
//    }
//    else {
//      MFace *mface = &modifier->mesh_final->mface[num];
//      const MCol *mc = (const MCol *)CustomData_get_layer_n(
//          &modifier->mesh_final->fdata, CD_MCOL, vcol_no);
//      MCol mcol;
//
//      psys_interpolate_mcol(&mc[num * 4], mface->v4, *fuv, &mcol);
//      r_mcol[0] = (float)mcol.b / 255.0f;
//      r_mcol[1] = (float)mcol.g / 255.0f;
//      r_mcol[2] = (float)mcol.r / 255.0f;
//    }
//  }
//}
//
//static void particle_recalc(Main *UNUSED(bmain), Scene *UNUSED(scene), PointerRNA *ptr, short flag)
//{
//  if (ptr->type == &RNA_ParticleSystem) {
//    Object *ob = (Object *)ptr->owner_id;
//    ParticleSystem *psys = (ParticleSystem *)ptr->data;
//
//    psys->recalc = flag;
//
//    DEG_id_tag_update(&ob->id, ID_RECALC_GEOMETRY);
//  }
//  else {
//    DEG_id_tag_update(ptr->owner_id, ID_RECALC_GEOMETRY | flag);
//  }
//
//  WM_main_add_notifier(NC_OBJECT | ND_PARTICLE | NA_EDITED, NULL);
//}
//static void rna_Particle_redo(Main *bmain, Scene *scene, PointerRNA *ptr)
//{
//  particle_recalc(bmain, scene, ptr, ID_RECALC_PSYS_REDO);
//}
//
//static void rna_Particle_redo_dependency(Main *bmain, Scene *scene, PointerRNA *ptr)
//{
//  DEG_relations_tag_update(bmain);
//  rna_Particle_redo(bmain, scene, ptr);
//}
//
//static void rna_Particle_redo_count(Main *bmain, Scene *scene, PointerRNA *ptr)
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->data;
//  DEG_relations_tag_update(bmain);
//  psys_check_group_weights(part);
//  particle_recalc(bmain, scene, ptr, ID_RECALC_PSYS_REDO);
//}
//
//static void rna_Particle_reset(Main *bmain, Scene *scene, PointerRNA *ptr)
//{
//  particle_recalc(bmain, scene, ptr, ID_RECALC_PSYS_RESET);
//}
//
//static void rna_Particle_reset_dependency(Main *bmain, Scene *scene, PointerRNA *ptr)
//{
//  DEG_relations_tag_update(bmain);
//  rna_Particle_reset(bmain, scene, ptr);
//}
//
//static void rna_Particle_change_type(Main *bmain, Scene *UNUSED(scene), PointerRNA *ptr)
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->owner_id;
//
//  /* Iterating over all object is slow, but no better solution exists at the moment. */
//  for (Object *ob = bmain->objects.first; ob; ob = ob->id.next) {
//    LISTBASE_FOREACH (ParticleSystem *, psys, &ob->particlesystem) {
//      if (psys->part == part) {
//        psys_changed_type(ob, psys);
//        psys->recalc |= ID_RECALC_PSYS_RESET;
//        DEG_id_tag_update(&ob->id, ID_RECALC_GEOMETRY);
//      }
//    }
//  }
//
//  WM_main_add_notifier(NC_OBJECT | ND_PARTICLE | NA_EDITED, NULL);
//  DEG_relations_tag_update(bmain);
//}
//
//static void rna_Particle_change_physics_type(Main *bmain, Scene *scene, PointerRNA *ptr)
//{
//  particle_recalc(bmain, scene, ptr, ID_RECALC_PSYS_RESET | ID_RECALC_PSYS_PHYS);
//
//  ParticleSettings *part = (ParticleSettings *)ptr->data;
//
//  if (part->phystype == PART_PHYS_BOIDS && part->boids == NULL) {
//    BoidState *state;
//
//    part->boids = MEM_callocN(sizeof(BoidSettings), "Boid Settings");
//    boid_default_settings(part->boids);
//
//    state = boid_new_state(part->boids);
//    BLI_addtail(&state->rules, boid_new_rule(eBoidRuleType_Separate));
//    BLI_addtail(&state->rules, boid_new_rule(eBoidRuleType_Flock));
//
//    ((BoidRule *)state->rules.first)->flag |= BOIDRULE_CURRENT;
//
//    state->flag |= BOIDSTATE_CURRENT;
//    BLI_addtail(&part->boids->states, state);
//  }
//  else if (part->phystype == PART_PHYS_FLUID && part->fluid == NULL) {
//    part->fluid = MEM_callocN(sizeof(SPHFluidSettings), "SPH Fluid Settings");
//    BKE_particlesettings_fluid_default_settings(part);
//  }
//
//  DEG_relations_tag_update(bmain);
//}
//
//static void rna_Particle_redo_child(Main *bmain, Scene *scene, PointerRNA *ptr)
//{
//  particle_recalc(bmain, scene, ptr, ID_RECALC_PSYS_CHILD);
//}
//
//static void rna_Particle_cloth_update(Main *UNUSED(bmain), Scene *UNUSED(scene), PointerRNA *ptr)
//{
//  Object *ob = (Object *)ptr->owner_id;
//
//  DEG_id_tag_update(&ob->id, ID_RECALC_GEOMETRY);
//  WM_main_add_notifier(NC_OBJECT | ND_MODIFIER, ob);
//}
//
//static ParticleSystem *rna_particle_system_for_target(Object *ob, ParticleTarget *target)
//{
//  ParticleSystem *psys;
//  ParticleTarget *pt;
//
//  for (psys = ob->particlesystem.first; psys; psys = psys->next) {
//    for (pt = psys->targets.first; pt; pt = pt->next) {
//      if (pt == target) {
//        return psys;
//      }
//    }
//  }
//
//  return NULL;
//}
//
//static void rna_Particle_target_reset(Main *bmain, Scene *UNUSED(scene), PointerRNA *ptr)
//{
//  if (ptr->type == &RNA_ParticleTarget) {
//    Object *ob = (Object *)ptr->owner_id;
//    ParticleTarget *pt = (ParticleTarget *)ptr->data;
//    ParticleSystem *kpsys = NULL, *psys = rna_particle_system_for_target(ob, pt);
//
//    if (ELEM(pt->ob, ob, NULL)) {
//      kpsys = BLI_findlink(&ob->particlesystem, pt->psys - 1);
//
//      if (kpsys) {
//        pt->flag |= PTARGET_VALID;
//      }
//      else {
//        pt->flag &= ~PTARGET_VALID;
//      }
//    }
//    else {
//      if (pt->ob) {
//        kpsys = BLI_findlink(&pt->ob->particlesystem, pt->psys - 1);
//      }
//
//      if (kpsys) {
//        pt->flag |= PTARGET_VALID;
//      }
//      else {
//        pt->flag &= ~PTARGET_VALID;
//      }
//    }
//
//    psys->recalc = ID_RECALC_PSYS_RESET;
//
//    DEG_id_tag_update(&ob->id, ID_RECALC_GEOMETRY);
//    DEG_relations_tag_update(bmain);
//  }
//
//  WM_main_add_notifier(NC_OBJECT | ND_PARTICLE | NA_EDITED, NULL);
//}
//
//static void rna_Particle_target_redo(Main *UNUSED(bmain), Scene *UNUSED(scene), PointerRNA *ptr)
//{
//  if (ptr->type == &RNA_ParticleTarget) {
//    Object *ob = (Object *)ptr->owner_id;
//    ParticleTarget *pt = (ParticleTarget *)ptr->data;
//    ParticleSystem *psys = rna_particle_system_for_target(ob, pt);
//
//    psys->recalc = ID_RECALC_PSYS_REDO;
//
//    DEG_id_tag_update(&ob->id, ID_RECALC_GEOMETRY);
//    WM_main_add_notifier(NC_OBJECT | ND_PARTICLE | NA_EDITED, NULL);
//  }
//}
//
//static void rna_Particle_hair_dynamics_update(Main *bmain, Scene *scene, PointerRNA *ptr)
//{
//  Object *ob = (Object *)ptr->owner_id;
//  ParticleSystem *psys = (ParticleSystem *)ptr->data;
//
//  if (psys && !psys->clmd) {
//    psys->clmd = (ClothModifierData *)BKE_modifier_new(eModifierType_Cloth);
//    psys->clmd->sim_parms->goalspring = 0.0f;
//    psys->clmd->sim_parms->flags |= CLOTH_SIMSETTINGS_FLAG_RESIST_SPRING_COMPRESS;
//    psys->clmd->coll_parms->flags &= ~CLOTH_COLLSETTINGS_FLAG_SELF;
//    rna_Particle_redo(bmain, scene, ptr);
//  }
//  else {
//    WM_main_add_notifier(NC_OBJECT | ND_PARTICLE | NA_EDITED, NULL);
//  }
//
//  DEG_id_tag_update(&ob->id, ID_RECALC_GEOMETRY);
//  DEG_relations_tag_update(bmain);
//}
//
//static PointerRNA rna_particle_settings_get(PointerRNA *ptr)
//{
//  ParticleSystem *psys = (ParticleSystem *)ptr->data;
//  ParticleSettings *part = psys->part;
//
//  return rna_pointer_inherit_refine(ptr, &RNA_ParticleSettings, part);
//}
//
//static void rna_particle_settings_set(PointerRNA *ptr,
//                                      PointerRNA value,
//                                      struct ReportList *UNUSED(reports))
//{
//  Object *ob = (Object *)ptr->owner_id;
//  ParticleSystem *psys = (ParticleSystem *)ptr->data;
//  int old_type = 0;
//
//  if (psys->part) {
//    old_type = psys->part->type;
//    id_us_min(&psys->part->id);
//  }
//
//  psys->part = (ParticleSettings *)value.data;
//
//  if (psys->part) {
//    id_us_plus(&psys->part->id);
//    psys_check_boid_data(psys);
//    if (old_type != psys->part->type) {
//      psys_changed_type(ob, psys);
//    }
//  }
//}
//static void rna_Particle_abspathtime_update(Main *bmain, Scene *scene, PointerRNA *ptr)
//{
//  ParticleSettings *settings = (ParticleSettings *)ptr->data;
//  float delta = settings->end + settings->lifetime - settings->sta;
//  if (settings->draw & PART_ABS_PATH_TIME) {
//    settings->path_start = settings->sta + settings->path_start * delta;
//    settings->path_end = settings->sta + settings->path_end * delta;
//  }
//  else {
//    settings->path_start = (settings->path_start - settings->sta) / delta;
//    settings->path_end = (settings->path_end - settings->sta) / delta;
//  }
//  rna_Particle_redo(bmain, scene, ptr);
//}
//static void rna_PartSettings_start_set(struct PointerRNA *ptr, float value)
//{
//  ParticleSettings *settings = (ParticleSettings *)ptr->data;
//
//  /* check for clipping */
//  if (value > settings->end) {
//    settings->end = value;
//  }
//
//#  if 0
//  if (settings->type==PART_REACTOR && value < 1.0)
//    value = 1.0;
//  else
//#  endif
//  if (value < MINAFRAMEF) {
//    value = MINAFRAMEF;
//  }
//
//  settings->sta = value;
//}
//
//static void rna_PartSettings_end_set(struct PointerRNA *ptr, float value)
//{
//  ParticleSettings *settings = (ParticleSettings *)ptr->data;
//
//  /* check for clipping */
//  if (value < settings->sta) {
//    settings->sta = value;
//  }
//
//  settings->end = value;
//}
//
//static void rna_PartSetings_timestep_set(struct PointerRNA *ptr, float value)
//{
//  ParticleSettings *settings = (ParticleSettings *)ptr->data;
//
//  settings->timetweak = value / 0.04f;
//}
//
//static float rna_PartSettings_timestep_get(struct PointerRNA *ptr)
//{
//  ParticleSettings *settings = (ParticleSettings *)ptr->data;
//
//  return settings->timetweak * 0.04f;
//}
//
//static void rna_PartSetting_hairlength_set(struct PointerRNA *ptr, float value)
//{
//  ParticleSettings *settings = (ParticleSettings *)ptr->data;
//  settings->normfac = value / 4.0f;
//}
//
//static float rna_PartSetting_hairlength_get(struct PointerRNA *ptr)
//{
//  ParticleSettings *settings = (ParticleSettings *)ptr->data;
//  return settings->normfac * 4.0f;
//}
//
//static void rna_PartSetting_linelentail_set(struct PointerRNA *ptr, float value)
//{
//  ParticleSettings *settings = (ParticleSettings *)ptr->data;
//  settings->draw_line[0] = value;
//}
//
//static float rna_PartSetting_linelentail_get(struct PointerRNA *ptr)
//{
//  ParticleSettings *settings = (ParticleSettings *)ptr->data;
//  return settings->draw_line[0];
//}
//static void rna_PartSetting_pathstartend_range(
//    PointerRNA *ptr, float *min, float *max, float *UNUSED(softmin), float *UNUSED(softmax))
//{
//  ParticleSettings *settings = (ParticleSettings *)ptr->data;
//
//  if (settings->type == PART_HAIR) {
//    *min = 0.0f;
//    *max = (settings->draw & PART_ABS_PATH_TIME) ? 100.0f : 1.0f;
//  }
//  else {
//    *min = (settings->draw & PART_ABS_PATH_TIME) ? settings->sta : 0.0f;
//    *max = (settings->draw & PART_ABS_PATH_TIME) ? MAXFRAMEF : 1.0f;
//  }
//}
//static void rna_PartSetting_linelenhead_set(struct PointerRNA *ptr, float value)
//{
//  ParticleSettings *settings = (ParticleSettings *)ptr->data;
//  settings->draw_line[1] = value;
//}
//
//static float rna_PartSetting_linelenhead_get(struct PointerRNA *ptr)
//{
//  ParticleSettings *settings = (ParticleSettings *)ptr->data;
//  return settings->draw_line[1];
//}
//
//static int rna_PartSettings_is_fluid_get(PointerRNA *ptr)
//{
//  ParticleSettings *part = ptr->data;
//  return (ELEM(part->type,
//               PART_FLUID,
//               PART_FLUID_FLIP,
//               PART_FLUID_FOAM,
//               PART_FLUID_SPRAY,
//               PART_FLUID_BUBBLE,
//               PART_FLUID_TRACER,
//               PART_FLUID_SPRAYFOAM,
//               PART_FLUID_SPRAYBUBBLE,
//               PART_FLUID_FOAMBUBBLE,
//               PART_FLUID_SPRAYFOAMBUBBLE));
//}
//
//static void rna_ParticleSettings_use_clump_curve_update(Main *bmain, Scene *scene, PointerRNA *ptr)
//{
//  ParticleSettings *part = ptr->data;
//
//  if (part->child_flag & PART_CHILD_USE_CLUMP_CURVE) {
//    if (!part->clumpcurve) {
//      BKE_particlesettings_clump_curve_init(part);
//    }
//  }
//
//  rna_Particle_redo_child(bmain, scene, ptr);
//}
//
//static void rna_ParticleSettings_use_roughness_curve_update(Main *bmain,
//                                                            Scene *scene,
//                                                            PointerRNA *ptr)
//{
//  ParticleSettings *part = ptr->data;
//
//  if (part->child_flag & PART_CHILD_USE_ROUGH_CURVE) {
//    if (!part->roughcurve) {
//      BKE_particlesettings_rough_curve_init(part);
//    }
//  }
//
//  rna_Particle_redo_child(bmain, scene, ptr);
//}
//
//static void rna_ParticleSettings_use_twist_curve_update(Main *bmain, Scene *scene, PointerRNA *ptr)
//{
//  ParticleSettings *part = ptr->data;
//
//  if (part->child_flag & PART_CHILD_USE_TWIST_CURVE) {
//    if (!part->twistcurve) {
//      BKE_particlesettings_twist_curve_init(part);
//    }
//  }
//
//  rna_Particle_redo_child(bmain, scene, ptr);
//}
//
//static void rna_ParticleSystem_name_set(PointerRNA *ptr, const char *value)
//{
//  Object *ob = (Object *)ptr->owner_id;
//  ParticleSystem *part = (ParticleSystem *)ptr->data;
//
//  /* copy the new name into the name slot */
//  BLI_strncpy_utf8(part->name, value, sizeof(part->name));
//
//  BLI_uniquename(&ob->particlesystem,
//                 part,
//                 DATA_("ParticleSystem"),
//                 '.',
//                 offsetof(ParticleSystem, name),
//                 sizeof(part->name));
//}
//
//static PointerRNA rna_ParticleSystem_active_particle_target_get(PointerRNA *ptr)
//{
//  ParticleSystem *psys = (ParticleSystem *)ptr->data;
//  ParticleTarget *pt = psys->targets.first;
//
//  for (; pt; pt = pt->next) {
//    if (pt->flag & PTARGET_CURRENT) {
//      return rna_pointer_inherit_refine(ptr, &RNA_ParticleTarget, pt);
//    }
//  }
//  return rna_pointer_inherit_refine(ptr, &RNA_ParticleTarget, NULL);
//}
//static void rna_ParticleSystem_active_particle_target_index_range(
//    PointerRNA *ptr, int *min, int *max, int *UNUSED(softmin), int *UNUSED(softmax))
//{
//  ParticleSystem *psys = (ParticleSystem *)ptr->data;
//  *min = 0;
//  *max = max_ii(0, BLI_listbase_count(&psys->targets) - 1);
//}
//
//static int rna_ParticleSystem_active_particle_target_index_get(PointerRNA *ptr)
//{
//  ParticleSystem *psys = (ParticleSystem *)ptr->data;
//  ParticleTarget *pt = psys->targets.first;
//  int i = 0;
//
//  for (; pt; pt = pt->next, i++) {
//    if (pt->flag & PTARGET_CURRENT) {
//      return i;
//    }
//  }
//
//  return 0;
//}
//
//static void rna_ParticleSystem_active_particle_target_index_set(struct PointerRNA *ptr, int value)
//{
//  ParticleSystem *psys = (ParticleSystem *)ptr->data;
//  ParticleTarget *pt = psys->targets.first;
//  int i = 0;
//
//  for (; pt; pt = pt->next, i++) {
//    if (i == value) {
//      pt->flag |= PTARGET_CURRENT;
//    }
//    else {
//      pt->flag &= ~PTARGET_CURRENT;
//    }
//  }
//}
//
//static void rna_ParticleTarget_name_get(PointerRNA *ptr, char *str)
//{
//  ParticleTarget *pt = ptr->data;
//
//  if (pt->flag & PTARGET_VALID) {
//    ParticleSystem *psys = NULL;
//
//    if (pt->ob) {
//      psys = BLI_findlink(&pt->ob->particlesystem, pt->psys - 1);
//    }
//    else {
//      Object *ob = (Object *)ptr->owner_id;
//      psys = BLI_findlink(&ob->particlesystem, pt->psys - 1);
//    }
//
//    if (psys) {
//      if (pt->ob) {
//        sprintf(str, "%s: %s", pt->ob->id.name + 2, psys->name);
//      }
//      else {
//        strcpy(str, psys->name);
//      }
//    }
//    else {
//      strcpy(str, "Invalid target!");
//    }
//  }
//  else {
//    strcpy(str, "Invalid target!");
//  }
//}
//
//static int rna_ParticleTarget_name_length(PointerRNA *ptr)
//{
//  char tstr[MAX_ID_NAME + MAX_ID_NAME + 64];
//
//  rna_ParticleTarget_name_get(ptr, tstr);
//
//  return strlen(tstr);
//}
//
//static int particle_id_check(const PointerRNA *ptr)
//{
//  const ID *id = ptr->owner_id;
//
//  return (GS(id->name) == ID_PA);
//}
//
//static char *rna_SPHFluidSettings_path(const PointerRNA *ptr)
//{
//  const SPHFluidSettings *fluid = (SPHFluidSettings *)ptr->data;
//
//  if (particle_id_check(ptr)) {
//    const ParticleSettings *part = (ParticleSettings *)ptr->owner_id;
//
//    if (part->fluid == fluid) {
//      return BLI_strdup("fluid");
//    }
//  }
//  return NULL;
//}
//
//static bool rna_ParticleSystem_multiple_caches_get(PointerRNA *ptr)
//{
//  ParticleSystem *psys = (ParticleSystem *)ptr->data;
//
//  return (psys->ptcaches.first != psys->ptcaches.last);
//}
//static bool rna_ParticleSystem_editable_get(PointerRNA *ptr)
//{
//  ParticleSystem *psys = (ParticleSystem *)ptr->data;
//
//  return psys_check_edited(psys);
//}
//static bool rna_ParticleSystem_edited_get(PointerRNA *ptr)
//{
//  ParticleSystem *psys = (ParticleSystem *)ptr->data;
//
//  if (psys->part && psys->part->type == PART_HAIR) {
//    return (psys->flag & PSYS_EDITED || (psys->edit && psys->edit->edited));
//  }
//  else {
//    return (psys->pointcache->edit && psys->pointcache->edit->edited);
//  }
//}
//static PointerRNA rna_ParticleDupliWeight_active_get(PointerRNA *ptr)
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->owner_id;
//  ParticleDupliWeight *dw = part->instance_weights.first;
//
//  for (; dw; dw = dw->next) {
//    if (dw->flag & PART_DUPLIW_CURRENT) {
//      return rna_pointer_inherit_refine(ptr, &RNA_ParticleDupliWeight, dw);
//    }
//  }
//  return rna_pointer_inherit_refine(ptr, &RNA_ParticleTarget, NULL);
//}
//static void rna_ParticleDupliWeight_active_index_range(
//    PointerRNA *ptr, int *min, int *max, int *UNUSED(softmin), int *UNUSED(softmax))
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->owner_id;
//  *min = 0;
//  *max = max_ii(0, BLI_listbase_count(&part->instance_weights) - 1);
//}
//
//static int rna_ParticleDupliWeight_active_index_get(PointerRNA *ptr)
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->owner_id;
//  ParticleDupliWeight *dw = part->instance_weights.first;
//  int i = 0;
//
//  for (; dw; dw = dw->next, i++) {
//    if (dw->flag & PART_DUPLIW_CURRENT) {
//      return i;
//    }
//  }
//
//  return 0;
//}
//
//static void rna_ParticleDupliWeight_active_index_set(struct PointerRNA *ptr, int value)
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->owner_id;
//  ParticleDupliWeight *dw = part->instance_weights.first;
//  int i = 0;
//
//  for (; dw; dw = dw->next, i++) {
//    if (i == value) {
//      dw->flag |= PART_DUPLIW_CURRENT;
//    }
//    else {
//      dw->flag &= ~PART_DUPLIW_CURRENT;
//    }
//  }
//}
//
//static void rna_ParticleDupliWeight_name_get(PointerRNA *ptr, char *str)
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->owner_id;
//  psys_find_group_weights(part);
//
//  ParticleDupliWeight *dw = ptr->data;
//
//  if (dw->ob) {
//    sprintf(str, "%s: %i", dw->ob->id.name + 2, dw->count);
//  }
//  else {
//    strcpy(str, "No object");
//  }
//}
//
//static int rna_ParticleDupliWeight_name_length(PointerRNA *ptr)
//{
//  char tstr[MAX_ID_NAME + 64];
//
//  rna_ParticleDupliWeight_name_get(ptr, tstr);
//
//  return strlen(tstr);
//}
//
//static const EnumPropertyItem *rna_Particle_type_itemf(bContext *UNUSED(C),
//                                                       PointerRNA *ptr,
//                                                       PropertyRNA *UNUSED(prop),
//                                                       bool *UNUSED(r_free))
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->owner_id;
//
//  if (ELEM(part->type, PART_HAIR, PART_EMITTER)) {
//    return part_type_items;
//  }
//  else {
//    return part_fluid_type_items;
//  }
//}
//
//static const EnumPropertyItem *rna_Particle_from_itemf(bContext *UNUSED(C),
//                                                       PointerRNA *UNUSED(ptr),
//                                                       PropertyRNA *UNUSED(prop),
//                                                       bool *UNUSED(r_free))
//{
//#  if 0
//  if (part->type == PART_REACTOR) {
//    return part_reactor_from_items;
//  }
//#  endif
//  return part_from_items;
//}
//
//static const EnumPropertyItem *rna_Particle_dist_itemf(bContext *UNUSED(C),
//                                                       PointerRNA *ptr,
//                                                       PropertyRNA *UNUSED(prop),
//                                                       bool *UNUSED(r_free))
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->owner_id;
//
//  if (part->type == PART_HAIR) {
//    return part_hair_dist_items;
//  }
//  else {
//    return part_dist_items;
//  }
//}
//
//static const EnumPropertyItem *rna_Particle_draw_as_itemf(bContext *UNUSED(C),
//                                                          PointerRNA *ptr,
//                                                          PropertyRNA *UNUSED(prop),
//                                                          bool *UNUSED(r_free))
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->owner_id;
//
//  if (part->type == PART_HAIR) {
//    return part_hair_draw_as_items;
//  }
//  else {
//    return part_draw_as_items;
//  }
//}
//
//static const EnumPropertyItem *rna_Particle_ren_as_itemf(bContext *UNUSED(C),
//                                                         PointerRNA *ptr,
//                                                         PropertyRNA *UNUSED(prop),
//                                                         bool *UNUSED(r_free))
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->owner_id;
//
//  if (part->type == PART_HAIR) {
//    return part_hair_ren_as_items;
//  }
//  else {
//    return part_ren_as_items;
//  }
//}
//
//static PointerRNA rna_Particle_field1_get(PointerRNA *ptr)
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->owner_id;
//  return rna_pointer_inherit_refine(ptr, &RNA_FieldSettings, part->pd);
//}
//
//static PointerRNA rna_Particle_field2_get(PointerRNA *ptr)
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->owner_id;
//  return rna_pointer_inherit_refine(ptr, &RNA_FieldSettings, part->pd2);
//}
//
//static void psys_vg_name_get__internal(PointerRNA *ptr, char *value, int index)
//{
//  Object *ob = (Object *)ptr->owner_id;
//  ParticleSystem *psys = (ParticleSystem *)ptr->data;
//  const ListBase *defbase = BKE_object_defgroup_list(ob);
//
//  if (psys->vgroup[index] > 0) {
//    bDeformGroup *defGroup = BLI_findlink(defbase, psys->vgroup[index] - 1);
//
//    if (defGroup) {
//      strcpy(value, defGroup->name);
//      return;
//    }
//  }
//
//  value[0] = '\0';
//}
//static int psys_vg_name_len__internal(PointerRNA *ptr, int index)
//{
//  Object *ob = (Object *)ptr->owner_id;
//  ParticleSystem *psys = (ParticleSystem *)ptr->data;
//
//  if (psys->vgroup[index] > 0) {
//    const ListBase *defbase = BKE_object_defgroup_list(ob);
//    bDeformGroup *defGroup = BLI_findlink(defbase, psys->vgroup[index] - 1);
//
//    if (defGroup) {
//      return strlen(defGroup->name);
//    }
//  }
//  return 0;
//}
//static void psys_vg_name_set__internal(PointerRNA *ptr, const char *value, int index)
//{
//  Object *ob = (Object *)ptr->owner_id;
//  ParticleSystem *psys = (ParticleSystem *)ptr->data;
//
//  if (value[0] == '\0') {
//    psys->vgroup[index] = 0;
//  }
//  else {
//    int defgrp_index = BKE_object_defgroup_name_index(ob, value);
//
//    if (defgrp_index == -1) {
//      return;
//    }
//
//    psys->vgroup[index] = defgrp_index + 1;
//  }
//}
//
//static char *rna_ParticleSystem_path(const PointerRNA *ptr)
//{
//  const ParticleSystem *psys = (ParticleSystem *)ptr->data;
//  char name_esc[sizeof(psys->name) * 2];
//
//  BLI_str_escape(name_esc, psys->name, sizeof(name_esc));
//  return BLI_sprintfN("particle_systems[\"%s\"]", name_esc);
//}
//
//static void rna_ParticleSettings_mtex_begin(CollectionPropertyIterator *iter, PointerRNA *ptr)
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->data;
//  rna_iterator_array_begin(iter, (void *)part->mtex, sizeof(MTex *), MAX_MTEX, 0, NULL);
//}
//
//static PointerRNA rna_ParticleSettings_active_texture_get(PointerRNA *ptr)
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->data;
//  Tex *tex;
//
//  tex = give_current_particle_texture(part);
//  return rna_pointer_inherit_refine(ptr, &RNA_Texture, tex);
//}
//
//static void rna_ParticleSettings_active_texture_set(PointerRNA *ptr,
//                                                    PointerRNA value,
//                                                    struct ReportList *UNUSED(reports))
//{
//  ParticleSettings *part = (ParticleSettings *)ptr->data;
//
//  set_current_particle_texture(part, value.data);
//}
//
///* irritating string functions for each index :/ */
//static void rna_ParticleVGroup_name_get_0(PointerRNA *ptr, char *value)
//{
//  psys_vg_name_get__internal(ptr, value, 0);
//}
//static void rna_ParticleVGroup_name_get_1(PointerRNA *ptr, char *value)
//{
//  psys_vg_name_get__internal(ptr, value, 1);
//}
//static void rna_ParticleVGroup_name_get_2(PointerRNA *ptr, char *value)
//{
//  psys_vg_name_get__internal(ptr, value, 2);
//}
//static void rna_ParticleVGroup_name_get_3(PointerRNA *ptr, char *value)
//{
//  psys_vg_name_get__internal(ptr, value, 3);
//}
//static void rna_ParticleVGroup_name_get_4(PointerRNA *ptr, char *value)
//{
//  psys_vg_name_get__internal(ptr, value, 4);
//}
//static void rna_ParticleVGroup_name_get_5(PointerRNA *ptr, char *value)
//{
//  psys_vg_name_get__internal(ptr, value, 5);
//}
//static void rna_ParticleVGroup_name_get_6(PointerRNA *ptr, char *value)
//{
//  psys_vg_name_get__internal(ptr, value, 6);
//}
//static void rna_ParticleVGroup_name_get_7(PointerRNA *ptr, char *value)
//{
//  psys_vg_name_get__internal(ptr, value, 7);
//}
//static void rna_ParticleVGroup_name_get_8(PointerRNA *ptr, char *value)
//{
//  psys_vg_name_get__internal(ptr, value, 8);
//}
//static void rna_ParticleVGroup_name_get_9(PointerRNA *ptr, char *value)
//{
//  psys_vg_name_get__internal(ptr, value, 9);
//}
//static void rna_ParticleVGroup_name_get_10(PointerRNA *ptr, char *value)
//{
//  psys_vg_name_get__internal(ptr, value, 10);
//}
//static void rna_ParticleVGroup_name_get_11(PointerRNA *ptr, char *value)
//{
//  psys_vg_name_get__internal(ptr, value, 11);
//}
//static void rna_ParticleVGroup_name_get_12(PointerRNA *ptr, char *value)
//{
//  psys_vg_name_get__internal(ptr, value, 12);
//}
//
//static int rna_ParticleVGroup_name_len_0(PointerRNA *ptr)
//{
//  return psys_vg_name_len__internal(ptr, 0);
//}
//static int rna_ParticleVGroup_name_len_1(PointerRNA *ptr)
//{
//  return psys_vg_name_len__internal(ptr, 1);
//}
//static int rna_ParticleVGroup_name_len_2(PointerRNA *ptr)
//{
//  return psys_vg_name_len__internal(ptr, 2);
//}
//static int rna_ParticleVGroup_name_len_3(PointerRNA *ptr)
//{
//  return psys_vg_name_len__internal(ptr, 3);
//}
//static int rna_ParticleVGroup_name_len_4(PointerRNA *ptr)
//{
//  return psys_vg_name_len__internal(ptr, 4);
//}
//static int rna_ParticleVGroup_name_len_5(PointerRNA *ptr)
//{
//  return psys_vg_name_len__internal(ptr, 5);
//}
//static int rna_ParticleVGroup_name_len_6(PointerRNA *ptr)
//{
//  return psys_vg_name_len__internal(ptr, 6);
//}
//static int rna_ParticleVGroup_name_len_7(PointerRNA *ptr)
//{
//  return psys_vg_name_len__internal(ptr, 7);
//}
//static int rna_ParticleVGroup_name_len_8(PointerRNA *ptr)
//{
//  return psys_vg_name_len__internal(ptr, 8);
//}
//static int rna_ParticleVGroup_name_len_9(PointerRNA *ptr)
//{
//  return psys_vg_name_len__internal(ptr, 9);
//}
//static int rna_ParticleVGroup_name_len_10(PointerRNA *ptr)
//{
//  return psys_vg_name_len__internal(ptr, 10);
//}
//static int rna_ParticleVGroup_name_len_11(PointerRNA *ptr)
//{
//  return psys_vg_name_len__internal(ptr, 11);
//}
//static int rna_ParticleVGroup_name_len_12(PointerRNA *ptr)
//{
//  return psys_vg_name_len__internal(ptr, 12);
//}
//
//static void rna_ParticleVGroup_name_set_0(PointerRNA *ptr, const char *value)
//{
//  psys_vg_name_set__internal(ptr, value, 0);
//}
//static void rna_ParticleVGroup_name_set_1(PointerRNA *ptr, const char *value)
//{
//  psys_vg_name_set__internal(ptr, value, 1);
//}
//static void rna_ParticleVGroup_name_set_2(PointerRNA *ptr, const char *value)
//{
//  psys_vg_name_set__internal(ptr, value, 2);
//}
//static void rna_ParticleVGroup_name_set_3(PointerRNA *ptr, const char *value)
//{
//  psys_vg_name_set__internal(ptr, value, 3);
//}
//static void rna_ParticleVGroup_name_set_4(PointerRNA *ptr, const char *value)
//{
//  psys_vg_name_set__internal(ptr, value, 4);
//}
//static void rna_ParticleVGroup_name_set_5(PointerRNA *ptr, const char *value)
//{
//  psys_vg_name_set__internal(ptr, value, 5);
//}
//static void rna_ParticleVGroup_name_set_6(PointerRNA *ptr, const char *value)
//{
//  psys_vg_name_set__internal(ptr, value, 6);
//}
//static void rna_ParticleVGroup_name_set_7(PointerRNA *ptr, const char *value)
//{
//  psys_vg_name_set__internal(ptr, value, 7);
//}
//static void rna_ParticleVGroup_name_set_8(PointerRNA *ptr, const char *value)
//{
//  psys_vg_name_set__internal(ptr, value, 8);
//}
//static void rna_ParticleVGroup_name_set_9(PointerRNA *ptr, const char *value)
//{
//  psys_vg_name_set__internal(ptr, value, 9);
//}
//static void rna_ParticleVGroup_name_set_10(PointerRNA *ptr, const char *value)
//{
//  psys_vg_name_set__internal(ptr, value, 10);
//}
//static void rna_ParticleVGroup_name_set_11(PointerRNA *ptr, const char *value)
//{
//  psys_vg_name_set__internal(ptr, value, 11);
//}
//static void rna_ParticleVGroup_name_set_12(PointerRNA *ptr, const char *value)
//{
//  psys_vg_name_set__internal(ptr, value, 12);
//}

#else

//static void rna_def_particle_hair_key(BlenderRNA *brna)
//{
//  StructRNA *srna;
//  PropertyRNA *prop;
//
//  FunctionRNA *func;
//  PropertyRNA *parm;
//
//  //srna = RNA_def_struct(brna, "ParticleHairKey", NULL);
//  //RNA_def_struct_sdna(srna, "HairKey");
//  //RNA_def_struct_ui_text(srna, "Particle Hair Key", "Particle key for hair particle system");
//
//  //prop = RNA_def_property(srna, "time", PROP_FLOAT, PROP_UNSIGNED);
//  //RNA_def_property_ui_text(prop, "Time", "Relative time of key over hair length");
//
//  //prop = RNA_def_property(srna, "weight", PROP_FLOAT, PROP_UNSIGNED);
//  //RNA_def_property_range(prop, 0.0, 1.0);
//  //RNA_def_property_ui_text(prop, "Weight", "Weight for cloth simulation");
//
//  //prop = RNA_def_property(srna, "co", PROP_FLOAT, PROP_TRANSLATION);
//  //RNA_def_property_array(prop, 3);
//  //RNA_def_property_ui_text(
//  //    prop, "Location (Object Space)", "Location of the hair key in object space");
//  //RNA_def_property_float_funcs(prop,
//  //                             "rna_ParticleHairKey_location_object_get",
//  //                             "rna_ParticleHairKey_location_object_set",
//  //                             NULL);
//
//  //prop = RNA_def_property(srna, "co_local", PROP_FLOAT, PROP_TRANSLATION);
//  //RNA_def_property_float_sdna(prop, NULL, "co");
//  //RNA_def_property_ui_text(prop,
//  //                         "Location",
//  //                         "Location of the hair key in its local coordinate system, "
//  //                         "relative to the emitting face");
//
//  ///* Aided co func */
//  //func = RNA_def_function(srna, "co_object", "rna_ParticleHairKey_co_object");
//  //RNA_def_function_ui_description(func, "Obtain hairkey location with particle and modifier data");
//  //parm = RNA_def_pointer(func, "object", "Object", "", "Object");
//  //RNA_def_parameter_flags(parm, PROP_NEVER_NULL, PARM_REQUIRED);
//  //parm = RNA_def_pointer(func, "modifier", "ParticleSystemModifier", "", "Particle modifier");
//  //RNA_def_parameter_flags(parm, PROP_NEVER_NULL, PARM_REQUIRED);
//  //parm = RNA_def_pointer(func, "particle", "Particle", "", "hair particle");
//  //RNA_def_parameter_flags(parm, PROP_NEVER_NULL, PARM_REQUIRED);
//  //parm = RNA_def_float_vector(
//  //    func, "co", 3, NULL, -FLT_MAX, FLT_MAX, "Co", "Exported hairkey location", -1e4, 1e4);
//  //RNA_def_parameter_flags(parm, PROP_THICK_WRAP, 0);
//  //RNA_def_function_output(func, parm);
//
//  //func = RNA_def_function(srna, "co_object_set", "rna_ParticleHairKey_co_object_set");
//  //RNA_def_function_flag(func, FUNC_USE_SELF_ID);
//  //RNA_def_function_ui_description(func, "Set hairkey location with particle and modifier data");
//  //parm = RNA_def_pointer(func, "object", "Object", "", "Object");
//  //RNA_def_parameter_flags(parm, PROP_NEVER_NULL, PARM_REQUIRED);
//  //parm = RNA_def_pointer(func, "modifier", "ParticleSystemModifier", "", "Particle modifier");
//  //RNA_def_parameter_flags(parm, PROP_NEVER_NULL, PARM_REQUIRED);
//  //parm = RNA_def_pointer(func, "particle", "Particle", "", "hair particle");
//  //RNA_def_parameter_flags(parm, PROP_NEVER_NULL, PARM_REQUIRED);
//  //parm = RNA_def_float_vector(
//  //    func, "co", 3, NULL, -FLT_MAX, FLT_MAX, "Co", "Specified hairkey location", -1e4, 1e4);
//  //RNA_def_parameter_flags(parm, PROP_THICK_WRAP, PARM_REQUIRED);
//}
//
//static void rna_def_particle_key(BlenderRNA *brna)
//{
//  StructRNA *srna;
//  PropertyRNA *prop;
//
//  //srna = RNA_def_struct(brna, "ParticleKey", NULL);
//  //RNA_def_struct_ui_text(srna, "Particle Key", "Key location for a particle over time");
//
//  //prop = RNA_def_property(srna, "location", PROP_FLOAT, PROP_TRANSLATION);
//  //RNA_def_property_float_sdna(prop, NULL, "co");
//  //RNA_def_property_ui_text(prop, "Location", "Key location");
//
//  //prop = RNA_def_property(srna, "velocity", PROP_FLOAT, PROP_VELOCITY);
//  //RNA_def_property_float_sdna(prop, NULL, "vel");
//  //RNA_def_property_ui_text(prop, "Velocity", "Key velocity");
//
//  //prop = RNA_def_property(srna, "rotation", PROP_FLOAT, PROP_QUATERNION);
//  //RNA_def_property_float_sdna(prop, NULL, "rot");
//  //RNA_def_property_ui_text(prop, "Rotation", "Key rotation quaternion");
//
//  //prop = RNA_def_property(srna, "angular_velocity", PROP_FLOAT, PROP_VELOCITY);
//  //RNA_def_property_float_sdna(prop, NULL, "ave");
//  //RNA_def_property_ui_text(prop, "Angular Velocity", "Key angular velocity");
//
//  //prop = RNA_def_property(srna, "time", PROP_FLOAT, PROP_UNSIGNED);
//  //RNA_def_property_ui_text(prop, "Time", "Time of key over the simulation");
//}
//
//static void rna_def_child_particle(BlenderRNA *brna)
//{
//  StructRNA *srna;
//  // PropertyRNA *prop;
//
//  srna = RNA_def_struct(brna, "ChildParticle", NULL);
//  RNA_def_struct_ui_text(
//      srna, "Child Particle", "Child particle interpolated from simulated or edited particles");
//
//  /*  int num, parent; */ /* num is face index on the final derived mesh */
//
//  /*  int pa[4]; */             /* nearest particles to the child, used for the interpolation */
//  /*  float w[4]; */            /* interpolation weights for the above particles */
//  /*  float fuv[4], foffset; */ /* face vertex weights and offset */
//  /*  float rand[3]; */
//}
//
//static void rna_def_particle(BlenderRNA *brna)
//{
//  StructRNA *srna;
//  PropertyRNA *prop;
//
//  FunctionRNA *func;
//  PropertyRNA *parm;
//
//  static const EnumPropertyItem alive_items[] = {
//      /*{PARS_KILLED, "KILLED", 0, "Killed", ""}, */
//      {PARS_DEAD, "DEAD", 0, "Dead", ""},
//      {PARS_UNBORN, "UNBORN", 0, "Unborn", ""},
//      {PARS_ALIVE, "ALIVE", 0, "Alive", ""},
//      {PARS_DYING, "DYING", 0, "Dying", ""},
//      {0, NULL, 0, NULL, NULL},
//  };
//}
//
//static void rna_def_particle_target(BlenderRNA *brna)
//{
//  StructRNA *srna;
//  PropertyRNA *prop;
//
//  static const EnumPropertyItem mode_items[] = {
//      {PTARGET_MODE_FRIEND, "FRIEND", 0, "Friend", ""},
//      {PTARGET_MODE_NEUTRAL, "NEUTRAL", 0, "Neutral", ""},
//      {PTARGET_MODE_ENEMY, "ENEMY", 0, "Enemy", ""},
//      {0, NULL, 0, NULL, NULL},
//  };
//}
//
//void RNA_def_particle(BlenderRNA *brna)
//{
//  rna_def_particle_target(brna);
//  rna_def_particle_hair_key(brna);
//  rna_def_particle_key(brna);
//
//  rna_def_child_particle(brna);
//  rna_def_particle(brna);
//}

#endif
