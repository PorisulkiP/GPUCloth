#pragma once

#ifndef __MOD_UTIL__
#define __MOD_UTIL__

#include "DNA_modifier_defaults.h"

#include "cloth_types.cuh"
#include "MEM_guardedalloc.cuh"

//#include "DEG_depsgraph_build.h"

#ifdef __cplusplus
extern "C" {
#endif

struct MDeformVert;
struct Mesh;
struct ModifierData;
struct ModifierEvalContext;
struct Object;

#ifdef __CUDA_ARCH__
	#define MY_CONSTANT __device__ const
#else
	#define MY_CONSTANT static constexpr
#endif


#define SDNA_TYPE_CHECKED(v) (&(v))

MY_CONSTANT ClothSimSettings DNA_DEFAULT_ClothSimSettings = _DNA_DEFAULT_ClothSimSettings;
MY_CONSTANT ClothCollSettings DNA_DEFAULT_ClothCollSettings = _DNA_DEFAULT_ClothCollSettings;
MY_CONSTANT ClothModifierData DNA_DEFAULT_ClothModifierData = _DNA_DEFAULT_ClothModifierData;

enum {
    _SDNA_TYPE_ClothSimSettings = 0,
    _SDNA_TYPE_ClothCollSettings = 1,
    _SDNA_TYPE_ClothModifierData = 2,
    SDNA_TYPE_MAX = 3,
};

__device__ inline const void* d_DNA_default_table[SDNA_TYPE_MAX] =
{
    &DNA_DEFAULT_ClothSimSettings,
    &DNA_DEFAULT_ClothCollSettings,
    &DNA_DEFAULT_ClothModifierData,
};

inline const void* DNA_default_table[SDNA_TYPE_MAX] =
{
    &DNA_DEFAULT_ClothSimSettings,
    &DNA_DEFAULT_ClothCollSettings,
    &DNA_DEFAULT_ClothModifierData,
};

/**
 * Wrap with macro that casts correctly.
 */
#define SDNA_TYPE_FROM_STRUCT(id) _SDNA_TYPE_##id

#define DNA_struct_default_get(struct_name) (const struct_name *)DNA_default_table[SDNA_TYPE_FROM_STRUCT(struct_name)]

#define DNA_struct_default_alloc(struct_name) (struct_name *)_DNA_struct_default_alloc_impl((const uint8_t *)DNA_default_table[SDNA_TYPE_FROM_STRUCT(struct_name)], sizeof(struct_name), __func__)

#define MEMCPY_STRUCT_AFTER(struct_dst, struct_src, member) \
  { \
    CHECK_TYPE_NONCONST(struct_dst); ((void)(struct_dst == struct_src), \
     memcpy((char *)(struct_dst) + OFFSETOF_STRUCT_AFTER(struct_dst, member), \
            (const char *)(struct_src) + OFFSETOF_STRUCT_AFTER(struct_dst, member), \
            sizeof(*(struct_dst)) - OFFSETOF_STRUCT_AFTER(struct_dst, member))); \
  }  ((void)0)


__host__ __device__ inline uint8_t* _DNA_struct_default_alloc_impl(const uint8_t* data_src, size_t size, const char* alloc_str)
{
    auto* data_dst = static_cast<uint8_t*>(MEM_lockfree_mallocN(size, alloc_str));
    memcpy(data_dst, data_src, size);
    return data_dst;
}

//void MOD_init_texture(struct MappingInfoModifierData *dmd, const struct ModifierEvalContext *ctx);
///**
// * \param cos: may be NULL, in which case we use directly mesh vertices' coordinates.
// */
//void MOD_get_texture_coords(struct MappingInfoModifierData *dmd,
//                            const struct ModifierEvalContext *ctx,
//                            struct Object *ob,
//                            struct Mesh *mesh,
//                            float (*cos)[3],
//                            float (*r_texco)[3]);
//
//void MOD_previous_vcos_store(struct ModifierData *md, const float (*vert_coords)[3]);
//
///**
// * \returns a mesh if mesh == NULL, for deforming modifiers that need it.
// */
//struct Mesh *MOD_deform_mesh_eval_get(struct Object *ob,
//                                      struct BMEditMesh *em,
//                                      struct Mesh *mesh,
//                                      const float (*vertexCos)[3],
//                                      int verts_num,
//                                      bool use_normals,
//                                      bool use_orco);
//
//void MOD_get_vgroup(struct Object *ob,
//                    struct Mesh *mesh,
//                    const char *name,
//                    struct MDeformVert **dvert,
//                    int *defgrp_index);
//
//void MOD_depsgraph_update_object_bone_relation(struct DepsNodeHandle *node,
//                                               struct Object *object,
//                                               const char *bonename,
//                                               const char *description);

#ifdef __cplusplus
}
#endif

#endif