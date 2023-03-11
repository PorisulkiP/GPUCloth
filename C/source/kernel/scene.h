#pragma once

#include "scene_types.cuh"

struct AviCodecData;
struct Collection;
struct Depsgraph;
struct GHash;
struct Main;
struct Object;
struct RenderData;
struct Scene;
struct TransformOrientation;
struct UnitSettings;
struct View3DCursor;
struct ViewLayer;

typedef enum eSceneCopyMethod {
  SCE_COPY_NEW = 0,
  SCE_COPY_EMPTY = 1,
  SCE_COPY_LINK_COLLECTION = 2,
  SCE_COPY_FULL = 3,
} eSceneCopyMethod;

/* Scene base iteration function.
 * Define struct here, so no need to bother with alloc/free it. */
typedef struct SceneBaseIter {
    struct ListBase* duplilist;
    struct DupliObject* dupob;
    float omat[4][4];
    struct Object* dupli_refob;
    int phase;
} SceneBaseIter;