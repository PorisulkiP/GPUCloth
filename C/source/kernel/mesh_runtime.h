#pragma once

#include "customdata.h"  /* for eCustomDataMask */

/* avoid fan-fill topology */
#define USE_CLIP_EVEN
#define USE_CONVEX_SKIP
/* sweep back-and-forth about convex ears (avoids lop-sided fans) */
#define USE_CLIP_SWEEP
// #define USE_CONVEX_SKIP_TEST

#ifdef USE_CONVEX_SKIP
#  define USE_KDTREE
#endif

#ifdef __cplusplus
extern "C" {
#endif

    struct CustomData;
    struct CustomData_MeshMasks;
    struct Depsgraph;
    struct KeyBlock;
    struct MLoop;
    struct MLoopTri;
    struct MVertTri;
    struct Mesh;
    struct Object;
    struct Scene;

    typedef struct KDTreeNode2D {
        uint32_t neg, pos;
        uint32_t index;
        axis_t axis; /* range is only (0-1) */
        uint16_t flag;
        uint32_t parent;
    } KDTreeNode2D;

    struct KDRange2D {
        float min, max;
    };

    struct KDTree2D {
        KDTreeNode2D* nodes;
        const float(*coords)[2];
        uint32_t root;
        uint32_t node_num;
        uint32_t* nodes_map; /* index -> node lookup */
    };

    enum {
        CONCAVE = -1,
        TANGENTIAL = 0,
        CONVEX = 1,
    };

    typedef struct PolyFill {
        struct PolyIndex* indices; /* vertex aligned */

        const float(*coords)[2];
        uint coords_num;
#ifdef USE_CONVEX_SKIP
        uint coords_num_concave;
#endif

        /* A polygon with n vertices has a triangulation of n-2 triangles. */
        uint(*tris)[3];
        uint tris_num;

#ifdef USE_KDTREE
        struct KDTree2D kdtree;
#endif
    } PolyFill;
    typedef signed char eSign;

    /** Circular double linked-list. */
    typedef struct PolyIndex {
        struct PolyIndex* next, * prev;
        uint index;
        signed char sign;
    } PolyIndex;

    struct TessellationUserTLS 
    {
        MemArena* pf_arena;
    };

    struct TessellationUserData {
        const MLoop* mloop;
        const MPoly* mpoly;
        const float(*positions)[3];

        /** Output array. */
        MLoopTri* mlooptri;

        /** Optional pre-calculated polygon normals array. */
        const float(*poly_normals)[3];
    };

    signed char span_tri_v2_sign(const float v1[2], const float v2[2], const float v3[2]);

    void mesh_ensure_looptri_data(const Mesh* mesh);

    //void BKE_mesh_batch_cache_free(Mesh* me);

    //void pf_coord_sign_calc(PolyFill* pf, PolyIndex* pi);
    /**
     * \brief Initialize the runtime of the given mesh.
     *
     * Function expects that the runtime is already cleared.
     */
    //void BKE_mesh_runtime_init_data(struct Mesh *mesh);
    /**
     * \brief Free all data (and mutexes) inside the runtime of the given mesh.
     */
    //void BKE_mesh_runtime_free_data(struct Mesh *mesh);
    /**
     * Clear all pointers which we don't want to be shared on copying the datablock.
     * However, keep all the flags which defines what the mesh is (for example, that
     * it's deformed only, or that its custom data layers are out of date.)
     */
    //void BKE_mesh_runtime_reset_on_copy(struct Mesh *mesh, int flag);
    int BKE_mesh_runtime_looptri_len(const struct Mesh *mesh);
    void BKE_mesh_runtime_looptri_recalc(const struct Mesh *mesh);
    /**
     * \note This function only fills a cache, and therefore the mesh argument can
     * be considered logically const. Concurrent access is protected by a mutex.
     * \note This is a ported copy of dm_getLoopTriArray(dm).
     */
    const struct MLoopTri *BKE_mesh_runtime_looptri_ensure(const struct Mesh *mesh);
    void BKE_mesh_recalc_looptri(const MLoop* mloop, const MPoly* mpoly, const float(*vert_positions)[3], int totloop, int totpoly, MLoopTri* mlooptri);
    bool BKE_mesh_runtime_ensure_edit_data(struct Mesh *mesh);
    //bool BKE_mesh_runtime_clear_edit_data(struct Mesh *mesh);
    //bool BKE_mesh_runtime_reset_edit_data(struct Mesh *mesh);
    void BKE_mesh_runtime_clear_geometry(struct Mesh *mesh);
    /**
     * \brief This function clears runtime cache of the given mesh.
     *
     * Call this function to recalculate runtime data when used.
     */
    void BKE_mesh_runtime_clear_cache(struct Mesh *mesh);

    /* This is a copy of DM_verttri_from_looptri(). */
    void BKE_mesh_runtime_verttri_from_looptri(struct MVertTri *r_verttri,
                                               const struct MLoop *mloop,
                                               const struct MLoopTri *looptri,
                                               int looptri_num);

    /* NOTE: the functions below are defined in DerivedMesh.cc, and are intended to be moved
     * to a more suitable location when that file is removed.
     * They should also be renamed to use conventions from BKE, not old DerivedMesh.cc.
     * For now keep the names similar to avoid confusion. */

    //struct Mesh *mesh_get_eval_final(struct Depsgraph *depsgraph,
    //                                 const struct Scene *scene,
    //                                 struct Object *ob,
    //                                 const struct CustomData_MeshMasks *dataMask);

    //struct Mesh *mesh_get_eval_deform(struct Depsgraph *depsgraph,
    //                                  const struct Scene *scene,
    //                                  struct Object *ob,
    //                                  const struct CustomData_MeshMasks *dataMask);

    //struct Mesh *mesh_create_eval_final(struct Depsgraph *depsgraph,
    //                                    const struct Scene *scene,
    //                                    struct Object *ob,
    //                                    const struct CustomData_MeshMasks *dataMask);

    //struct Mesh *mesh_create_eval_no_deform(struct Depsgraph *depsgraph,
    //                                        const struct Scene *scene,
    //                                        struct Object *ob,
    //                                        const struct CustomData_MeshMasks *dataMask);
    //struct Mesh *mesh_create_eval_no_deform_render(struct Depsgraph *depsgraph,
    //                                               const struct Scene *scene,
    //                                               struct Object *ob,
    //                                               const struct CustomData_MeshMasks *dataMask);

    //void BKE_mesh_runtime_eval_to_meshkey(struct Mesh *me_deformed,
    //                                      struct Mesh *me,
    //                                      struct KeyBlock *kb);

#ifndef NDEBUG
    bool BKE_mesh_runtime_is_valid(Mesh* me_eval);
    char* BKE_mesh_debug_info(const Mesh* me);
#endif /* NDEBUG */

#ifdef __cplusplus
}
#endif
