#pragma once

#include "ID.h"
#include "B_math.h"

struct BVHTree;
struct DistProjectedAABBPrecalc;

#define USE_KDOPBVH_WATERTIGHT

typedef struct BVHTreeAxisRange 
{
    union 
    {
        float min, max;
        float range[2];  /* alternate access */
    };
} BVHTreeAxisRange;


typedef struct BVHTreeOverlap {
    int indexA;
    int indexB;
} BVHTreeOverlap;

typedef struct BVHTreeNearest {
    /** The index of the nearest found
     * (untouched if none is found within a dist radius from the given coordinates) */
    int index;
    /** Nearest coordinates
     * (untouched it none is found within a dist radius from the given coordinates). */
    float co[3];
    /** Normal at nearest coordinates
     * (untouched it none is found within a dist radius from the given coordinates). */
    float no[3];
    /** squared distance to search around */
    float dist_sq;
    int flags;
} BVHTreeNearest;

typedef struct BVHTreeRay {
    /** ray origin */
    float origin[3];
    /** ray direction */
    float direction[3];
    /** radius around ray */
    float radius;
#ifdef USE_KDOPBVH_WATERTIGHT
    struct IsectRayPrecalc* isect_precalc;
#endif
} BVHTreeRay;

typedef struct BVHTreeRayHit {
    /** Index of the tree node (untouched if no hit is found). */
    int index;
    /** Coordinates of the hit point. */
    float co[3];
    /** Normal on hit point. */
    float no[3];
    /** Distance to the hit point. */
    float dist;
} BVHTreeRayHit;

enum {
    /* Use a priority queue to process nodes in the optimal order (for slow callbacks) */
    BVH_OVERLAP_USE_THREADING = (1 << 0),
    BVH_OVERLAP_RETURN_PAIRS = (1 << 1),
};
enum {
    /* Use a priority queue to process nodes in the optimal order (for slow callbacks) */
    BVH_NEAREST_OPTIMAL_ORDER = (1 << 0),
};
enum {
    /* calculate IsectRayPrecalc data */
    BVH_RAYCAST_WATERTIGHT = (1 << 0),
};
#define BVH_RAYCAST_DEFAULT (BVH_RAYCAST_WATERTIGHT)
#define BVH_RAYCAST_DIST_MAX (FLT_MAX / 2.0f)

/**
 * Callback must update nearest in case it finds a nearest result.
 */
typedef void (*BVHTree_NearestPointCallback)(void* userdata,
    int index,
    const float co[3],
    BVHTreeNearest* nearest);

/**
 * Callback must update hit in case it finds a nearest successful hit.
 */
typedef void (*BVHTree_RayCastCallback)(void* userdata,
    int index,
    const BVHTreeRay* ray,
    BVHTreeRayHit* hit);

/**
 * Callback to check if 2 nodes overlap (use thread if intersection results need to be stored).
 */
typedef bool (*BVHTree_OverlapCallback)(void* userdata, int index_a, int index_b, int thread);

/**
 * Callback to range search query.
 */
typedef void (*BVHTree_RangeQuery)(void* userdata, int index, const float co[3], float dist_sq);

/**
 * Callback to find nearest projected.
 */
typedef void (*BVHTree_NearestProjectedCallback)(void* userdata, int index, const struct DistProjectedAABBPrecalc* precalc, const float(*clip_plane)[4], int clip_plane_len, BVHTreeNearest* nearest);

/* callbacks to BLI_bvhtree_walk_dfs */

/**
 * Return true to traverse into this nodes children, else skip.
 */
typedef bool (*BVHTree_WalkParentCallback)(const BVHTreeAxisRange* bounds, void* userdata);
/**
 * Return true to keep walking, else early-exit the search.
 */
typedef bool (*BVHTree_WalkLeafCallback)(const BVHTreeAxisRange* bounds, int index, void* userdata);
/**
 * Return true to search (min, max) else (max, min).
 */
typedef bool (*BVHTree_WalkOrderCallback)(const BVHTreeAxisRange* bounds, char axis, void* userdata);

/**
 * \note many callers don't check for `NULL` return.
 */
__host__ __device__ BVHTree* BLI_bvhtree_new(uint maxsize, float epsilon, char tree_type, char axis);
__host__ __device__ void BLI_bvhtree_free(BVHTree* tree);

/**
 * Construct: first insert points, then call balance.
 */
__host__ __device__ void BLI_bvhtree_insert(BVHTree* tree, int index, const float co[3], int numpoints);
__host__ __device__ void BLI_bvhtree_balance(BVHTree* tree);

/**
 * Update: first update points/nodes, then call update_tree to refit the bounding volumes.
 * \note call before #BLI_bvhtree_update_tree().
 */
__device__ bool BLI_bvhtree_update_node(const BVHTree* tree, int index, const float co[3], const float co_moving[3], int numpoints);
/**
 * Call #BLI_bvhtree_update_node() first for every node/point/triangle.
 */
__host__ __device__ void BLI_bvhtree_update_tree(BVHTree* tree);

/**
 * Use to check the total number of threads #BLI_bvhtree_overlap will use.
 *
 * \warning Must be the first tree passed to #BLI_bvhtree_overlap!
 */
__host__ __device__ int BLI_bvhtree_overlap_thread_num(const BVHTree* tree);

/**
 * Collision/overlap: check two trees if they overlap,
 * alloc's *overlap with length of the int return value.
 *
 * \param callback: optional, to test the overlap before adding (must be thread-safe!).
 */
__host__ __device__ BVHTreeOverlap* BLI_bvhtree_overlap_ex(const BVHTree* tree1,
    const BVHTree* tree2,
    uint* r_overlap_num,
    BVHTree_OverlapCallback callback,
    void* userdata,
    uint max_interactions,
    int flag);

__host__ __device__ BVHTreeOverlap* BLI_bvhtree_overlap(const BVHTree* tree1,
    const BVHTree* tree2,
    uint* r_overlap_num,
    BVHTree_OverlapCallback callback,
    void* userdata);
__host__ __device__ int* BLI_bvhtree_intersect_plane(BVHTree* tree, float plane[4], uint* r_intersect_num);

/**
 * Number of times #BLI_bvhtree_insert has been called.
 * mainly useful for asserts functions to check we added the correct number.
 */
__host__ __device__ int BLI_bvhtree_get_len(const BVHTree* tree);
/**
 * Maximum number of children that a node can have.
 */
__host__ __device__ int BLI_bvhtree_get_tree_type(const BVHTree* tree);
__host__ __device__ float BLI_bvhtree_get_epsilon(const BVHTree* tree);
/**
 * This function returns the bounding box of the BVH tree.
 */
__host__ __device__ void BLI_bvhtree_get_bounding_box(const BVHTree* tree, float r_bb_min[3], float r_bb_max[3]);

/**
 * Find nearest node to the given coordinates
 * (if nearest is given it will only search nodes where
 * square distance is smaller than nearest->dist).
 */
int BLI_bvhtree_find_nearest_ex(const BVHTree* tree,
                                const float co[3],
                                BVHTreeNearest* nearest,
                                BVHTree_NearestPointCallback callback,
                                void* userdata,
                                int flag);
int BLI_bvhtree_find_nearest(BVHTree* tree,
    const float co[3],
    BVHTreeNearest* nearest,
    BVHTree_NearestPointCallback callback,
    void* userdata);

/**
 * Find the first node nearby.
 * Favors speed over quality since it doesn't find the best target node.
 */
int BLI_bvhtree_find_nearest_first(const BVHTree* tree,
                                   const float co[3],
                                   float dist_sq,
                                   BVHTree_NearestPointCallback callback,
                                   void* userdata);

int BLI_bvhtree_ray_cast_ex(const BVHTree* tree,
                            const float co[3],
                            const float dir[3],
                            float radius,
                            BVHTreeRayHit* hit,
                            BVHTree_RayCastCallback callback,
                            void* userdata,
                            int flag);
int BLI_bvhtree_ray_cast(BVHTree* tree,
    const float co[3],
    const float dir[3],
    float radius,
    BVHTreeRayHit* hit,
    BVHTree_RayCastCallback callback,
    void* userdata);

/**
 * Calls the callback for every ray intersection
 *
 * \note Using a \a callback which resets or never sets the #BVHTreeRayHit index & dist works too,
 * however using this function means existing generic callbacks can be used from custom callbacks
 * without having to handle resetting the hit beforehand.
 * It also avoid redundant argument and return value which aren't meaningful
 * when collecting multiple hits.
 */
void BLI_bvhtree_ray_cast_all_ex(const BVHTree* tree,
                                 const float co[3],
                                 const float dir[3],
                                 float radius,
                                 float hit_dist,
                                 BVHTree_RayCastCallback callback,
                                 void* userdata,
                                 int flag);
void BLI_bvhtree_ray_cast_all(BVHTree* tree,
    const float co[3],
    const float dir[3],
    float radius,
    float hit_dist,
    BVHTree_RayCastCallback callback,
    void* userdata);

float BLI_bvhtree_bb_raycast(const float bv[6],
    const float light_start[3],
    const float light_end[3],
    float pos[3]);

/**
 * Range query.
 */
int BLI_bvhtree_range_query(
    BVHTree* tree, const float co[3], float radius, BVHTree_RangeQuery callback, void* userdata);

int BLI_bvhtree_find_nearest_projected(BVHTree* tree,
    float projmat[4][4],
    float winsize[2],
    float mval[2],
    float clip_planes[6][4],
    int clip_plane_len,
    BVHTreeNearest* nearest,
    BVHTree_NearestProjectedCallback callback,
    void* userdata);

/**
 * This is a generic function to perform a depth first search on the #BVHTree
 * where the search order and nodes traversed depend on callbacks passed in.
 *
 * \param tree: Tree to walk.
 * \param walk_parent_cb: Callback on a parents bound-box to test if it should be traversed.
 * \param walk_leaf_cb: Callback to test leaf nodes, callback must store its own result,
 * returning false exits early.
 * \param walk_order_cb: Callback that indicates which direction to search,
 * either from the node with the lower or higher K-DOP axis value.
 * \param userdata: Argument passed to all callbacks.
 */
void BLI_bvhtree_walk_dfs(const BVHTree* tree,
                          BVHTree_WalkParentCallback walk_parent_cb,
                          BVHTree_WalkLeafCallback walk_leaf_cb,
                          BVHTree_WalkOrderCallback walk_order_cb,
                          void* userdata);

/**
 * Expose for BVH callbacks to use.
 */
extern const float bvhtree_kdop_axes[13][3];


typedef unsigned char axis_t;

typedef struct BVHNode {
    BVHNode** children;
    BVHNode* parent; /* some user defined traversed need that */
    float* bv;      /* Bounding volume of all nodes, max 13 axis */
    int index;      /* face, edge, vertex index */
    char totnode;   /* how many nodes are used, used for speedup */
    char main_axis; /* Axis used to split this node */
} BVHNode;

/* keep under 26 bytes for speed purposes */
struct BVHTree {
    BVHNode** nodes;
    BVHNode* nodearray;  /* pre-alloc branch nodes */
    BVHNode** nodechild; /* pre-alloc children for nodes */
    float* nodebv;       /* pre-alloc bounding-volumes for nodes */
    float epsilon;       /* Epsilon is used for inflation of the K-DOP. */
    int totleaf;         /* leafs */
    int totbranch;
    axis_t start_axis, stop_axis; /* bvhtree_kdop_axes array indices according to axis */
    axis_t axis;                  /* kdop type (6 => OBB, 7 => AABB, ...) */
    char tree_type;               /* type of tree (4 => quadtree) */
};

/* optimization, ensure we stay small */
BLI_STATIC_ASSERT((sizeof(void*) == 8 && sizeof(BVHTree) <= 48) || (sizeof(void*) == 4 && sizeof(BVHTree) <= 32),  "over sized")

/* avoid duplicating vars in BVHOverlapData_Thread */
typedef struct BVHOverlapData_Shared
{
const BVHTree* tree1, * tree2;
axis_t start_axis, stop_axis;

/* use for callbacks */
BVHTree_OverlapCallback callback;
void* userdata;
} BVHOverlapData_Shared;

typedef struct BVHOverlapData_Thread {
    BVHOverlapData_Shared* shared;
    struct BLI_Stack* overlap; /* store BVHTreeOverlap */
    uint max_interactions;
    /* use for callbacks */
    int thread;
} BVHOverlapData_Thread;

typedef struct BVHNearestData {
    const BVHTree* tree;
    const float* co;
    BVHTree_NearestPointCallback callback;
    void* userdata;
    float proj[13]; /* coordinates projection over axis */
    BVHTreeNearest nearest;

} BVHNearestData;

typedef struct BVHRayCastData {
    const BVHTree* tree;

    BVHTree_RayCastCallback callback;
    void* userdata;

    BVHTreeRay ray;

#ifdef USE_KDOPBVH_WATERTIGHT
    struct IsectRayPrecalc isect_precalc;
#endif

    /* initialized by bvhtree_ray_cast_data_precalc */
    float ray_dot_axis[13];
    float idot_axis[13];
    int index[6];

    BVHTreeRayHit hit;
} BVHRayCastData;