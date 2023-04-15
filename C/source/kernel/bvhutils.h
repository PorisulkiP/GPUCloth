#pragma once

//#include "BLI_bitmap.h"
#include "kdopbvh.h"
#include "threads.h"


/**
 * This header encapsulates necessary code to build a BVH
 */

struct BMEditMesh;
struct MFace;
struct MVert;
struct Mesh;
struct PointCloud;

struct BVHCache;

/**
 * Struct that stores basic information about a BVHTree built from a edit-mesh.
 */
typedef struct BVHTreeFromEditMesh {
  struct BVHTree *tree;

  /* default callbacks to bvh nearest and raycast */
  BVHTree_NearestPointCallback nearest_callback;
  BVHTree_RayCastCallback raycast_callback;

  struct BMEditMesh *em;

  /* Private data */
  bool cached;

} BVHTreeFromEditMesh;

/**
 * Struct that stores basic information about a BVHTree built from a mesh.
 */
typedef struct BVHTreeFromMesh {
  struct BVHTree *tree;

  /* default callbacks to bvh nearest and raycast */
  BVHTree_NearestPointCallback nearest_callback;
  BVHTree_RayCastCallback raycast_callback;

  /* Vertex array, so that callbacks have instante access to data */
  const struct MVert *vert;
  const struct MEdge *edge; /* only used for BVHTreeFromMeshEdges */
  const struct MFace *face;
  const struct MLoop *loop;
  const struct MLoopTri *looptri;
  bool vert_allocated;
  bool edge_allocated;
  bool face_allocated;
  bool loop_allocated;
  bool looptri_allocated;

  /* Private data */
  bool cached;

} BVHTreeFromMesh;

typedef enum BVHCacheType {
  BVHTREE_FROM_VERTS,
  BVHTREE_FROM_EDGES,
  BVHTREE_FROM_FACES,
  BVHTREE_FROM_LOOPTRI,
  BVHTREE_FROM_LOOPTRI_NO_HIDDEN,

  BVHTREE_FROM_LOOSEVERTS,
  BVHTREE_FROM_LOOSEEDGES,

  BVHTREE_FROM_EM_VERTS,
  BVHTREE_FROM_EM_EDGES,
  BVHTREE_FROM_EM_LOOPTRI,

  /* Keep `BVHTREE_MAX_ITEM` as last item. */
  BVHTREE_MAX_ITEM,
} BVHCacheType;

const float(*BKE_mesh_vertex_normals_ensure(const struct Mesh* mesh))[3];

/**
 * Builds a bvh tree where nodes are the relevant elements of the given mesh.
 * Configures #BVHTreeFromMesh.
 *
 * The tree is build in mesh space coordinates, this means special care must be made on queries
 * so that the coordinates and rays are first translated on the mesh local coordinates.
 * Reason for this is that bvh_from_mesh_* can use a cache in some cases and so it
 * becomes possible to reuse a #BVHTree.
 *
 * free_bvhtree_from_mesh should be called when the tree is no longer needed.
 */
BVHTree *bvhtree_from_editmesh_verts(BVHTreeFromEditMesh *data, struct BMEditMesh *em, float epsilon, int tree_type, int axis);
BVHTree *bvhtree_from_editmesh_edges(BVHTreeFromEditMesh *data, struct BMEditMesh *em, float epsilon, int tree_type, int axis);
BVHTree *bvhtree_from_editmesh_looptri(BVHTreeFromEditMesh *data, struct BMEditMesh *em, float epsilon, int tree_type, int axis);

BVHTree *BKE_bvhtree_from_mesh_get(BVHTreeFromMesh *data,
                                   struct Mesh *mesh,
                                   const BVHCacheType bvh_cache_type,
                                   const int tree_type);

BVHTree *BKE_bvhtree_from_editmesh_get(BVHTreeFromEditMesh *data,
                                       struct BMEditMesh *em,
                                       const int tree_type,
                                       const BVHCacheType bvh_cache_type,
                                       struct BVHCache **bvh_cache_p,
                                       ThreadMutex *mesh_eval_mutex);

/**
 * Frees data allocated by a call to bvhtree_from_mesh_*.
 */
void free_bvhtree_from_editmesh(struct BVHTreeFromEditMesh *data);
void free_bvhtree_from_mesh(struct BVHTreeFromMesh *data);

/**
 * Math functions used by callbacks
 */
float bvhtree_ray_tri_intersection(const BVHTreeRay *ray, const float m_dist, const float v0[3], const float v1[3], const float v2[3]);
float bvhtree_sphereray_tri_intersection(const BVHTreeRay *ray,
                                         float radius,
                                         const float m_dist,
                                         const float v0[3],
                                         const float v1[3],
                                         const float v2[3]);

typedef struct BVHTreeFromPointCloud {
  struct BVHTree *tree;

  BVHTree_NearestPointCallback nearest_callback;

  const float (*coords)[3];
} BVHTreeFromPointCloud;

//BVHTree *BKE_bvhtree_from_pointcloud_get(struct BVHTreeFromPointCloud *data,
//                                         const struct PointCloud *pointcloud,
//                                         const int tree_type);

void free_bvhtree_from_pointcloud(struct BVHTreeFromPointCloud *data);

/**
 * BVHCache
 */

/* Using local coordinates */

bool bvhcache_has_tree(const struct BVHCache *bvh_cache, const BVHTree *tree);
struct BVHCache *bvhcache_init(void);
void bvhcache_free(struct BVHCache *bvh_cache);
