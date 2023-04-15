#pragma once

#include "customdata_types.cuh"
#include "listBase.h"
#include "math_vector.cuh"

/**
 * Mesh Vertices.
 *
 * Typically accessed from #Mesh.mvert
 */
typedef struct MVert {
  float co[3];
  /**
   * Cache the normal, can always be recalculated from surrounding faces.
   * See #CD_CUSTOMLOOPNORMAL for custom normals.
   */
  short no[3];
  char flag, bweight;
} MVert;

inline void print_MVert(MVert& mvert)
{
    print_v3("\tcoo", mvert.co);
    //printf("\n\tnormals: %d, %d, %d", mvert.no[0], mvert.no[1], mvert.no[2]);
    //printf("\n\tflag: %d", mvert.flag);
    //printf("\n\tmbweight: %d", mvert.bweight);
}

/** #MVert.flag */
enum {
  /*  SELECT = (1 << 0), */
  ME_VERT_TMP_TAG = (1 << 2),
  ME_HIDE = (1 << 4),
  ME_VERT_FACEDOT = (1 << 5),
  /*  ME_VERT_MERGED = (1 << 6), */
  ME_VERT_PBVH_UPDATE = (1 << 7),
};

/**
 * Mesh Edges.
 *
 * Typically accessed from #Mesh.medge
 */
typedef struct MEdge {
  /** Un-ordered vertex indices (cannot match). */
  uint v1, v2;
  char crease, bweight;
  short flag;
} MEdge;

/** #MEdge.flag */
enum {
  /*  SELECT = (1 << 0), */
  ME_EDGEDRAW = (1 << 1),
  ME_SEAM = (1 << 2),
  /*  ME_HIDE = (1 << 4), */
  ME_EDGERENDER = (1 << 5),
  ME_LOOSEEDGE = (1 << 7),
  ME_EDGE_TMP_TAG = (1 << 8),
  ME_SHARP = (1 << 9), /* only reason this flag remains a 'short' */
};

/**
 * Mesh Faces
 * This only stores the polygon size & flags, the vertex & edge indices are stored in the #MLoop.
 *
 * Typically accessed from #Mesh.mpoly.
 */
typedef struct MPoly {
  /** Offset into loop array and number of loops in the face. */
  int loopstart;
  /** Keep signed since we need to subtract when getting the previous loop. */
  int totloop;
  short mat_nr;
  char flag;// , _pad;
} MPoly;

/** #MPoly.flag */
enum {
  ME_SMOOTH = (1 << 0),
  ME_FACE_SEL = (1 << 1),
  /* ME_HIDE = (1 << 4), */
};

/**
 * Mesh Loops.
 * Each loop represents the corner of a polygon (#MPoly).
 *
 * Typically accessed from #Mesh.mloop.
 */
typedef struct MLoop {
  /** Vertex index. */
  uint v;
  /**
   * Edge index.
   *
   * \note The e here is because we want to move away from relying on edge hashes.
   */
  uint e;
} MLoop;

/**
 * Optionally store the order of selected elements.
 * This wont always be set since only some selection operations have an order.
 *
 * Typically accessed from #Mesh.mselect
 */
typedef struct MSelect 
{
  /** Index in the vertex, edge or polygon array. */
  int index;
  /** #ME_VSEL, #ME_ESEL, #ME_FSEL. */
  int type;
} MSelect;

enum {
  ME_VSEL = 0,
  ME_ESEL = 1,
  ME_FSEL = 2,
};

typedef struct MLoopTri 
{
  uint tri[3];
  uint poly;
} MLoopTri;


typedef struct MVertTri 
{
  uint tri[3];
} MVertTri;

typedef struct MFloatProperty 
{
  float f;
} MFloatProperty;

typedef struct MIntProperty 
{
  int i;
} MIntProperty;

typedef struct MStringProperty 
{
  char s[255], s_len;
} MStringProperty;

typedef struct MBoolProperty 
{
  uint8_t b;
} MBoolProperty;


/**
 * Vertex group index and weight for #MDeformVert.dw
 */
typedef struct MDeformWeight {
  /** The index for the vertex group, must *always* be unique when in an array. */
  uint def_nr;
  /** Weight between 0.0 and 1.0. */
  float weight;
} MDeformWeight;

typedef struct MDeformVert {
  struct MDeformWeight *dw;
  int totweight;
  /** Flag is only in use as a run-time tag at the moment. */
  int flag;
} MDeformVert;

typedef struct MVertSkin {
  /**
   * Radii of the skin, define how big the generated frames are.
   * Currently only the first two elements are used.
   */
  float radius[3];

  /** #eMVertSkinFlag */
  int flag;
} MVertSkin;

typedef enum eMVertSkinFlag {
  /** Marks a vertex as the edge-graph root, used for calculating rotations for all connected
   * edges (recursively). Also used to choose a root when generating an armature.
   */
  MVERT_SKIN_ROOT = 1,

  /** Marks a branch vertex (vertex with more than two connected edges), so that its neighbors
   * are directly hulled together, rather than the default of generating intermediate frames.
   */
  MVERT_SKIN_LOOSE = 2,
} eMVertSkinFlag;

/** \} */

/* -------------------------------------------------------------------- */
/** \name Custom Data (Loop)
 * \{ */

/**
 * UV coordinate for a polygon face & flag for selection & other options.
 */
typedef struct MLoopUV {
  float uv[2];
  int flag;
} MLoopUV;

/** #MLoopUV.flag */
enum {
  /* MLOOPUV_DEPRECATED = (1 << 0), MLOOPUV_EDGESEL removed */
  MLOOPUV_VERTSEL = (1 << 1),
  MLOOPUV_PINNED = (1 << 2),
};

/**
 * \note While alpha is not currently in the 3D Viewport,
 * this may eventually be added back, keep this value set to 255.
 */
typedef struct MLoopCol {
  unsigned char r, g, b, a;
} MLoopCol;

typedef struct MPropCol {
  float color[4];
} MPropCol;

/** Multi-Resolution loop data. */
typedef struct MDisps {
  /* Strange bug in SDNA: if disps pointer comes first, it fails to see totdisp */
  int totdisp;
  int level;
  float (*disps)[3];

  /**
   * Used for hiding parts of a multires mesh.
   * Essentially the multires equivalent of #MVert.flag's ME_HIDE bit.
   *
   * \note This is a bitmap, keep in sync with type used in BLI_bitmap.h
   */
  uint *hidden;
} MDisps;

/** Multi-Resolution grid loop data. */
typedef struct GridPaintMask {
  /**
   * The data array contains `grid_size * grid_size` elements.
   * Where `grid_size = (1 << (level - 1)) + 1`.
   */
  float *data;

  /** The maximum multires level associated with this grid. */
  uint level;

  char _pad[4];
} GridPaintMask;

typedef struct OrigSpaceFace {
  float uv[4][2];
} OrigSpaceFace;


typedef struct OrigSpaceLoop {
  float uv[2];
} OrigSpaceLoop;

typedef struct FreestyleEdge {
  char flag;
} FreestyleEdge;

/** #FreestyleEdge.flag */
enum {
  FREESTYLE_EDGE_MARK = 1,
};

typedef struct FreestyleFace {
  char flag;
} FreestyleFace;

/** #FreestyleFace.flag */
enum {
  FREESTYLE_FACE_MARK = 1,
};

#define ME_POLY_LOOP_PREV(mloop, mp, i) \
  (&(mloop)[(mp)->loopstart + (((i) + (mp)->totloop - 1) % (mp)->totloop)])
#define ME_POLY_LOOP_NEXT(mloop, mp, i) (&(mloop)[(mp)->loopstart + (((i) + 1) % (mp)->totloop)])

/** Number of tri's that make up this polygon once tessellated. */
#define ME_POLY_TRI_TOT(mp) ((mp)->totloop - 2)

/**
 * Check out-of-bounds material, note that this is nearly always prevented,
 * yet its still possible in rare cases.
 * So usage such as array lookup needs to check.
 */
#define ME_MAT_NR_TEST(mat_nr, totmat) \
  (CHECK_TYPE_ANY(mat_nr, short, const short), \
   CHECK_TYPE_ANY(totmat, short, const short), \
   (LIKELY(mat_nr < totmat) ? mat_nr : 0))

/**
 * Used in Blender pre 2.63, See #MLoop, #MPoly for face data stored in the blend file.
 * Use for reading old files and in a handful of cases which should be removed eventually.
 */
typedef struct MFace {
  int v1, v2, v3, v4;
  short mat_nr;
  /** We keep edcode, for conversion to edges draw flags in old files. */
  char edcode, flag;
} MFace;

/** #MFace.edcode */
enum {
  ME_V1V2 = (1 << 0),
  ME_V2V3 = (1 << 1),
  ME_V3V1 = (1 << 2),
  ME_V3V4 = ME_V3V1,
  ME_V4V1 = (1 << 3),
};

/** Tessellation uv face data. */
typedef struct MTFace {
  float uv[4][2];
} MTFace;

/**
 * Tessellation vertex color data.
 *
 * \note The red and blue are swapped for historical reasons.
 */
typedef struct MCol {
  unsigned char a, r, g, b;
} MCol;

#define MESH_MLOOPCOL_FROM_MCOL(_mloopcol, _mcol) \
  { \
    MLoopCol *mloopcol__tmp = _mloopcol; \
    const MCol *mcol__tmp = _mcol; \
    mloopcol__tmp->r = mcol__tmp->b; \
    mloopcol__tmp->g = mcol__tmp->g; \
    mloopcol__tmp->b = mcol__tmp->r; \
    mloopcol__tmp->a = mcol__tmp->a; \
  } \
  (void)0

#define MESH_MLOOPCOL_TO_MCOL(_mloopcol, _mcol) \
  { \
    const MLoopCol *mloopcol__tmp = _mloopcol; \
    MCol *mcol__tmp = _mcol; \
    mcol__tmp->b = mloopcol__tmp->r; \
    mcol__tmp->g = mloopcol__tmp->g; \
    mcol__tmp->r = mloopcol__tmp->b; \
    mcol__tmp->a = mloopcol__tmp->a; \
  } \
  (void)0

/** Old game engine recast navigation data, while unused 2.7x files may contain this. */
typedef struct MRecast { int i; } MRecast;

