#pragma once

#include "ID.h"
#include "customdata_types.cuh"
#include "defs.cuh"

#include <mutex>

struct AnimData;
struct BVHCache;
struct Mesh;
struct Key;
struct MCol;
struct MEdge;
struct MFace;
struct MLoop;
struct MLoopCol;
struct MLoopTri;
struct MLoopUV;
struct MPoly;
struct MVert;
struct Material;
struct SubdivCCG;
struct SubsurfRuntimeData;

typedef struct EditMeshData {
  /** when set, \a vertexNos, polyNos are lazy initialized */
  const float (*vertexCos)[3];

  /** lazy initialize (when \a vertexCos is set) */
  float const (*vertexNos)[3];
  float const (*polyNos)[3];
  /** also lazy init but don't depend on \a vertexCos */
  const float (*polyCos)[3];
} EditMeshData;

/**
 * \warning Typical access is done via
 * #BKE_mesh_runtime_looptri_ensure, #BKE_mesh_runtime_looptri_len.
 */
struct MLoopTri_Store {
  
  /* WARNING! swapping between array (ready-to-be-used data) and array_wip
   * (where data is actually computed)
   * shall always be protected by same lock as one used for looptris computing. */
    //
  struct MLoopTri *array, *array_wip;
  int len;
  int len_alloc;
};


/* **************** MESH ********************* */

/** #Mesh_Runtime.wrapper_type */
typedef enum eMeshWrapperType {
    /** Use mesh data (#Mesh.mvert, #Mesh.medge, #Mesh.mloop, #Mesh.mpoly). */
    ME_WRAPPER_TYPE_MDATA = 0,
    /** Use edit-mesh data (#Mesh.edit_mesh, #Mesh_Runtime.edit_data). */
    ME_WRAPPER_TYPE_BMESH = 1,
    /** Use subdivision mesh data (#Mesh_Runtime.mesh_eval). */
    ME_WRAPPER_TYPE_SUBD = 2,
} eMeshWrapperType;

/** #Mesh.texflag */
enum {
    ME_AUTOSPACE = 1,
    ME_AUTOSPACE_EVALUATED = 2,
};

/** #Mesh.editflag */
enum {
    ME_EDIT_MIRROR_VERTEX_GROUPS = 1 << 0,
    ME_EDIT_MIRROR_Y = 1 << 1, /* unused so far */
    ME_EDIT_MIRROR_Z = 1 << 2, /* unused so far */

    ME_EDIT_PAINT_FACE_SEL = 1 << 3,
    ME_EDIT_MIRROR_TOPO = 1 << 4,
    ME_EDIT_PAINT_VERT_SEL = 1 << 5,
};


/**
 * Cache of a mesh's loose edges, accessed with #Mesh::loose_edges(). *
 */
struct LooseEdgeCache {
    /**
     * A bitmap set to true for each loose edge, false if the edge is used by any face.
     * Allocated only if there is at least one loose edge.
     */
     /**
     * Для растрового изображения установлено значение true для каждого незакрепленного ребра, false, если ребро используется какой-либо гранью.
     * Выделяется только в том случае, если имеется хотя бы одно незакрепленное ребро.
     */
    //blender::BitVector<> is_loose_bits;
    /**
     * The number of loose edges. If zero, the #is_loose_bits shouldn't be accessed.
     * If less than zero, the cache has been accessed in an invalid way
     * (i.e.directly instead of through #Mesh::loose_edges()).
     */
    int count = -1;
};

struct MeshRuntime {
    /* Evaluated mesh for objects which do not have effective modifiers.
     * This mesh is used as a result of modifier stack evaluation.
     * Since modifier stack evaluation is threaded on object level we need some synchronization. */
    Mesh* mesh_eval = nullptr;
    std::mutex eval_mutex;

    /* A separate mutex is needed for normal calculation, because sometimes
     * the normals are needed while #eval_mutex is already locked. */
    std::mutex normals_mutex;

    /** Needed to ensure some thread-safety during render data pre-processing. */
    std::mutex render_mutex;

    /**
     * A cache of bounds shared between data-blocks with unchanged positions. When changing positions
     * affect the bounds, the cache is "un-shared" with other geometries. See #SharedCache comments.
     */
     /**
     * Кэш границ, совместно используемых блоками данных с неизменными позициями. При смене положения
     * влияет на границы, кэш "не используется совместно" с другими геометриями. Смотрите комментарии #SharedCache.
     */
    //SharedCache<Bounds<float3>> bounds_cache;

    /** Lazily initialized SoA data from the #edit_mesh field in #Mesh. */
    EditMeshData* edit_data = nullptr;

    /**
     * Data used to efficiently draw the mesh in the viewport, especially useful when
     * the same mesh is used in many objects or instances. See `draw_cache_impl_mesh.cc`.
     */
    void* batch_cache = nullptr;

    /** Cache for derived triangulation of the mesh, accessed with #Mesh::looptris(). */
    MLoopTri_Store looptris;

    /** Cache for BVH trees generated for the mesh. Defined in 'BKE_bvhutil.c' */
    BVHCache* bvh_cache = nullptr;

    /** Cache of non-manifold boundary data for Shrink-wrap Target Project. */
    //ShrinkwrapBoundaryData* shrinkwrap_data = nullptr;

    /** Needed in case we need to lazily initialize the mesh. */
    CustomData_MeshMasks cd_mask_extra = {};

    SubdivCCG* subdiv_ccg = nullptr;
    int subdiv_ccg_tot_level = 0;

    /** Set by modifier stack if only deformed from original. */
    bool deformed_only = false;
    /**
     * Copied from edit-mesh (hint, draw with edit-mesh data when true).
     *
     * Modifiers that edit the mesh data in-place must set this to false
     * (most #eModifierTypeType_NonGeometrical modifiers). Otherwise the edit-mesh
     * data will be used for drawing, missing changes from modifiers. See #79517.
     */
    bool is_original_bmesh = false;

    /** #eMeshWrapperType and others. */
    eMeshWrapperType wrapper_type = ME_WRAPPER_TYPE_MDATA;
    /**
     * A type mask from wrapper_type,
     * in case there are differences in finalizing logic between types.
     */
    eMeshWrapperType wrapper_type_finalize = ME_WRAPPER_TYPE_MDATA;

    /**
     * Settings for lazily evaluating the subdivision on the CPU if needed. These are
     * set in the modifier when GPU subdivision can be performed, and owned by the by
     * the modifier in the object.
     */
    SubsurfRuntimeData* subsurf_runtime_data = nullptr;

    /**
     * Caches for lazily computed vertex and polygon normals. These are stored here rather than in
     * #CustomData because they can be calculated on a `const` mesh, and adding custom data layers on
     * a `const` mesh is not thread-safe.
     */
    bool vert_normals_dirty = true;
    bool poly_normals_dirty = true;
    float(*vert_normals)[3] = nullptr;
    float(*poly_normals)[3] = nullptr;

    /**
     * A cache of data about the loose edges. Can be shared with other data-blocks with unchanged
     * topology. Accessed with #Mesh::loose_edges().
     */
    /**
     * Кэш данных о незакрепленных краях. Может использоваться совместно с другими блоками данных с неизменным
     * топология. Доступ осуществляется с помощью #Mesh::loose_edges().
     */
    //SharedCache<LooseEdgeCache> loose_edges_cache;

    /**
     * A bit vector the size of the number of vertices, set to true for the center vertices of
     * subdivided polygons. The values are set by the subdivision surface modifier and used by
     * drawing code instead of polygon center face dots. Otherwise this will be empty.
     */
     /**
     * Битовый вектор размера числа вершин, установленный в значение true для центральных вершин
     * разделенных полигонов. Значения задаются модификатором subdivision surface и используются
     * кодом рисования вместо точек центральной грани полигона. В противном случае это поле будет пустым.
      */
    //BitVector<> subsurf_face_dot_tags;

    /**
     * A bit vector the size of the number of edges, set to true for edges that should be drawn in
     * the viewport. Created by the "Optimal Display" feature of the subdivision surface modifier.
     * Otherwise it will be empty.
     */
     /**
     * Битовый вектор, определяющий размер количества ребер, установленный в значение true для ребер, которые должны быть нарисованы в
     * окно просмотра. Создано с помощью функции "Оптимальное отображение" модификатора subdivision surface.
     * В противном случае он будет пустым.
     */
    //BitVector<> subsurf_optimal_display_edges;

    MeshRuntime() = default;
    ~MeshRuntime();

    //MEM_CXX_CLASS_ALLOC_FUNCS("MeshRuntime")
};


typedef struct Mesh {
    ID id;

    //struct Key* key;
    /**
    * Array of vertices. Edges and faces are defined by indices into this array.
    * \note This pointer is for convenient access to the #CD_MVERT layer in #vdata.
    */
    struct MVert *mvert;
    /**
    * Array of edges, containing vertex indices. For simple triangle or quad meshes, edges could be
    * calculated from the #MPoly and #MLoop arrays, however, edges need to be stored explicitly to
    * edge domain attributes and to support loose edges that aren't connected to faces.
    * \note This pointer is for convenient access to the #CD_MEDGE layer in #edata.
    */
    struct MEdge *medge;
    /**
    * Face topology storage of the size and offset of each face's section of the #mloop face corner
    * array. Also stores various flags and the `material_index` attribute.
    * \note This pointer is for convenient access to the #CD_MPOLY layer in #pdata.
    */
    struct MPoly *mpoly;
    /**
    * The vertex and edge index at each face corner.
    * \note This pointer is for convenient access to the #CD_MLOOP layer in #ldata.
    */
    struct MLoop *mloop;

    /** The number of vertices (#MVert) in the mesh, and the size of #vdata. */
    int totvert;
    /** The number of edges (#MEdge) in the mesh, and the size of #edata. */
    int totedge;
    /** The number of polygons/faces (#MPoly) in the mesh, and the size of #pdata. */
    int totpoly;
    /** The number of face corners (#MLoop) in the mesh, and the size of #ldata. */
    int totloop;

    CustomData vdata, edata, pdata, ldata;

    /** "Vertex group" vertices. */
    struct MDeformVert *dvert;
    /**
    * List of vertex group (#bDeformGroup) names and flags only. Actual weights are stored in dvert.
    * \note This pointer is for convenient access to the #CD_MDEFORMVERT layer in #vdata.
    */
    ListBase vertex_group_names;
    /** The active index in the #vertex_group_names list. */
    int vertex_group_active_index;

    /**
    * The index of the active attribute in the UI. The attribute list is a combination of the
    * generic type attributes from vertex, edge, face, and corner custom data.
    */
    int attributes_active_index;

    /**
    * Runtime storage of the edit mode mesh. If it exists, it generally has the most up-to-date
    * information about the mesh.
    * \note When the object is available, the preferred access method is #BKE_editmesh_from_object.
    */
    //struct BMEditMesh *edit_mesh;

    /** The length of the #mselect array. */
    int totselect;

    /**
    * In most cases the last selected element (see #mselect) represents the active element.
    * For faces we make an exception and store the active face separately so it can be active
    * even when no faces are selected. This is done to prevent flickering in the material properties
    * and UV Editor which base the content they display on the current material which is controlled
    * by the active face.
    *
    * \note This is mainly stored for use in edit-mode.
    */
    int act_face;

    /** Texture space location and size, used for procedural coordinates when rendering. */
    float loc[3];
    float size[3];
    char texflag;

    /** Various flags used when editing the mesh. */
    char editflag;
    /** Mostly more flags used when editing or displaying the mesh. */
    uint16_t flag;

    /**
    * The angle for auto smooth in radians. `M_PI` (180 degrees) causes all edges to be smooth.
    */
    float smoothresh;

    /**
    * Flag for choosing whether or not so store bevel weight and crease as custom data layers in the
    * edit mesh (they are always stored in #MVert and #MEdge currently). In the future, this data
    * may be stored as generic named attributes (see T89054 and T93602).
    */
    char cd_flag;

    /**
    * User-defined symmetry flag (#eMeshSymmetryType) that causes editing operations to maintain
    * symmetrical geometry. Supported by operations such as transform and weight-painting.
    */
    char symmetry;

    /** The length of the #mat array. */
    short totcol;

    /** Choice between different remesh methods in the UI. */
    char remesh_mode;

    char subdiv;
    char subdivr;
    char subsurftype;

    /**
    * Deprecated. Store of runtime data for tessellation face UVs and texture.
    *
    * \note This would be marked deprecated, however the particles still use this at run-time
    * for placing particles on the mesh (something which should be eventually upgraded).
    */
    struct MTFace *mtface;

    /**
    * Deprecated face storage (quads & triangles only);
    * faces are now pointed to by #Mesh.mpoly and #Mesh.mloop.
    *
    * \note This would be marked deprecated, however the particles still use this at run-time
    * for placing particles on the mesh (something which should be eventually upgraded).
    */
    struct MFace *mface;
    /* Deprecated storage of old faces (only triangles or quads). */
    CustomData fdata;
    /* Deprecated size of #fdata. */
    int totface;

    /** Per-mesh settings for voxel remesh. */
    float remesh_voxel_size;
    float remesh_voxel_adaptivity;

    int face_sets_color_seed;
    /* Stores the initial Face Set to be rendered white. This way the overlay can be enabled by
    * default and Face Sets can be used without affecting the color of the mesh. */
    int face_sets_color_default;

    /**
     * Data that isn't saved in files, including caches of derived data, temporary data to improve
     * the editing experience, etc. The struct is created when reading files and can be accessed
     * without null checks, with the exception of some temporary meshes which should allocate and
     * free the data if they are passed to functions that expect run-time data.
     */
    /**
     * Данные, которые не сохраняются в файлах, включая кеши производных данных, временные данные для улучшения
     * опыт редактирования и т.д. Структура создается при чтении файлов, и к ней можно получить доступ
     * без нулевых проверок, за исключением некоторых временных сеток, которые должны выделять и
     * освободите данные, если они передаются функциям, которые ожидают данных во время выполнения.
     */
    MeshRuntime* runtime;
} Mesh;

/* Helper macro to see if vertex group X mirror is on. */
#define ME_USING_MIRROR_X_VERTEX_GROUPS(_me) \
  (((_me)->editflag & ME_EDIT_MIRROR_VERTEX_GROUPS) && ((_me)->symmetry & ME_SYMMETRY_X))

/* We can't have both flags enabled at once,
 * flags defined in scene_types.cuh */
#define ME_EDIT_PAINT_SEL_MODE(_me) \
  (((_me)->editflag & ME_EDIT_PAINT_FACE_SEL) ? SCE_SELECT_FACE : \
   ((_me)->editflag & ME_EDIT_PAINT_VERT_SEL) ? SCE_SELECT_VERTEX : \
                                                0)

/** #Mesh.flag */
enum {
  ME_FLAG_UNUSED_0 = 1 << 0,     /* cleared */
  ME_FLAG_UNUSED_1 = 1 << 1,     /* cleared */
  ME_FLAG_DEPRECATED_2 = 1 << 2, /* deprecated */
  ME_FLAG_UNUSED_3 = 1 << 3,     /* cleared */
  ME_FLAG_UNUSED_4 = 1 << 4,     /* cleared */
  ME_AUTOSMOOTH = 1 << 5,
  ME_FLAG_UNUSED_6 = 1 << 6, /* cleared */
  ME_FLAG_UNUSED_7 = 1 << 7, /* cleared */
  ME_REMESH_REPROJECT_VERTEX_COLORS = 1 << 8,
  ME_DS_EXPAND = 1 << 9,
  ME_SCULPT_DYNAMIC_TOPOLOGY = 1 << 10,
  ME_FLAG_UNUSED_8 = 1 << 11, /* cleared */
  ME_REMESH_REPROJECT_PAINT_MASK = 1 << 12,
  ME_REMESH_FIX_POLES = 1 << 13,
  ME_REMESH_REPROJECT_VOLUME = 1 << 14,
  ME_REMESH_REPROJECT_SCULPT_FACE_SETS = 1 << 15,
};

/** #Mesh.cd_flag */
enum {
  ME_CDFLAG_VERT_BWEIGHT = 1 << 0,
  ME_CDFLAG_EDGE_BWEIGHT = 1 << 1,
  ME_CDFLAG_EDGE_CREASE = 1 << 2,
  ME_CDFLAG_VERT_CREASE = 1 << 3,
};

/** #Mesh.remesh_mode */
enum {
  REMESH_VOXEL = 0,
  REMESH_QUAD = 1,
};

/** #SubsurfModifierData.subdivType */
enum {
  ME_CC_SUBSURF = 0,
  ME_SIMPLE_SUBSURF = 1,
};

/** #Mesh.symmetry */
typedef enum eMeshSymmetryType {
  ME_SYMMETRY_X = 1 << 0,
  ME_SYMMETRY_Y = 1 << 1,
  ME_SYMMETRY_Z = 1 << 2,
} eMeshSymmetryType;

#define MESH_MAX_VERTS 2000000000L
