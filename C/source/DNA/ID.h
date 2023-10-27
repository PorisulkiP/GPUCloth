#pragma once

#include "sys_types.cuh"
#include "pointcache_types.cuh"

/* 2 characters for ID code and 64 for actual name */
#define MAX_ID_NAME 66


typedef struct IDPropertyUIData {
	/** Tooltip / property description pointer. Owned by the IDProperty. */
	char* description;
	/** RNA subtype, used for every type except string properties (PropertySubType). */
	int rna_subtype;

	char _pad[4];
} IDPropertyUIData;

/* IDP_UI_DATA_TYPE_INT */
typedef struct IDPropertyUIDataInt {
	IDPropertyUIData base;
	int* default_array; /* Only for array properties. */
	int default_array_len;
	char _pad[4];

	int min;
	int max;
	int soft_min;
	int soft_max;
	int step;
	int default_value;
} IDPropertyUIDataInt;

/* IDP_UI_DATA_TYPE_FLOAT */
typedef struct IDPropertyUIDataFloat {
	IDPropertyUIData base;
	double* default_array; /* Only for array properties. */
	int default_array_len;
	char _pad[4];

	float step;
	int precision;

	double min;
	double max;
	double soft_min;
	double soft_max;
	double default_value;
} IDPropertyUIDataFloat;

/* IDP_UI_DATA_TYPE_ID */
typedef struct IDPropertyUIDataID {
	IDPropertyUIData base;
} IDPropertyUIDataID;

typedef struct IDPropertyData {
	void* pointer;
	ListBase group;
	/** NOTE: we actually fit a double into these two 32bit integers. */
	int val, val2;
} IDPropertyData;

typedef struct IDOverrideLibraryRuntime {
	struct GHash* rna_path_to_override_properties;
	uint tag;
} IDOverrideLibraryRuntime;

/* IDOverrideLibraryRuntime->tag. */
enum {
	/** This override needs to be reloaded. */
	IDOVERRIDE_LIBRARY_RUNTIME_TAG_NEEDS_RELOAD = 1 << 0,
};

/* Main container for all overriding data info of a data-block. */
typedef struct IDOverrideLibrary {
	struct ID* reference;
	ListBase properties;
	struct ID* hierarchy_root;
	struct ID* storage;
	uint flag;
} IDOverrideLibrary;

typedef struct IDProperty {
	IDProperty* next, * prev;
	char type, subtype;
	short flag;
	char name[64];
	int saved;
	int len;
	int totallen;
} IDProperty;

/* There's a nasty circular dependency here.... 'void *' to the rescue! I
 * really wonder why this is needed. */
/* Здесь есть неприятная циклическая зависимость .... 'void *' на помощь! 
 * Мне действительно интересно, зачем это нужно. */
typedef struct ID {
	void* next, * prev;
	ID* newid;

	struct Library* lib;

	char name[66];
	short flag;
	int tag;
	int us;
	int icon_id;
	int recalc;
	unsigned int session_uuid;

	IDProperty* properties;
	IDOverrideLibrary* override_library;
	ID* orig_id;
} ID;


struct FileData;
struct GHash;
struct GPUTexture;
struct ID;
struct Library;
struct PackedFile;
struct UniqueName_Map;

/* Runtime display data */
struct DrawData;
typedef void (*DrawDataInitCb)(DrawData* engine_data);
typedef void (*DrawDataFreeCb)(DrawData* engine_data);

typedef struct DrawData {
	DrawData* next, * prev;
	struct DrawEngineType* engine_type;
	/* Only nested data, NOT the engine data itself. */
	DrawDataFreeCb free;
	/* Accumulated recalc flags, which corresponds to ID->recalc flags. */
	int recalc;
} DrawData;

typedef struct DrawDataList {
	DrawData* first, * last;
} DrawDataList;


#define MAX_IDPROP_NAME 64
#define DEFAULT_ALLOC_FOR_NULL_STRINGS 64

/*->type*/
typedef enum eIDPropertyType {
	IDP_STRING = 0,
	IDP_INT = 1,
	IDP_FLOAT = 2,
	/** Array containing int, floats, doubles or groups. */
	IDP_ARRAY = 5,
	IDP_GROUP = 6,
	IDP_ID = 7,
	IDP_DOUBLE = 8,
	IDP_IDPARRAY = 9,
} eIDPropertyType;
#define IDP_NUMTYPES 10

/** Used by some IDP utils, keep values in sync with type enum above. */
enum {
	IDP_TYPE_FILTER_STRING = 1 << 0,
	IDP_TYPE_FILTER_INT = 1 << 1,
	IDP_TYPE_FILTER_FLOAT = 1 << 2,
	IDP_TYPE_FILTER_ARRAY = 1 << 5,
	IDP_TYPE_FILTER_GROUP = 1 << 6,
	IDP_TYPE_FILTER_ID = 1 << 7,
	IDP_TYPE_FILTER_DOUBLE = 1 << 8,
	IDP_TYPE_FILTER_IDPARRAY = 1 << 9,
};

/*->subtype */

/* IDP_STRING */
enum {
	IDP_STRING_SUB_UTF8 = 0, /* default */
	IDP_STRING_SUB_BYTE = 1, /* arbitrary byte array, _not_ null terminated */
};

/*->flag*/
enum {
	/** This IDProp may be statically overridden.
	 * Should only be used/be relevant for custom properties. */
	IDP_FLAG_OVERRIDABLE_LIBRARY = 1 << 0,

	/** This collection item IDProp has been inserted in a local override.
	 * This is used by internal code to distinguish between library-originated items and
	 * local-inserted ones, as many operations are not allowed on the former. */
	 IDP_FLAG_OVERRIDELIBRARY_LOCAL = 1 << 1,

	 /** This means the property is set but RNA will return false when checking
	  * 'RNA_property_is_set', currently this is a runtime flag */
	  IDP_FLAG_GHOST = 1 << 7,
};

/* add any future new id property types here. */

/* Static ID override structs. */

typedef struct IDOverrideLibraryPropertyOperation {
	IDOverrideLibraryPropertyOperation* next, * prev;

	/* Type of override. */
	short operation;
	short flag;

	/** Runtime, tags are common to both IDOverrideProperty and IDOverridePropertyOperation. */
	short tag;
	char _pad0[2];

	/* Sub-item references, if needed (for arrays or collections only).
	 * We need both reference and local values to allow e.g. insertion into RNA collections
	 * (constraints, modifiers...).
	 * In RNA collection case, if names are defined, they are used in priority.
	 * Names are pointers (instead of char[64]) to save some space, NULL or empty string when unset.
	 * Indices are -1 when unset.
	 *
	 * NOTE: For insertion operations in RNA collections, reference may not actually exist in the
	 * linked reference data. It is used to identify the anchor of the insertion operation (i.e. the
	 * item after or before which the new local item should be inserted), in the local override. */
	char* subitem_reference_name;
	char* subitem_local_name;
	int subitem_reference_index;
	int subitem_local_index;
} IDOverrideLibraryPropertyOperation;

/* IDOverrideLibraryPropertyOperation->operation. */
enum {
	/* Basic operations. */
	IDOVERRIDE_LIBRARY_OP_NOOP = 0, /* Special value, forbids any overriding. */

	IDOVERRIDE_LIBRARY_OP_REPLACE = 1, /* Fully replace local value by reference one. */

	/* Numeric-only operations. */
	IDOVERRIDE_LIBRARY_OP_ADD = 101, /* Add local value to reference one. */
	/* Subtract local value from reference one (needed due to unsigned values etc.). */
	IDOVERRIDE_LIBRARY_OP_SUBTRACT = 102,
	/* Multiply reference value by local one (more useful than diff for scales and the like). */
	IDOVERRIDE_LIBRARY_OP_MULTIPLY = 103,

	/* Collection-only operations. */
	IDOVERRIDE_LIBRARY_OP_INSERT_AFTER = 201,  /* Insert after given reference's subitem. */
	IDOVERRIDE_LIBRARY_OP_INSERT_BEFORE = 202, /* Insert before given reference's subitem. */
	/* We can add more if needed (move, delete, ...). */
};

/* IDOverrideLibraryPropertyOperation->flag. */
enum {
	/** User cannot remove that override operation. */
	IDOVERRIDE_LIBRARY_FLAG_MANDATORY = 1 << 0,
	/** User cannot change that override operation. */
	IDOVERRIDE_LIBRARY_FLAG_LOCKED = 1 << 1,

	/** For overrides of ID pointers: this override still matches (follows) the hierarchy of the
	 *  reference linked data. */
	 IDOVERRIDE_LIBRARY_FLAG_IDPOINTER_MATCH_REFERENCE = 1 << 8,
};

/** A single overridden property, contain all operations on this one. */
typedef struct IDOverrideLibraryProperty {
	IDOverrideLibraryProperty* next, * prev;

	/**
	 * Path from ID to overridden property.
	 * *Does not* include indices/names for final arrays/collections items.
	 */
	char* rna_path;

	/**
	 * List of overriding operations (IDOverrideLibraryPropertyOperation) applied to this property.
	 */
	ListBase operations;

	/**
	 * Runtime, tags are common to both IDOverrideLibraryProperty and
	 * IDOverrideLibraryPropertyOperation. */
	short tag;
	char _pad[2];

	/** The property type matching the rna_path. */
	uint rna_prop_type;
} IDOverrideLibraryProperty;

/* IDOverrideLibraryProperty->tag and IDOverrideLibraryPropertyOperation->tag. */
enum {
	/** This override property (operation) is unused and should be removed by cleanup process. */
	IDOVERRIDE_LIBRARY_TAG_UNUSED = 1 << 0,
};


/* IDOverrideLibrary->flag */
enum {
	/**
	 * The override data-block should not be considered as part of an override hierarchy (generally
	 * because it was created as an single override, outside of any hierarchy consideration).
	 */
	IDOVERRIDE_LIBRARY_FLAG_NO_HIERARCHY = 1 << 0,
	/**
	 * The override ID is required for the system to work (because of ID dependencies), but is not
	 * seen as editable by the user.
	 */
	 IDOVERRIDE_LIBRARY_FLAG_SYSTEM_DEFINED = 1 << 1,
};

/* watch it: Sequence has identical beginning. */
/**
 * ID is the first thing included in all serializable types. It
 * provides a common handle to place all data in double-linked lists.
 */

 /* 2 characters for ID code and 64 for actual name */
#define MAX_ID_NAME 66

/** Status used and counters created during id-remapping. */
typedef struct ID_Runtime_Remap {
	/** Status during ID remapping. */
	int status;
	/** During ID remapping the number of skipped use cases that refcount the data-block. */
	int skipped_refcounted;
	/**
	 * During ID remapping the number of direct use cases that could be remapped
	 * (e.g. obdata when in edit mode).
	 */
	int skipped_direct;
	/** During ID remapping, the number of indirect use cases that could not be remapped. */
	int skipped_indirect;
} ID_Runtime_Remap;

typedef struct ID_Runtime {
	ID_Runtime_Remap remap;
} ID_Runtime;

typedef struct Library_Runtime {
	/* Used for efficient calculations of unique names. */
	UniqueName_Map* name_map;
} Library_Runtime;

/**
 * For each library file used, a Library struct is added to Main
 * WARNING: readfile.c, expand_doit() reads this struct without DNA check!
 */
typedef struct Library {
	ID id;
} Library;

/** #Library.tag */
enum eLibrary_Tag {
	/* Automatic recursive resync was needed when linking/loading data from that library. */
	LIBRARY_TAG_RESYNC_REQUIRED = 1 << 0,
};

/**
 * A weak library/ID reference for local data that has been appended, to allow re-using that local
 * data instead of creating a new copy of it in future appends.
 *
 * NOTE: This is by design a week reference, in other words code should be totally fine and perform
 * a regular append if it cannot find a valid matching local ID.
 *
 * NOTE: There should always be only one single ID in current Main matching a given linked
 * reference.
 */
typedef struct LibraryWeakReference {
	/**  Expected to match a `Library.filepath`. */
	char library_filepath[1024];

	/** MAX_ID_NAME. May be different from the current local ID name. */
	char library_id_name[66];

	char _pad[2];
} LibraryWeakReference;

/* for PreviewImage->flag */
enum ePreviewImage_Flag {
	PRV_CHANGED = (1 << 0),
	PRV_USER_EDITED = (1 << 1), /* if user-edited, do not auto-update this anymore! */
	PRV_RENDERING = (1 << 2),   /* Rendering was invoked. Cleared on file read. */
};

/* for PreviewImage->tag */
enum {
	PRV_TAG_DEFFERED = (1 << 0),           /* Actual loading of preview is deferred. */
	PRV_TAG_DEFFERED_RENDERING = (1 << 1), /* Deferred preview is being loaded. */
	PRV_TAG_DEFFERED_DELETE = (1 << 2),    /* Deferred preview should be deleted asap. */
};

typedef struct PreviewImage {
	/* All values of 2 are really NUM_ICON_SIZES */
	uint w[2];
	uint h[2];
	short flag[2];
	short changed_timestamp[2];
	uint* rect[2];

	/* Runtime-only data. */
	GPUTexture* gputexture[2];
	/** Used by previews outside of ID context. */
	int icon_id;

	/** Runtime data. */
	short tag;
	char _pad[2];
} PreviewImage;

#define PRV_DEFERRED_DATA(prv) \
  (CHECK_TYPE_INLINE(prv, PreviewImage *), \
   BLI_assert((prv)->tag & PRV_TAG_DEFFERED), \
   (void *)((prv) + 1))

#define ID_FAKE_USERS(id) ((((const ID *)id)->flag & LIB_FAKEUSER) ? 1 : 0)
#define ID_REAL_USERS(id) (((const ID *)id)->us - ID_FAKE_USERS(id))
#define ID_EXTRA_USERS(id) (((const ID *)id)->tag & LIB_TAG_EXTRAUSER ? 1 : 0)

#define ID_CHECK_UNDO(id) \
  ((GS((id)->name) != ID_SCR) && (GS((id)->name) != ID_WM) && (GS((id)->name) != ID_WS))

#define ID_BLEND_PATH(_bmain, _id) \
  ((_id)->lib ? (_id)->lib->filepath_abs : BKE_main_blendfile_path((_bmain)))
#define ID_BLEND_PATH_FROM_GLOBAL(_id) \
  ((_id)->lib ? (_id)->lib->filepath_abs : BKE_main_blendfile_path_from_global())

#define ID_MISSING(_id) ((((const ID *)(_id))->tag & LIB_TAG_MISSING) != 0)

#define ID_IS_LINKED(_id) (((const ID *)(_id))->lib != NULL)

/* Note that these are fairly high-level checks, should be used at user interaction level, not in
 * BKE_library_override typically (especially due to the check on LIB_TAG_EXTERN). */
#define ID_IS_OVERRIDABLE_LIBRARY_HIERARCHY(_id) \
  (ID_IS_LINKED(_id) && !ID_MISSING(_id) && \
   (BKE_idtype_get_info_from_id((const ID *)(_id))->flags & IDTYPE_FLAGS_NO_LIBLINKING) == 0 && \
   !ELEM(GS(((ID *)(_id))->name), ID_SCE))
#define ID_IS_OVERRIDABLE_LIBRARY(_id) \
  (ID_IS_OVERRIDABLE_LIBRARY_HIERARCHY((_id)) && (((const ID *)(_id))->tag & LIB_TAG_EXTERN) != 0)

 /* NOTE: The three checks below do not take into account whether given ID is linked or not (when
  * chaining overrides over several libraries). User must ensure the ID is not linked itself
  * currently. */
  /* TODO: add `_EDITABLE` versions of those macros (that would check if ID is linked or not)? */
#define ID_IS_OVERRIDE_LIBRARY_REAL(_id) \
  (((const ID *)(_id))->override_library != NULL && \
   ((const ID *)(_id))->override_library->reference != NULL)

#define ID_IS_OVERRIDE_LIBRARY_VIRTUAL(_id) \
  ((((const ID *)(_id))->flag & LIB_EMBEDDED_DATA_LIB_OVERRIDE) != 0)

#define ID_IS_OVERRIDE_LIBRARY(_id) \
  (ID_IS_OVERRIDE_LIBRARY_REAL(_id) || ID_IS_OVERRIDE_LIBRARY_VIRTUAL(_id))

#define ID_IS_OVERRIDE_LIBRARY_HIERARCHY_ROOT(_id) \
  (!ID_IS_OVERRIDE_LIBRARY_REAL(_id) || \
   ((ID *)(_id))->override_library->hierarchy_root == ((ID *)(_id)))

#define ID_IS_OVERRIDE_LIBRARY_TEMPLATE(_id) \
  (((ID *)(_id))->override_library != NULL && ((ID *)(_id))->override_library->reference == NULL)

#define ID_IS_ASSET(_id) (((const ID *)(_id))->asset_data != NULL)

/* Check whether datablock type is covered by copy-on-write. */
#define ID_TYPE_IS_COW(_id_type) \
  (!ELEM(_id_type, ID_LI, ID_IP, ID_SCR, ID_VF, ID_BR, ID_WM, ID_PAL, ID_PC, ID_WS, ID_IM))

/* Check whether data-block type requires copy-on-write from #ID_RECALC_PARAMETERS.
 * Keep in sync with #BKE_id_eval_properties_copy. */
#define ID_TYPE_SUPPORTS_PARAMS_WITHOUT_COW(id_type) ELEM(id_type, ID_ME)

#define ID_TYPE_IS_DEPRECATED(id_type) ELEM(id_type, ID_IP)

#ifdef GS
#  undef GS
#endif
#define GS(a) \
  (CHECK_TYPE_ANY(a, char *, const char *, char[66], const char[66]), \
   (ID_Type)(*((const short *)(a))))

#define ID_NEW_SET(_id, _idn) \
  (((ID *)(_id))->newid = (ID *)(_idn), \
   ((ID *)(_id))->newid->tag |= LIB_TAG_NEW, \
   (void *)((ID *)(_id))->newid)
#define ID_NEW_REMAP(a) \
  if ((a) && (a)->id.newid) { \
    (a) = (void *)(a)->id.newid; \
  } \
  ((void)0)

/**
 * This enum defines the index assigned to each type of IDs in the array returned by
 * #set_listbasepointers, and by extension, controls the default order in which each ID type is
 * processed during standard 'foreach' looping over all IDs of a #Main data-base.
 *
 * About Order:
 * ------------
 *
 * This is (loosely) defined with a relationship order in mind, from lowest level (ID types using,
 * referencing almost no other ID types) to highest level (ID types potentially using many other ID
 * types).
 *
 * So e.g. it ensures that this dependency chain is respected:
 *   #Material <- #Mesh <- #Object <- #Collection <- #Scene
 *
 * Default order of processing of IDs in 'foreach' macros (#FOREACH_MAIN_ID_BEGIN and the like),
 * built on top of #set_listbasepointers, is actually reversed compared to the order defined here,
 * since processing usually needs to happen on users before it happens on used IDs (when freeing
 * e.g.).
 *
 * DO NOT rely on this order as being full-proofed dependency order, there are many cases were it
 * can be violated (most obvious cases being custom properties and drivers, which can reference any
 * other ID types).
 *
 * However, this order can be considered as an optimization heuristic, especially when processing
 * relationships in a non-recursive pattern: in typical cases, a vast majority of those
 * relationships can be processed fine in the first pass, and only few additional passes are
 * required to address all remaining relationship cases.
 * See e.g. how #BKE_library_unused_linked_data_set_tag is doing this.
 */
enum {
	/* Special case: Library, should never ever depend on any other type. */
	INDEX_ID_LI = 0,

	/* Animation types, might be used by almost all other types. */
	INDEX_ID_IP, /* Deprecated. */
	INDEX_ID_AC,

	/* Grease Pencil, special case, should be with the other obdata, but it can also be used by many
	 * other ID types, including node trees e.g.
	 * So there is no proper place for those, for now keep close to the lower end of the processing
	 * hierarchy, but we may want to re-evaluate that at some point. */
	 INDEX_ID_GD,

	 /* Node trees, abstraction for procedural data, potentially used by many other ID types.
	  *
	  * NOTE: While node trees can also use many other ID types, they should not /own/ any of those,
	  * while they are being owned by many other ID types. This is why they are placed here. */
	  INDEX_ID_NT,

	  /* File-wrapper types, those usually 'embed' external files in Blender, with no dependencies to
	   * other ID types. */
	   INDEX_ID_VF,
	   INDEX_ID_TXT,
	   INDEX_ID_SO,

	   /* Image/movie types, can be used by shading ID types, but also directly by Objects, Scenes, etc.
		*/
		INDEX_ID_MSK,
		INDEX_ID_IM,
		INDEX_ID_MC,

		/* Shading types. */
		INDEX_ID_TE,
		INDEX_ID_MA,
		INDEX_ID_LS,
		INDEX_ID_WO,

		/* Simulation-related types. */
		INDEX_ID_CF,
		INDEX_ID_SIM,
		INDEX_ID_PA,

		/* Shape Keys snow-flake, can be used by several obdata types. */
		INDEX_ID_KE,

		/* Object data types. */
		INDEX_ID_AR,
		INDEX_ID_ME,
		INDEX_ID_CU_LEGACY,
		INDEX_ID_MB,
		INDEX_ID_CV,
		INDEX_ID_PT,
		INDEX_ID_VO,
		INDEX_ID_LT,
		INDEX_ID_LA,
		INDEX_ID_CA,
		INDEX_ID_SPK,
		INDEX_ID_LP,

		/* Collection and object types. */
		INDEX_ID_OB,
		INDEX_ID_GR,

		/* Preset-like, not-really-data types, can use many other ID types but should never be used by
		 * any actual data type (besides Scene, due to tool settings). */
		 INDEX_ID_PAL,
		 INDEX_ID_PC,
		 INDEX_ID_BR,

		 /* Scene, after preset-like ID types because of tool settings. */
		 INDEX_ID_SCE,

		 /* UI-related types, should never be used by any other data type. */
		 INDEX_ID_SCR,
		 INDEX_ID_WS,
		 INDEX_ID_WM,

		 /* Special values. */
		 INDEX_ID_NULL,
		 INDEX_ID_MAX,
};


struct Object;

typedef struct Collection {
	ID id;
	/** CollectionObject. */
	ListBase gobject;
	/** CollectionChild. */
	ListBase children;
} Collection;


typedef struct CollectionObject {
	CollectionObject* next, * prev;
	Object* ob;
} CollectionObject;

typedef struct CollectionChild {
	CollectionChild* next, * prev;
	Collection* collection;
} CollectionChild;

enum eCollectionLineArt_Usage {
	COLLECTION_LRT_INCLUDE = 0,
	COLLECTION_LRT_OCCLUSION_ONLY = (1 << 0),
	COLLECTION_LRT_EXCLUDE = (1 << 1),
	COLLECTION_LRT_INTERSECTION_ONLY = (1 << 2),
	COLLECTION_LRT_NO_INTERSECTION = (1 << 3),
};

enum eCollectionLineArt_Flags {
	COLLECTION_LRT_USE_INTERSECTION_MASK = (1 << 0),
	COLLECTION_LRT_USE_INTERSECTION_PRIORITY = (1 << 1),
};

/* Collection->flag */
enum {
	COLLECTION_HIDE_VIEWPORT = (1 << 0),             /* Disable in viewports. */
	COLLECTION_HIDE_SELECT = (1 << 1),               /* Not selectable in viewport. */
	/* COLLECTION_DISABLED_DEPRECATED = (1 << 2), */ /* Not used anymore */
	COLLECTION_HIDE_RENDER = (1 << 3),               /* Disable in renders. */
	COLLECTION_HAS_OBJECT_CACHE = (1 << 4),          /* Runtime: object_cache is populated. */
	COLLECTION_IS_MASTER = (1 << 5), /* Is master collection embedded in the scene. */
	COLLECTION_HAS_OBJECT_CACHE_INSTANCED = (1 << 6), /* for object_cache_instanced. */
};

/* Collection->tag */
enum {
	/* That code (BKE_main_collections_parent_relations_rebuild and the like)
	 * is called from very low-level places, like e.g ID remapping...
	 * Using a generic tag like LIB_TAG_DOIT for this is just impossible, we need our very own. */
	COLLECTION_TAG_RELATION_REBUILD = (1 << 0),
};

/* Collection->color_tag. */
typedef enum CollectionColorTag {
	COLLECTION_COLOR_NONE = -1,
	COLLECTION_COLOR_01,
	COLLECTION_COLOR_02,
	COLLECTION_COLOR_03,
	COLLECTION_COLOR_04,
	COLLECTION_COLOR_05,
	COLLECTION_COLOR_06,
	COLLECTION_COLOR_07,
	COLLECTION_COLOR_08,

	COLLECTION_COLOR_TOT,
} CollectionColorTag;
