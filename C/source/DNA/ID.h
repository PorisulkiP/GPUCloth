#pragma once

#include "sys_types.cuh"
#include "pointcache_types.h"

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
	/** Reference linked ID which this one overrides. */
	struct ID* reference;
	/** List of IDOverrideLibraryProperty structs. */
	ListBase properties;

	/** Override hierarchy root ID. Usually the actual root of the hierarchy, but not always
	 * in degenerated cases.
	 *
	 * All liboverrides of a same hierarchy (e.g. a character collection) share the same root.
	 */
	struct ID* hierarchy_root;

	/* Read/write data. */
	/* Temp ID storing extra override data (used for differential operations only currently).
	 * Always NULL outside of read/write context. */
	struct ID* storage;

	//IDOverrideLibraryRuntime* runtime;

	uint flag;
	//char _pad_1[4];
} IDOverrideLibrary;

typedef struct IDProperty {
	struct IDProperty* next, * prev;
	char type, subtype;
	short flag;
	/** MAX_IDPROP_NAME. */
	char name[64];

	/* saved is used to indicate if this struct has been saved yet.
	 * seemed like a good idea as a '_pad' var was needed anyway :) */
	int saved;
	/** NOTE: alignment for 64 bits. */
	//IDPropertyData data;

	/* Array length, also (this is important!) string length + 1.
	 * the idea is to be able to reuse array realloc functions on strings. */
	int len;

	/* Strings and arrays are both buffered, though the buffer isn't saved. */
	/* totallen is total length of allocated array/string, including a buffer.
	 * Note that the buffering is mild; the code comes from python's list implementation. */
	int totallen;

	//IDPropertyUIData* ui_data;
} IDProperty;

/* There's a nasty circular dependency here.... 'void *' to the rescue! I
 * really wonder why this is needed. */
/* Здесь есть неприятная циклическая зависимость .... 'void *' на помощь! 
 * Мне действительно интересно, зачем это нужно. */
typedef struct ID {
	void* next, * prev;
	struct ID* newid;

	struct Library* lib;

	/** If the ID is an asset, this pointer is set. Owning pointer. */
	/** Если идентификатор является активом, этот указатель устанавливается. Владеющий указателем. */
	//struct AssetMetaData* asset_data;

	/** MAX_ID_NAME. */
	char name[66];
	/**
	 * LIB_... flags report on status of the data-block this ID belongs to
	 * (persistent, saved to and read from .blend).
	 */
	short flag;
	/**
	 * LIB_TAG_... tags (runtime only, cleared at read time).
	 */
	int tag;
	int us;
	int icon_id;
	int recalc;
	/**
	 * Used by undo code. recalc_after_undo_push contains the changes between the
	 * last undo push and the current state. This is accumulated as IDs are tagged
	 * for update in the depsgraph, and only cleared on undo push.
	 *
	 * recalc_up_to_undo_push is saved to undo memory, and is the value of
	 * recalc_after_undo_push at the time of the undo push. This means it can be
	 * used to find the changes between undo states.
	 */
	//int recalc_up_to_undo_push;
	//int recalc_after_undo_push;

	/**
	 * A session-wide unique identifier for a given ID, that remain the same across potential
	 * re-allocations (e.g. due to undo/redo steps).
	 */
	unsigned int session_uuid;

	IDProperty* properties;

	/** Reference linked ID which this one overrides. */
	IDOverrideLibrary* override_library;

	/**
	 * Only set for data-blocks which are coming from copy-on-write, points to
	 * the original version of it.
	 * Also used temporarily during memfile undo to keep a reference to old ID when found.
	 */
	struct ID* orig_id;

	/**
	 * Holds the #PyObject reference to the ID (initialized on demand).
	 *
	 * This isn't essential, it could be removed however it gives some advantages:
	 *
	 * - Every time the #ID is accessed a #BPy_StructRNA doesn't have to be created & destroyed
	 *   (consider all the polling and drawing functions that access ID's).
	 *
	 * - When this #ID is deleted, the #BPy_StructRNA can be invalidated
	 *   so accessing it from Python raises an exception instead of crashing.
	 *
	 *   This is of limited benefit though, as it doesn't apply to non #ID data
	 *   that references this ID (the bones of an armature or the modifiers of an object for e.g.).
	 */
	//void* py_instance;
	//void* _pad1;
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
typedef void (*DrawDataInitCb)(struct DrawData* engine_data);
typedef void (*DrawDataFreeCb)(struct DrawData* engine_data);

typedef struct DrawData {
	struct DrawData* next, * prev;
	struct DrawEngineType* engine_type;
	/* Only nested data, NOT the engine data itself. */
	DrawDataFreeCb free;
	/* Accumulated recalc flags, which corresponds to ID->recalc flags. */
	int recalc;
} DrawData;

typedef struct DrawDataList {
	struct DrawData* first, * last;
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
	struct IDOverrideLibraryPropertyOperation* next, * prev;

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
	struct IDOverrideLibraryProperty* next, * prev;

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
	struct UniqueName_Map* name_map;
} Library_Runtime;

/**
 * For each library file used, a Library struct is added to Main
 * WARNING: readfile.c, expand_doit() reads this struct without DNA check!
 */
typedef struct Library {
	ID id;
	//struct FileData* filedata;
	/** Path name used for reading, can be relative and edited in the outliner. */
	//char filepath[1024];

	/**
	 * Run-time only, absolute file-path (set on read).
	 * This is only for convenience, `filepath` is the real path
	 * used on file read but in some cases its useful to access the absolute one.
	 *
	 * Use #BKE_library_filepath_set() rather than setting `filepath`
	 * directly and it will be kept in sync - campbell
	 */
	//char filepath_abs[1024];

	/** Set for indirectly linked libs, used in the outliner and while reading. */
	//struct Library* parent;

	//struct PackedFile* packedfile;

	//ushort tag;
	//char _pad_0[6];

	/** Temp data needed by read/write code, and lib-override recursive re-synchronized. */
	//int temp_index;
	/** See BLENDER_FILE_VERSION, BLENDER_FILE_SUBVERSION, needed for do_versions. */
	//short versionfile, subversionfile;

	//struct Library_Runtime runtime;
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
	struct GPUTexture* gputexture[2];
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

	//struct PreviewImage* preview;

	//uint layer;
	//float instance_offset[3];

	//short flag;
	/* Runtime-only, always cleared on file load. */
	//short tag;

	//short lineart_usage;         /* eCollectionLineArt_Usage */
	//unsigned char lineart_flags; /* eCollectionLineArt_Flags */
	//unsigned char lineart_intersection_mask;
	//unsigned char lineart_intersection_priority;
	//char _pad[5];

	//int16_t color_tag;

	/* Runtime. Cache of objects in this collection and all its
	 * children. This is created on demand when e.g. some physics
	 * simulation needs it, we don't want to have it for every
	 * collections due to memory usage reasons. */
	//ListBase object_cache;

	/* Need this for line art sub-collection selections. */
	//ListBase object_cache_instanced;

	/* Runtime. List of collections that are a parent of this
	 * datablock. */
	//ListBase parents;

	/* Deprecated */
	//struct SceneCollection* collection;
	//struct ViewLayer* view_layer;
} Collection;


typedef struct CollectionObject {
	struct CollectionObject* next, * prev;
	struct Object* ob;
} CollectionObject;

typedef struct CollectionChild {
	struct CollectionChild* next, * prev;
	struct Collection* collection;
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
