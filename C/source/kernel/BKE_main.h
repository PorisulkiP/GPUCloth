#pragma once
#include "listBase.h"

#include "compiler_attrs.cuh"
#include "sys_types.cuh"

struct BLI_mempool;
struct BlendThumbnail;
struct GHash;
struct GSet;
struct ImBuf;
struct Library;
struct MainLock;

/* Blender thumbnail, as written on file (width, height, and data as char RGBA). */
/* We pack pixel data after that struct. */
typedef struct BlendThumbnail {
  int width, height;
  char rect[0];
} BlendThumbnail;

/* Structs caching relations between data-blocks in a given Main. */
typedef struct MainIDRelationsEntryItem {
  struct MainIDRelationsEntryItem *next;

  union {
    /* For `from_ids` list, a user of the hashed ID. */
    struct ID *from;
    /* For `to_ids` list, an ID used by the hashed ID. */
    struct ID **to;
  } id_pointer;
  /* Session uuid of the `id_pointer`. */
  uint session_uuid;

  int usage_flag; /* Using IDWALK_ enums, defined in lib_query.h */
} MainIDRelationsEntryItem;

typedef struct MainIDRelationsEntry {
  /* Linked list of IDs using that ID. */
  struct MainIDRelationsEntryItem *from_ids;
  /* Linked list of IDs used by that ID. */
  struct MainIDRelationsEntryItem *to_ids;

  /* Session uuid of the ID matching that entry. */
  uint session_uuid;

  /* Runtime tags, users should ensure those are reset after usage. */
  uint tags;
} MainIDRelationsEntry;

/* MainIDRelationsEntry.tags */
typedef enum MainIDRelationsEntryTags {
  /* Generic tag marking the entry as to be processed. */
  MAINIDRELATIONS_ENTRY_TAGS_DOIT = 1 << 0,
  /* Generic tag marking the entry as processed. */
  MAINIDRELATIONS_ENTRY_TAGS_PROCESSED = 1 << 1,
} MainIDRelationsEntryTags;

typedef struct MainIDRelations {
  /* Mapping from an ID pointer to all of its parents (IDs using it) and children (IDs it uses).
   * Values are `MainIDRelationsEntry` pointers. */
  struct GHash *relations_from_pointers;
  /* Note: we could add more mappings when needed (e.g. from session uuid?). */

  short flag;

  /* Private... */
  struct BLI_mempool *entry_items_pool;
} MainIDRelations;

enum {
  /* Those bmain relations include pointers/usages from editors. */
  MAINIDRELATIONS_INCLUDE_UI = 1 << 0,
};

typedef struct Main {
  struct Main *next, *prev;
  char name[1024];                   /* 1024 = FILE_MAX */
  short versionfile, subversionfile; /* see BLENDER_FILE_VERSION, BLENDER_FILE_SUBVERSION */
  short minversionfile, minsubversionfile;
  uint64_t build_commit_timestamp; /* commit's timestamp from buildinfo */
  char build_hash[16];             /* hash from buildinfo */
  char recovered;                  /* indicate the main->name (file) is the recovered one */
  /** All current ID's exist in the last memfile undo step. */
  char is_memfile_undo_written;
  /**
   * An ID needs its data to be flushed back.
   * use "needs_flush_to_id" in edit data to flag data which needs updating.
   */
  char is_memfile_undo_flush_needed;
  /**
   * Indicates that next memfile undo step should not allow reusing old bmain when re-read, but
   * instead do a complete full re-read/update from stored memfile.
   */
  char use_memfile_full_barrier;

  /**
   * When linking, disallow creation of new data-blocks.
   * Make sure we don't do this by accident, see T76738.
   */
  char is_locked_for_linking;

  BlendThumbnail *blen_thumb;

  struct Library *curlib;
  ListBase scenes;
  ListBase libraries;
  ListBase objects;
  ListBase meshes;
  ListBase curves;
  ListBase metaballs;
  ListBase materials;
  ListBase textures;
  ListBase images;
  ListBase lattices;
  ListBase lights;
  ListBase cameras;
  ListBase ipo; /* Deprecated (only for versioning). */
  ListBase shapekeys;
  ListBase worlds;
  ListBase screens;
  ListBase fonts;
  ListBase texts;
  ListBase speakers;
  ListBase lightprobes;
  ListBase sounds;
  ListBase collections;
  ListBase armatures;
  ListBase actions;
  ListBase nodetrees;
  ListBase brushes;
  ListBase particles;
  ListBase palettes;
  ListBase paintcurves;
  ListBase wm; /* Singleton (exception). */
  ListBase gpencils;
  ListBase movieclips;
  ListBase masks;
  ListBase linestyles;
  ListBase cachefiles;
  ListBase workspaces;
  ListBase hairs;
  ListBase pointclouds;
  ListBase volumes;
  ListBase simulations;

  /**
   * Must be generated, used and freed by same code - never assume this is valid data unless you
   * know when, who and how it was created.
   * Used by code doing a lot of remapping etc. at once to speed things up.
   */
  struct MainIDRelations *relations;

  struct MainLock *lock;
} Main;

struct Main *BKE_main_new(void);
void BKE_main_free(struct Main *mainvar);

void BKE_main_lock(struct Main *bmain);
void BKE_main_unlock(struct Main *bmain);

void BKE_main_relations_create(struct Main *bmain, const short flag);
void BKE_main_relations_free(struct Main *bmain);
void BKE_main_relations_tag_set(struct Main *bmain,
                                const MainIDRelationsEntryTags tag,
                                const bool value);

struct GSet *BKE_main_gset_create(struct Main *bmain, struct GSet *gset);

/* *** Generic utils to loop over whole Main database. *** */

#define FOREACH_MAIN_LISTBASE_ID_BEGIN(_lb, _id) \
  { \
    ID *_id_next = (ID *)(_lb)->first; \
    for ((_id) = _id_next; (_id) != NULL; (_id) = _id_next) { \
      _id_next = (ID *)(_id)->next;

#define FOREACH_MAIN_LISTBASE_ID_END \
  } \
  } \
  ((void)0)

#define FOREACH_MAIN_LISTBASE_BEGIN(_bmain, _lb) \
  { \
    ListBase *_lbarray[INDEX_ID_MAX]; \
    int _i = set_listbasepointers((_bmain), _lbarray); \
    while (_i--) { \
      (_lb) = _lbarray[_i];

#define FOREACH_MAIN_LISTBASE_END \
  } \
  } \
  ((void)0)

/**
 * DO NOT use break statement with that macro,
 * use #FOREACH_MAIN_LISTBASE and #FOREACH_MAIN_LISTBASE_ID instead
 * if you need that kind of control flow. */
#define FOREACH_MAIN_ID_BEGIN(_bmain, _id) \
  { ListBase *_lb; \
    FOREACH_MAIN_LISTBASE_BEGIN ((_bmain), _lb) { \
      FOREACH_MAIN_LISTBASE_ID_BEGIN (_lb, (_id))

#define FOREACH_MAIN_ID_END \
  FOREACH_MAIN_LISTBASE_ID_END; } \
  FOREACH_MAIN_LISTBASE_END; } ((void)0)

struct BlendThumbnail *BKE_main_thumbnail_from_imbuf(struct Main *bmain, struct ImBuf *img);
struct ImBuf *BKE_main_thumbnail_to_imbuf(struct Main *bmain, struct BlendThumbnail *data);
void BKE_main_thumbnail_create(struct Main *bmain);

const char *BKE_main_blendfile_path(const struct Main *bmain) ATTR_NONNULL();
//const char *BKE_main_blendfile_path_from_global(void);

struct ListBase *which_libbase(struct Main *bmain, short type);

int set_listbasepointers(Main* bmain, ListBase** lb);

#define MAIN_VERSION_ATLEAST(main, ver, subver) \
  ((main)->versionfile > (ver) || \
   ((main)->versionfile == (ver) && (main)->subversionfile >= (subver)))

#define MAIN_VERSION_OLDER(main, ver, subver) \
  ((main)->versionfile < (ver) || \
   ((main)->versionfile == (ver) && (main)->subversionfile < (subver)))

/**
 * The size of thumbnails (optionally) stored in the `.blend` files header.
 *
 * NOTE(@campbellbarton): This is kept small as it's stored uncompressed in the `.blend` file,
 * where a larger size would increase the size of every `.blend` file unreasonably.
 * If we wanted to increase the size, we'd want to use compression (JPEG or similar).
 */
#define BLEN_THUMB_SIZE 128

#define BLEN_THUMB_MEMSIZE(_x, _y) \
  (sizeof(BlendThumbnail) + ((size_t)(_x) * (size_t)(_y)) * sizeof(int))
 /** Protect against buffer overflow vulnerability & negative sizes. */
#define BLEN_THUMB_MEMSIZE_IS_VALID(_x, _y) \
  (((_x) > 0 && (_y) > 0) && ((uint64_t)(_x) * (uint64_t)(_y) < (SIZE_MAX / (sizeof(int) * 4))))

