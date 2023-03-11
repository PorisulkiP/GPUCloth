#pragma once

#include "BLI_bitmap.h"
#include "ghash.h"
#include "types.h"
#include "listBase.h"
#include "ID.h"

#include "utildefines.h"

typedef uint BLI_bitmap;

/* warning: the bitmap does not keep track of its own size or check
 * for out-of-bounds access */

 /* internal use */
 /* 2^5 = 32 (bits) */
#define _BITMAP_POWER 5
/* 0b11111 */
#define _BITMAP_MASK 31

/* number of blocks needed to hold '_tot' bits */
#define _BITMAP_NUM_BLOCKS(_tot) (((_tot) >> _BITMAP_POWER) + 1)

/* size (in bytes) used to hold '_tot' bits */
#define BLI_BITMAP_SIZE(_tot) ((size_t)(_BITMAP_NUM_BLOCKS(_tot)) * sizeof(BLI_bitmap))

/* allocate memory for a bitmap with '_tot' bits; free with MEM_freeN() */
#define BLI_BITMAP_NEW(_tot, _alloc_string) \
  ((BLI_bitmap *)MEM_callocN(BLI_BITMAP_SIZE(_tot), _alloc_string))

/* allocate a bitmap on the stack */
#define BLI_BITMAP_NEW_ALLOCA(_tot) \
  ((BLI_bitmap *)memset(alloca(BLI_BITMAP_SIZE(_tot)), 0, BLI_BITMAP_SIZE(_tot)))

/* Allocate using given MemArena */
#define BLI_BITMAP_NEW_MEMARENA(_mem, _tot) \
  (CHECK_TYPE_INLINE(_mem, MemArena *), \
   ((BLI_bitmap *)BLI_memarena_calloc(_mem, BLI_BITMAP_SIZE(_tot))))

/* get the value of a single bit at '_index' */
#define BLI_BITMAP_TEST(_bitmap, _index) \
  (CHECK_TYPE_ANY(_bitmap, BLI_bitmap *, const BLI_bitmap *), \
   ((_bitmap)[(_index) >> _BITMAP_POWER] & (1u << ((_index)&_BITMAP_MASK))))

#define BLI_BITMAP_TEST_AND_SET_ATOMIC(_bitmap, _index) \
  (CHECK_TYPE_ANY(_bitmap, BLI_bitmap *, const BLI_bitmap *), \
   (atomic_fetch_and_or_uint32((uint32_t *)&(_bitmap)[(_index) >> _BITMAP_POWER], \
                               (1u << ((_index)&_BITMAP_MASK))) & \
    (1u << ((_index)&_BITMAP_MASK))))

#define BLI_BITMAP_TEST_BOOL(_bitmap, _index) \
  (CHECK_TYPE_ANY(_bitmap, BLI_bitmap *, const BLI_bitmap *), \
   (BLI_BITMAP_TEST(_bitmap, _index) != 0))

/* set the value of a single bit at '_index' */
#define BLI_BITMAP_ENABLE(_bitmap, _index) \
  (CHECK_TYPE_ANY(_bitmap, BLI_bitmap *, const BLI_bitmap *), \
   ((_bitmap)[(_index) >> _BITMAP_POWER] |= (1u << ((_index)&_BITMAP_MASK))))

/* clear the value of a single bit at '_index' */
#define BLI_BITMAP_DISABLE(_bitmap, _index) \
  (CHECK_TYPE_ANY(_bitmap, BLI_bitmap *, const BLI_bitmap *), \
   ((_bitmap)[(_index) >> _BITMAP_POWER] &= ~(1u << ((_index)&_BITMAP_MASK))))

/* flip the value of a single bit at '_index' */
#define BLI_BITMAP_FLIP(_bitmap, _index) \
  (CHECK_TYPE_ANY(_bitmap, BLI_bitmap *, const BLI_bitmap *), \
   ((_bitmap)[(_index) >> _BITMAP_POWER] ^= (1u << ((_index)&_BITMAP_MASK))))

/* set or clear the value of a single bit at '_index' */
#define BLI_BITMAP_SET(_bitmap, _index, _set) \
  { \
    CHECK_TYPE(_bitmap, BLI_bitmap *); \
    if (_set) { \
      BLI_BITMAP_ENABLE(_bitmap, _index); \
    } \
    else { \
      BLI_BITMAP_DISABLE(_bitmap, _index); \
    } \
  } \
  (void)0

/* resize bitmap to have space for '_tot' bits */
#define BLI_BITMAP_RESIZE(_bitmap, _tot) \
  { \
    CHECK_TYPE(_bitmap, BLI_bitmap *); \
    (_bitmap) = MEM_recallocN(_bitmap, BLI_BITMAP_SIZE(_tot)); \
  } \
  (void)0

typedef struct NlaTrack {
    struct NlaTrack* next, * prev;

    /** BActionStrips in this track. */
    ListBase strips;

    /** Settings for this track. */
    int flag;
    /** Index of the track in the stack
     * \note not really useful, but we need a '_pad' var anyways! */
    int index;

    /** Short user-description of this track - `MAX_ID_NAME - 2`. */
    char name[64];
} NlaTrack;

typedef struct bAction {
    /** ID-serialization for relinking. */
    ID id;

    /** Function-curves (FCurve). */
    ListBase curves;
    /** Legacy data - Action Channels (bActionChannel) in pre-2.5 animation system. */
    ListBase chanbase;
    /** Groups of function-curves (bActionGroup). */
    ListBase groups;
    /** Markers local to the Action (used to provide Pose-Libraries). */
    ListBase markers;

    /** Settings for this action. */
    int flag;
    /** Index of the active marker. */
    int active_marker;

    /**
     * Type of ID-blocks that action can be assigned to
     * (if 0, will be set to whatever ID first evaluates it).
     */
    int idroot;
    char _pad[4];

    /** Start and end of the manually set intended playback frame range. Used by UI and
     *  some editing tools, but doesn't directly affect animation evaluation in any way. */
    float frame_start, frame_end;

    PreviewImage* preview;
} bAction;

typedef struct NlaStrip {
    struct NlaStrip* next, * prev;

    /** 'Child' strips (used for 'meta' strips). */
    ListBase strips;
    /** Action that is referenced by this strip (strip is 'user' of the action). */
    bAction* act;

    /** F-Curves for controlling this strip's influence and timing */ /* TODO: move out? */
    ListBase fcurves;
    /** F-Curve modifiers to be applied to the entire strip's referenced F-Curves. */
    ListBase modifiers;

    /** User-Visible Identifier for Strip - `MAX_ID_NAME - 2`. */
    char name[64];

    /** Influence of strip. */
    float influence;
    /** Current 'time' within action being used (automatically evaluated, but can be overridden). */
    float strip_time;

    /** Extents of the strip. */
    float start, end;
    /** Range of the action to use. */
    float actstart, actend;

    /** The number of times to repeat the action range (only when no F-Curves). */
    float repeat;
    /** The amount the action range is scaled by (only when no F-Curves). */
    float scale;

    /** Strip blending length (only used when there are no F-Curves). */
    float blendin, blendout;
    /** Strip blending mode (layer-based mixing). */
    short blendmode;

    /** Strip extrapolation mode (time-based mixing). */
    short extendmode;
    char _pad1[2];

    /** Type of NLA strip. */
    short type;

    /** Handle for speaker objects. */
    void* speaker_handle;

    /** Settings. */
    int flag;
    char _pad2[4];

    /* Pointer to an original NLA strip. */
    struct NlaStrip* orig_strip;

    void* _pad3;
} NlaStrip;


struct AnimationEvalContext;

/* --------------- NLA Evaluation DataTypes ----------------------- */

/* used for list of strips to accumulate at current time */
typedef struct NlaEvalStrip {
  struct NlaEvalStrip *next, *prev;

  NlaTrack *track; /* track that this strip belongs to */
  NlaStrip *strip; /* strip that's being used */

  short track_index; /* the index of the track within the list */
  short strip_mode;  /* which end of the strip are we looking at */

  float strip_time; /* time at which which strip is being evaluated */
} NlaEvalStrip;

/* NlaEvalStrip->strip_mode */
enum eNlaEvalStrip_StripMode {
  /* standard evaluation */
  NES_TIME_BEFORE = -1,
  NES_TIME_WITHIN,
  NES_TIME_AFTER,

  /* transition-strip evaluations */
  NES_TIME_TRANSITION_START,
  NES_TIME_TRANSITION_END,
};

struct NlaEvalChannel;
struct NlaEvalData;

/* Unique channel key for GHash. */
typedef struct NlaEvalChannelKey {
  struct PointerRNA ptr;
  struct PropertyRNA *prop;
} NlaEvalChannelKey;

/* Bitmask of array indices touched by actions. */
typedef struct NlaValidMask {
  BLI_bitmap *ptr;
  BLI_bitmap buffer[sizeof(uint64_t) / sizeof(BLI_bitmap)];
} NlaValidMask;

/* Set of property values for blending. */
typedef struct NlaEvalChannelSnapshot {
  struct NlaEvalChannel *channel;

  /** For an upper snapshot channel, marks values that should be blended. */
  NlaValidMask blend_domain;

  int length;   /* Number of values in the property. */
  bool is_base; /* Base snapshot of the channel. */

  float values[]; /* Item values. */
  /* Memory over-allocated to provide space for values. */
} NlaEvalChannelSnapshot;

/* NlaEvalChannel->mix_mode */
enum eNlaEvalChannel_MixMode {
  NEC_MIX_ADD,
  NEC_MIX_MULTIPLY,
  NEC_MIX_QUATERNION,
  NEC_MIX_AXIS_ANGLE,
};

/* Temp channel for accumulating data from NLA for a single property.
 * Handles array properties as a unit to allow intelligent blending. */
typedef struct NlaEvalChannel {
  struct NlaEvalChannel *next, *prev;
  struct NlaEvalData *owner;

  /* Original RNA path string and property key. */
  const char *rna_path;
  NlaEvalChannelKey key;

  int index;
  bool is_array;
  char mix_mode;

  /* Associated with the RNA property's value(s), marks which elements are affected by NLA. */
  NlaValidMask domain;

  /* Base set of values. */
  NlaEvalChannelSnapshot base_snapshot;
  /* Memory over-allocated to provide space for base_snapshot.values. */
} NlaEvalChannel;

/* Set of values for all channels. */
typedef struct NlaEvalSnapshot {
  /* Snapshot this one defaults to. */
  struct NlaEvalSnapshot *base;

  int size;
  NlaEvalChannelSnapshot **channels;
} NlaEvalSnapshot;

/* Set of all channels covered by NLA. */
typedef struct NlaEvalData {
  ListBase channels;

  /* Mapping of paths and NlaEvalChannelKeys to channels. */
  GHash *path_hash;
  GHash *key_hash;

  /* Base snapshot. */
  int num_channels;
  NlaEvalSnapshot base_snapshot;

  /* Evaluation result shapshot. */
  NlaEvalSnapshot eval_snapshot;
} NlaEvalData;

/* Information about the currently edited strip and ones below it for keyframing. */
typedef struct NlaKeyframingContext {
  struct NlaKeyframingContext *next, *prev;

  /* AnimData for which this context was built. */
  struct AnimData *adt;

  /* Data of the currently edited strip (copy, or fake strip for the main action). */
  NlaStrip strip;
  NlaEvalStrip *eval_strip;

  /* Evaluated NLA stack below the tweak strip. */
  NlaEvalData lower_eval_data;
} NlaKeyframingContext;

/* --------------- NLA Functions (not to be used as a proper API) ----------------------- */

/* convert from strip time <-> global time */
float nlastrip_get_frame(NlaStrip *strip, float cframe, short mode);

/* --------------- NLA Evaluation (very-private stuff) ----------------------- */
/* these functions are only defined here to avoid problems with the order
 * in which they get defined. */

NlaEvalStrip *nlastrips_ctime_get_strip(ListBase *list,
                                        ListBase *strips,
                                        short index,
                                        const struct AnimationEvalContext *anim_eval_context,
                                        const bool flush_to_original);
void nlastrip_evaluate(PointerRNA *ptr,
                       NlaEvalData *channels,
                       ListBase *modifiers,
                       NlaEvalStrip *nes,
                       NlaEvalSnapshot *snapshot,
                       const struct AnimationEvalContext *anim_eval_context,
                       const bool flush_to_original);
void nladata_flush_channels(PointerRNA *ptr,
                            NlaEvalData *channels,
                            NlaEvalSnapshot *snapshot,
                            const bool flush_to_original);

void nlasnapshot_enable_all_blend_domain(NlaEvalSnapshot *snapshot);

void nlasnapshot_ensure_channels(NlaEvalData *eval_data, NlaEvalSnapshot *snapshot);

void nlasnapshot_blend(NlaEvalData *eval_data,
                       NlaEvalSnapshot *lower_snapshot,
                       NlaEvalSnapshot *upper_snapshot,
                       const short upper_blendmode,
                       const float upper_influence,
                       NlaEvalSnapshot *r_blended_snapshot);
