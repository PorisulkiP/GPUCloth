#pragma once

#include "threads.h"

struct GHash;
struct MovieTracking;
struct MovieTrackingMarker;

struct libmv_CameraIntrinsicsOptions;

typedef struct TracksMap {
  char object_name[255];
  bool is_camera;

  int num_tracks;
  int customdata_size;

  char *customdata;
  MovieTrackingTrack *tracks;

  struct GHash *hash;

  int ptr;

  /* Spin lock is used to sync context during tracking. */
  SpinLock spin_lock;
} TracksMap;

struct libmv_FrameAccessor;

#define MAX_ACCESSOR_CLIP 64
typedef struct TrackingImageAccessor {
  struct MovieClip *clips[MAX_ACCESSOR_CLIP];
  int num_clips;

  /* Array of tracks which are being tracked.
   * Points to actual track from the `MovieClip` (or multiple of them).
   * This accessor owns the array, but not the tracks themselves. */
  struct MovieTrackingTrack **tracks;
  int num_tracks;

  struct libmv_FrameAccessor *libmv_accessor;
  SpinLock cache_lock;
} TrackingImageAccessor;
