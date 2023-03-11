#pragma once

/* types */

/** vector of two shorts. */
typedef struct vec2s {
  short x, y;
} vec2s;

/** vector of two floats. */
typedef struct vec2f {
  float x, y;
} vec2f;

typedef struct vec3f {
  float x, y, z;
} vec3f;

/** integer rectangle. */
typedef struct rcti {
  int xmin, xmax;
  int ymin, ymax;
} rcti;

/** float rectangle. */
typedef struct rctf {
  float xmin, xmax;
  float ymin, ymax;
} rctf;

/** dual quaternion. */
typedef struct DualQuat {
  float quat[4];
  float trans[4];

  float scale[4][4];
  float scale_weight;
} DualQuat;
