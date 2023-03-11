/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2005 Blender Foundation. All rights reserved. */

/** \file
 * \ingroup DNA
 */

#pragma once

#include "ID.h"

struct CurveMapping;
struct Image;
struct MTex;
struct Material;

typedef struct BrushClone {
  /** Image for clone tool. */
  struct Image *image;
  /** Offset of clone image from canvas. */
  float offset[2];
  /** Transparency for drawing of clone image. */
  float alpha;
  char _pad[4];
} BrushClone;

typedef struct BrushGpencilSettings {
  /** Amount of smoothing to apply to newly created strokes. */
  float draw_smoothfac;
  /** Fill zoom factor */
  float fill_factor;
  /** Amount of alpha strength to apply to newly created strokes. */
  float draw_strength;
  /** Amount of jitter to apply to newly created strokes. */
  float draw_jitter;
  /** Angle when the brush has full thickness. */
  float draw_angle;
  /** Factor to apply when angle change (only 90 degrees). */
  float draw_angle_factor;
  /** Factor of randomness for pressure. */
  float draw_random_press;
  /** Factor of strength for strength. */
  float draw_random_strength;
  /** Number of times to apply smooth factor to new strokes. */
  short draw_smoothlvl;
  /** Number of times to subdivide new strokes. */
  short draw_subdivide;
  /** Layers used for fill. */
  short fill_layer_mode;
  short fill_direction;

  /** Factor for transparency. */
  float fill_threshold;
  /** Number of pixel to consider the leak is too small (x 2). */
  short fill_leak;
  /* Type of caps: eGPDstroke_Caps. */
  int8_t caps_type;
  char _pad;

  int flag2;

  /** Number of simplify steps. */
  int fill_simplylvl;
  /** Type of control lines drawing mode. */
  int fill_draw_mode;
  /** Icon identifier. */
  int icon_id;

  /** Maximum distance before generate new point for very fast mouse movements. */
  int input_samples;
  /** Random factor for UV rotation. */
  float uv_random;
  /** Moved to 'Brush.gpencil_tool'. */
  int brush_type;
  /** Soft, hard or stroke. */
  int eraser_mode;
  /** Smooth while drawing factor. */
  float active_smooth;
  /** Factor to apply to strength for soft eraser. */
  float era_strength_f;
  /** Factor to apply to thickness for soft eraser. */
  float era_thickness_f;
  /** Internal grease pencil drawing flags. */
  int flag;

  /** gradient control along y for color */
  float hardeness;
  /** factor xy of shape for dots gradients */
  float aspect_ratio[2];
  /** Simplify adaptive factor */
  float simplify_f;

  /** Mix color-factor. */
  float vertex_factor;
  int vertex_mode;

  /** eGP_Sculpt_Flag. */
  int sculpt_flag;
  /** eGP_Sculpt_Mode_Flag. */
  int sculpt_mode_flag;
  /** Preset type (used to reset brushes - internal). */
  short preset_type;
  /** Brush preselected mode (Active/Material/Vertex-color). */
  short brush_draw_mode;

  /** Randomness for Hue. */
  float random_hue;
  /** Randomness for Saturation. */
  float random_saturation;
  /** Randomness for Value. */
  float random_value;

  /** Factor to extend stroke extremes using fill tool. */
  float fill_extend_fac;
  /** Number of pixels to dilate fill area. */
  int dilate_pixels;

  struct CurveMapping *curve_sensitivity;
  struct CurveMapping *curve_strength;
  struct CurveMapping *curve_jitter;
  struct CurveMapping *curve_rand_pressure;
  struct CurveMapping *curve_rand_strength;
  struct CurveMapping *curve_rand_uv;
  struct CurveMapping *curve_rand_hue;
  struct CurveMapping *curve_rand_saturation;
  struct CurveMapping *curve_rand_value;

  /* optional link of material to replace default in context */
  /** Material. */
  struct Material *material;
} BrushGpencilSettings;

typedef struct BrushCurvesSculptSettings {
  /** Number of curves added by the add brush. */
  int add_amount;
  /** Number of control points in new curves added by the add brush. */
  int points_per_curve;
  /* eBrushCurvesSculptFlag. */
  uint32_t flag;
  /** When shrinking curves, they shouldn't become shorter than this length. */
  float minimum_length;
  /** Length of newly added curves when it is not interpolated from other curves. */
  float curve_length;
  /** Minimum distance between curve root points used by the Density brush. */
  float minimum_distance;
  /** How often the Density brush tries to add a new curve. */
  int density_add_attempts;
  /** #eBrushCurvesSculptDensityMode. */
  uint8_t density_mode;
  char _pad[7];
} BrushCurvesSculptSettings;

/* Struct to hold palette colors for sorting. */
typedef struct tPaletteColorHSV {
  float rgb[3];
  float value;
  float h;
  float s;
  float v;
} tPaletteColorHSV;

typedef struct PaletteColor {
  struct PaletteColor *next, *prev;
  /* two values, one to store rgb, other to store values for sculpt/weight */
  float rgb[3];
  float value;
} PaletteColor;

typedef struct Palette {
  ID id;

  /** Pointer to individual colors. */
  ListBase colors;

  int active_color;
  char _pad[4];
} Palette;
