/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2001-2002 by NaN Holding BV.
 * All rights reserved.
 */

#pragma once

/** \file
 * \ingroup bke
 */

#include "compiler_attrs.cuh"


struct BMEditMesh;
struct BPoint;
struct Depsgraph;
struct Lattice;
struct MDeformVert;
struct Main;
struct Mesh;
struct Object;
struct Scene;

void BKE_lattice_resize(struct Lattice *lt, int u, int v, int w, struct Object *ltOb);
struct Lattice *BKE_lattice_add(struct Main *bmain, const char *name);
void calc_lat_fudu(int flag, int res, float *r_fu, float *r_du);

bool object_deform_mball(struct Object *ob, struct ListBase *dispbase);
void outside_lattice(struct Lattice *lt);

float (*BKE_lattice_vert_coords_alloc(const struct Lattice *lt, int *r_vert_len))[3];
void BKE_lattice_vert_coords_get(const struct Lattice *lt, float (*vert_coords)[3]);
void BKE_lattice_vert_coords_apply_with_mat4(struct Lattice *lt,
                                             const float (*vert_coords)[3],
                                             const float mat[4][4]);
void BKE_lattice_vert_coords_apply(struct Lattice *lt, const float (*vert_coords)[3]);
void BKE_lattice_modifiers_calc(struct Depsgraph *depsgraph,
                                struct Scene *scene,
                                struct Object *ob);

struct MDeformVert *BKE_lattice_deform_verts_get(const struct Object *oblatt);
struct BPoint *BKE_lattice_active_point_get(struct Lattice *lt);

struct BoundBox *BKE_lattice_boundbox_get(struct Object *ob);
void BKE_lattice_minmax_dl(struct Object *ob, struct Lattice *lt, float min[3], float max[3]);
void BKE_lattice_minmax(struct Lattice *lt, float min[3], float max[3]);
void BKE_lattice_center_median(struct Lattice *lt, float cent[3]);
void BKE_lattice_center_bounds(struct Lattice *lt, float cent[3]);
void BKE_lattice_translate(struct Lattice *lt, const float offset[3], bool do_keys);
void BKE_lattice_transform(struct Lattice *lt, const float mat[4][4], bool do_keys);

bool BKE_lattice_is_any_selected(const struct Lattice *lt);

int BKE_lattice_index_from_uvw(struct Lattice *lt, const int u, const int v, const int w);
void BKE_lattice_index_to_uvw(struct Lattice *lt, const int index, int *r_u, int *r_v, int *r_w);
int BKE_lattice_index_flip(
    struct Lattice *lt, const int index, const bool flip_u, const bool flip_v, const bool flip_w);
void BKE_lattice_bitmap_from_flag(struct Lattice *lt,
                                  unsigned int *bitmap,
                                  const uint8_t flag,
                                  const bool clear,
                                  const bool respecthide);

/* **** Depsgraph evaluation **** */

struct Depsgraph;

void BKE_lattice_eval_geometry(struct Depsgraph *depsgraph, struct Lattice *latt);

/* Draw Cache */
enum {
  BKE_LATTICE_BATCH_DIRTY_ALL = 0,
  BKE_LATTICE_BATCH_DIRTY_SELECT,
};
void BKE_lattice_batch_cache_dirty_tag(struct Lattice *lt, int mode);
void BKE_lattice_batch_cache_free(struct Lattice *lt);

extern void (*BKE_lattice_batch_cache_dirty_tag_cb)(struct Lattice *lt, int mode);
extern void (*BKE_lattice_batch_cache_free_cb)(struct Lattice *lt);

/* -------------------------------------------------------------------- */
/** \name Deform 3D Coordinates by Lattice (lattice_deform.c)
 * \{ */

struct LatticeDeformData *BKE_lattice_deform_data_create(
    const struct Object *oblatt, const struct Object *ob) ATTR_WARN_UNUSED_RESULT;
void BKE_lattice_deform_data_eval_co(struct LatticeDeformData *lattice_deform_data,
                                     float co[3],
                                     float weight);
void BKE_lattice_deform_data_destroy(struct LatticeDeformData *lattice_deform_data);

void BKE_lattice_deform_coords(const struct Object *ob_lattice,
                               const struct Object *ob_target,
                               float (*vert_coords)[3],
                               const int vert_coords_len,
                               const short flag,
                               const char *defgrp_name,
                               float fac);

void BKE_lattice_deform_coords_with_mesh(const struct Object *ob_lattice,
                                         const struct Object *ob_target,
                                         float (*vert_coords)[3],
                                         const int vert_coords_len,
                                         const short flag,
                                         const char *defgrp_name,
                                         const float fac,
                                         const struct Mesh *me_target);

void BKE_lattice_deform_coords_with_editmesh(const struct Object *ob_lattice,
                                             const struct Object *ob_target,
                                             float (*vert_coords)[3],
                                             const int vert_coords_len,
                                             const short flag,
                                             const char *defgrp_name,
                                             const float fac,
                                             struct BMEditMesh *em_target);
/** \} */
