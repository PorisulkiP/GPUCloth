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

/** \file
 * \ingroup bke
 *
 * Functions to evaluate mesh data.
 */

#include <limits.h>

#include "MEM_guardedalloc.cuh"

#include "DNA_mesh_types.h"
#include "meshdata_types.cuh"
#include "object_types.cuh"

#include "BLI_alloca.h"
#include "BLI_bitmap.h"
#include "edgehash.h"
#include "linklist.cuh"
#include "BLI_linklist_stack.h"
#include "math.cuh"
#include "memarena.cuh"
#include "BLI_polyfill_2d.h"
#include "BLI_stack.h"
#include "BLI_task.h"
#include "utildefines.h"

#include "BKE_customdata.h"
#include "BKE_editmesh_cache.h"
#include "BKE_global.h"
#include "BKE_mesh.h"
#include "BKE_multires.h"
#include "BKE_report.h"

/* -------------------------------------------------------------------- */
/** \name Mesh Normal Calculation
 * \{ */

/**
 * Call when there are no polygons.
 */
static void mesh_calc_normals_vert_fallback(MVert *mverts, int numVerts)
{
  for (int i = 0; i < numVerts; i++) {
    MVert *mv = &mverts[i];
    float no[3];

    normalize_v3_v3(no, mv->co);
    normal_float_to_short_v3(mv->no, no);
  }
}

/* TODO(Sybren): we can probably rename this to BKE_mesh_calc_normals_mapping(),
 * and remove the function of the same name below, as that one doesn't seem to be
 * called anywhere. */
void BKE_mesh_calc_normals_mapping_simple(struct Mesh *mesh)
{
  const bool only_face_normals = CustomData_is_referenced_layer(&mesh->vdata, CD_MVERT);

  BKE_mesh_calc_normals_mapping_ex(mesh->mvert,
                                   mesh->totvert,
                                   mesh->mloop,
                                   mesh->mpoly,
                                   mesh->totloop,
                                   mesh->totpoly,
                                   NULL,
                                   mesh->mface,
                                   mesh->totface,
                                   NULL,
                                   NULL,
                                   only_face_normals);
}

/* Calculate vertex and face normals, face normals are returned in *r_faceNors if non-NULL
 * and vertex normals are stored in actual mverts.
 */
void BKE_mesh_calc_normals_mapping(MVert *mverts,
                                   int numVerts,
                                   const MLoop *mloop,
                                   const MPoly *mpolys,
                                   int numLoops,
                                   int numPolys,
                                   float (*r_polyNors)[3],
                                   const MFace *mfaces,
                                   int numFaces,
                                   const int *origIndexFace,
                                   float (*r_faceNors)[3])
{
  BKE_mesh_calc_normals_mapping_ex(mverts,
                                   numVerts,
                                   mloop,
                                   mpolys,
                                   numLoops,
                                   numPolys,
                                   r_polyNors,
                                   mfaces,
                                   numFaces,
                                   origIndexFace,
                                   r_faceNors,
                                   false);
}

typedef struct MeshCalcNormalsData {
  const MPoly *mpolys;
  const MLoop *mloop;
  MVert *mverts;
  float (*pnors)[3];
  float (*lnors_weighted)[3];
  float (*vnors)[3];
} MeshCalcNormalsData;

static void mesh_calc_normals_poly_cb(void *__restrict userdata,
                                      const int pidx,
                                      const TaskParallelTLS *__restrict UNUSED(tls))
{
  MeshCalcNormalsData *data = userdata;
  const MPoly *mp = &data->mpolys[pidx];

  BKE_mesh_calc_poly_normal(mp, data->mloop + mp->loopstart, data->mverts, data->pnors[pidx]);
}

static void mesh_calc_normals_poly_finalize_cb(void *__restrict userdata,
                                               const int vidx,
                                               const TaskParallelTLS *__restrict UNUSED(tls))
{
  MeshCalcNormalsData *data = userdata;

  MVert *mv = &data->mverts[vidx];
  float *no = data->vnors[vidx];

  if (UNLIKELY(normalize_v3(no) == 0.0f)) {
    /* following Mesh convention; we use vertex coordinate itself for normal in this case */
    normalize_v3_v3(no, mv->co);
  }

  normal_float_to_short_v3(mv->no, no);
}

void BKE_lnor_spacearr_clear(MLoopNorSpaceArray *lnors_spacearr)
{
  lnors_spacearr->num_spaces = 0;
  lnors_spacearr->lspacearr = NULL;
  lnors_spacearr->loops_pool = NULL;
  if (lnors_spacearr->mem != NULL) {
    BLI_memarena_clear(lnors_spacearr->mem);
  }
}

void BKE_lnor_spacearr_free(MLoopNorSpaceArray *lnors_spacearr)
{
  lnors_spacearr->num_spaces = 0;
  lnors_spacearr->lspacearr = NULL;
  lnors_spacearr->loops_pool = NULL;
  BLI_memarena_free(lnors_spacearr->mem);
  lnors_spacearr->mem = NULL;
}

MLoopNorSpace *BKE_lnor_space_create(MLoopNorSpaceArray *lnors_spacearr)
{
  lnors_spacearr->num_spaces++;
  return BLI_memarena_calloc(lnors_spacearr->mem, sizeof(MLoopNorSpace));
}

/* This threshold is a bit touchy (usual float precision issue), this value seems OK. */
#define LNOR_SPACE_TRIGO_THRESHOLD (1.0f - 1e-4f)

/* Should only be called once.
 * Beware, this modifies ref_vec and other_vec in place!
 * In case no valid space can be generated, ref_alpha and ref_beta are set to zero
 * (which means 'use auto lnors').
 */
void BKE_lnor_space_define(MLoopNorSpace *lnor_space,
                           const float lnor[3],
                           float vec_ref[3],
                           float vec_other[3],
                           BLI_Stack *edge_vectors)
{
  const float pi2 = (float)M_PI * 2.0f;
  float tvec[3], dtp;
  const float dtp_ref = dot_v3v3(vec_ref, lnor);
  const float dtp_other = dot_v3v3(vec_other, lnor);

  if (UNLIKELY(fabsf(dtp_ref) >= LNOR_SPACE_TRIGO_THRESHOLD ||
               fabsf(dtp_other) >= LNOR_SPACE_TRIGO_THRESHOLD)) {
    /* If vec_ref or vec_other are too much aligned with lnor, we can't build lnor space,
     * tag it as invalid and abort. */
    lnor_space->ref_alpha = lnor_space->ref_beta = 0.0f;

    if (edge_vectors) {
      BLI_stack_clear(edge_vectors);
    }
    return;
  }

  copy_v3_v3(lnor_space->vec_lnor, lnor);

  /* Compute ref alpha, average angle of all available edge vectors to lnor. */
  if (edge_vectors) {
    float alpha = 0.0f;
    int nbr = 0;
    while (!BLI_stack_is_empty(edge_vectors)) {
      const float *vec = BLI_stack_peek(edge_vectors);
      alpha += saacosf(dot_v3v3(vec, lnor));
      BLI_stack_discard(edge_vectors);
      nbr++;
    }
    /* Note: In theory, this could be 'nbr > 2',
     *       but there is one case where we only have two edges for two loops:
     *       a smooth vertex with only two edges and two faces (our Monkey's nose has that, e.g.).
     */
    BLI_assert(nbr >= 2); /* This piece of code shall only be called for more than one loop... */
    lnor_space->ref_alpha = alpha / (float)nbr;
  }
  else {
    lnor_space->ref_alpha = (saacosf(dot_v3v3(vec_ref, lnor)) +
                             saacosf(dot_v3v3(vec_other, lnor))) /
                            2.0f;
  }

  /* Project vec_ref on lnor's ortho plane. */
  mul_v3_v3fl(tvec, lnor, dtp_ref);
  sub_v3_v3(vec_ref, tvec);
  normalize_v3_v3(lnor_space->vec_ref, vec_ref);

  cross_v3_v3v3(tvec, lnor, lnor_space->vec_ref);
  normalize_v3_v3(lnor_space->vec_ortho, tvec);

  /* Project vec_other on lnor's ortho plane. */
  mul_v3_v3fl(tvec, lnor, dtp_other);
  sub_v3_v3(vec_other, tvec);
  normalize_v3(vec_other);

  /* Beta is angle between ref_vec and other_vec, around lnor. */
  dtp = dot_v3v3(lnor_space->vec_ref, vec_other);
  if (LIKELY(dtp < LNOR_SPACE_TRIGO_THRESHOLD)) {
    const float beta = saacos(dtp);
    lnor_space->ref_beta = (dot_v3v3(lnor_space->vec_ortho, vec_other) < 0.0f) ? pi2 - beta : beta;
  }
  else {
    lnor_space->ref_beta = pi2;
  }
}

/**
 * Add a new given loop to given lnor_space.
 * Depending on \a lnor_space->data_type, we expect \a bm_loop to be a pointer to BMLoop struct
 * (in case of BMLOOP_PTR), or NULL (in case of LOOP_INDEX), loop index is then stored in pointer.
 * If \a is_single is set, the BMLoop or loop index is directly stored in \a lnor_space->loops
 * pointer (since there is only one loop in this fan),
 * else it is added to the linked list of loops in the fan.
 */
void BKE_lnor_space_add_loop(MLoopNorSpaceArray *lnors_spacearr,
                             MLoopNorSpace *lnor_space,
                             const int ml_index,
                             void *bm_loop,
                             const bool is_single)
{
  BLI_assert((lnors_spacearr->data_type == MLNOR_SPACEARR_LOOP_INDEX && bm_loop == NULL) ||
             (lnors_spacearr->data_type == MLNOR_SPACEARR_BMLOOP_PTR && bm_loop != NULL));

  lnors_spacearr->lspacearr[ml_index] = lnor_space;
  if (bm_loop == NULL) {
    bm_loop = POINTER_FROM_INT(ml_index);
  }
  if (is_single) {
    BLI_assert(lnor_space->loops == NULL);
    lnor_space->flags |= MLNOR_SPACE_IS_SINGLE;
    lnor_space->loops = bm_loop;
  }
  else {
    BLI_assert((lnor_space->flags & MLNOR_SPACE_IS_SINGLE) == 0);
    BLI_linklist_prepend_nlink(&lnor_space->loops, bm_loop, &lnors_spacearr->loops_pool[ml_index]);
  }
}

float unit_short_to_float(const short val)
{
  return (float)val / (float)SHRT_MAX;
}

short unit_float_to_short(const float val)
{
  /* Rounding... */
  return (short)floorf(val * (float)SHRT_MAX + 0.5f);
}

void BKE_lnor_space_custom_data_to_normal(MLoopNorSpace *lnor_space,
                                          const short clnor_data[2],
                                          float r_custom_lnor[3])
{
  /* NOP custom normal data or invalid lnor space, return. */
  if (clnor_data[0] == 0 || lnor_space->ref_alpha == 0.0f || lnor_space->ref_beta == 0.0f) {
    copy_v3_v3(r_custom_lnor, lnor_space->vec_lnor);
    return;
  }

  {
    /* TODO Check whether using sincosf() gives any noticeable benefit
     *      (could not even get it working under linux though)! */
    const float pi2 = (float)(M_PI * 2.0);
    const float alphafac = unit_short_to_float(clnor_data[0]);
    const float alpha = (alphafac > 0.0f ? lnor_space->ref_alpha : pi2 - lnor_space->ref_alpha) *
                        alphafac;
    const float betafac = unit_short_to_float(clnor_data[1]);

    mul_v3_v3fl(r_custom_lnor, lnor_space->vec_lnor, cosf(alpha));

    if (betafac == 0.0f) {
      madd_v3_v3fl(r_custom_lnor, lnor_space->vec_ref, sinf(alpha));
    }
    else {
      const float sinalpha = sinf(alpha);
      const float beta = (betafac > 0.0f ? lnor_space->ref_beta : pi2 - lnor_space->ref_beta) *
                         betafac;
      madd_v3_v3fl(r_custom_lnor, lnor_space->vec_ref, sinalpha * cosf(beta));
      madd_v3_v3fl(r_custom_lnor, lnor_space->vec_ortho, sinalpha * sinf(beta));
    }
  }
}

void BKE_lnor_space_custom_normal_to_data(MLoopNorSpace *lnor_space,
                                          const float custom_lnor[3],
                                          short r_clnor_data[2])
{
  /* We use null vector as NOP custom normal (can be simpler than giving autocomputed lnor...). */
  if (is_zero_v3(custom_lnor) || compare_v3v3(lnor_space->vec_lnor, custom_lnor, 1e-4f)) {
    r_clnor_data[0] = r_clnor_data[1] = 0;
    return;
  }

  {
    const float pi2 = (float)(M_PI * 2.0);
    const float cos_alpha = dot_v3v3(lnor_space->vec_lnor, custom_lnor);
    float vec[3], cos_beta;
    float alpha;

    alpha = saacosf(cos_alpha);
    if (alpha > lnor_space->ref_alpha) {
      /* Note we could stick to [0, pi] range here,
       * but makes decoding more complex, not worth it. */
      r_clnor_data[0] = unit_float_to_short(-(pi2 - alpha) / (pi2 - lnor_space->ref_alpha));
    }
    else {
      r_clnor_data[0] = unit_float_to_short(alpha / lnor_space->ref_alpha);
    }

    /* Project custom lnor on (vec_ref, vec_ortho) plane. */
    mul_v3_v3fl(vec, lnor_space->vec_lnor, -cos_alpha);
    add_v3_v3(vec, custom_lnor);
    normalize_v3(vec);

    cos_beta = dot_v3v3(lnor_space->vec_ref, vec);

    if (cos_beta < LNOR_SPACE_TRIGO_THRESHOLD) {
      float beta = saacosf(cos_beta);
      if (dot_v3v3(lnor_space->vec_ortho, vec) < 0.0f) {
        beta = pi2 - beta;
      }

      if (beta > lnor_space->ref_beta) {
        r_clnor_data[1] = unit_float_to_short(-(pi2 - beta) / (pi2 - lnor_space->ref_beta));
      }
      else {
        r_clnor_data[1] = unit_float_to_short(beta / lnor_space->ref_beta);
      }
    }
    else {
      r_clnor_data[1] = 0;
    }
  }
}

#define LOOP_SPLIT_TASK_BLOCK_SIZE 1024

typedef struct LoopSplitTaskData {
  /* Specific to each instance (each task). */

  /** We have to create those outside of tasks, since afaik memarena is not threadsafe. */
  MLoopNorSpace *lnor_space;
  float (*lnor)[3];
  const MLoop *ml_curr;
  const MLoop *ml_prev;
  int ml_curr_index;
  int ml_prev_index;
  /** Also used a flag to switch between single or fan process! */
  const int *e2l_prev;
  int mp_index;

  /** This one is special, it's owned and managed by worker tasks,
   * avoid to have to create it for each fan! */
  BLI_Stack *edge_vectors;

  char pad_c;
} LoopSplitTaskData;

typedef struct LoopSplitTaskDataCommon {
  /* Read/write.
   * Note we do not need to protect it, though, since two different tasks will *always* affect
   * different elements in the arrays. */
  MLoopNorSpaceArray *lnors_spacearr;
  float (*loopnors)[3];
  short (*clnors_data)[2];

  /* Read-only. */
  const MVert *mverts;
  const MEdge *medges;
  const MLoop *mloops;
  const MPoly *mpolys;
  int (*edge_to_loops)[2];
  int *loop_to_poly;
  const float (*polynors)[3];

  int numEdges;
  int numLoops;
  int numPolys;
} LoopSplitTaskDataCommon;

#define INDEX_UNSET INT_MIN
#define INDEX_INVALID -1
/* See comment about edge_to_loops below. */
#define IS_EDGE_SHARP(_e2l) (ELEM((_e2l)[1], INDEX_UNSET, INDEX_INVALID))

static void mesh_edges_sharp_tag(LoopSplitTaskDataCommon *data,
                                 const bool check_angle,
                                 const float split_angle,
                                 const bool do_sharp_edges_tag)
{
  const MVert *mverts = data->mverts;
  const MEdge *medges = data->medges;
  const MLoop *mloops = data->mloops;

  const MPoly *mpolys = data->mpolys;

  const int numEdges = data->numEdges;
  const int numPolys = data->numPolys;

  float(*loopnors)[3] = data->loopnors; /* Note: loopnors may be NULL here. */
  const float(*polynors)[3] = data->polynors;

  int(*edge_to_loops)[2] = data->edge_to_loops;
  int *loop_to_poly = data->loop_to_poly;

  BLI_bitmap *sharp_edges = do_sharp_edges_tag ? BLI_BITMAP_NEW(numEdges, __func__) : NULL;

  const MPoly *mp;
  int mp_index;

  const float split_angle_cos = check_angle ? cosf(split_angle) : -1.0f;

  for (mp = mpolys, mp_index = 0; mp_index < numPolys; mp++, mp_index++) {
    const MLoop *ml_curr;
    int *e2l;
    int ml_curr_index = mp->loopstart;
    const int ml_last_index = (ml_curr_index + mp->totloop) - 1;

    ml_curr = &mloops[ml_curr_index];

    for (; ml_curr_index <= ml_last_index; ml_curr++, ml_curr_index++) {
      e2l = edge_to_loops[ml_curr->e];

      loop_to_poly[ml_curr_index] = mp_index;

      /* Pre-populate all loop normals as if their verts were all-smooth,
       * this way we don't have to compute those later!
       */
      if (loopnors) {
        normal_short_to_float_v3(loopnors[ml_curr_index], mverts[ml_curr->v].no);
      }

      /* Check whether current edge might be smooth or sharp */
      if ((e2l[0] | e2l[1]) == 0) {
        /* 'Empty' edge until now, set e2l[0] (and e2l[1] to INDEX_UNSET to tag it as unset). */
        e2l[0] = ml_curr_index;
        /* We have to check this here too, else we might miss some flat faces!!! */
        e2l[1] = (mp->flag & ME_SMOOTH) ? INDEX_UNSET : INDEX_INVALID;
      }
      else if (e2l[1] == INDEX_UNSET) {
        const bool is_angle_sharp = (check_angle &&
                                     dot_v3v3(polynors[loop_to_poly[e2l[0]]], polynors[mp_index]) <
                                         split_angle_cos);

        /* Second loop using this edge, time to test its sharpness.
         * An edge is sharp if it is tagged as such, or its face is not smooth,
         * or both poly have opposed (flipped) normals, i.e. both loops on the same edge share the
         * same vertex, or angle between both its polys' normals is above split_angle value.
         */
        if (!(mp->flag & ME_SMOOTH) || (medges[ml_curr->e].flag & ME_SHARP) ||
            ml_curr->v == mloops[e2l[0]].v || is_angle_sharp) {
          /* Note: we are sure that loop != 0 here ;) */
          e2l[1] = INDEX_INVALID;

          /* We want to avoid tagging edges as sharp when it is already defined as such by
           * other causes than angle threshold... */
          if (do_sharp_edges_tag && is_angle_sharp) {
            BLI_BITMAP_SET(sharp_edges, ml_curr->e, true);
          }
        }
        else {
          e2l[1] = ml_curr_index;
        }
      }
      else if (!IS_EDGE_SHARP(e2l)) {
        /* More than two loops using this edge, tag as sharp if not yet done. */
        e2l[1] = INDEX_INVALID;

        /* We want to avoid tagging edges as sharp when it is already defined as such by
         * other causes than angle threshold... */
        if (do_sharp_edges_tag) {
          BLI_BITMAP_SET(sharp_edges, ml_curr->e, false);
        }
      }
      /* Else, edge is already 'disqualified' (i.e. sharp)! */
    }
  }

  /* If requested, do actual tagging of edges as sharp in another loop. */
  if (do_sharp_edges_tag) {
    MEdge *me;
    int me_index;
    for (me = (MEdge *)medges, me_index = 0; me_index < numEdges; me++, me_index++) {
      if (BLI_BITMAP_TEST(sharp_edges, me_index)) {
        me->flag |= ME_SHARP;
      }
    }

    MEM_freeN(sharp_edges);
  }
}

void BKE_mesh_loop_manifold_fan_around_vert_next(const MLoop *mloops,
                                                 const MPoly *mpolys,
                                                 const int *loop_to_poly,
                                                 const int *e2lfan_curr,
                                                 const uint mv_pivot_index,
                                                 const MLoop **r_mlfan_curr,
                                                 int *r_mlfan_curr_index,
                                                 int *r_mlfan_vert_index,
                                                 int *r_mpfan_curr_index)
{
  const MLoop *mlfan_next;
  const MPoly *mpfan_next;

  /* Warning! This is rather complex!
   * We have to find our next edge around the vertex (fan mode).
   * First we find the next loop, which is either previous or next to mlfan_curr_index, depending
   * whether both loops using current edge are in the same direction or not, and whether
   * mlfan_curr_index actually uses the vertex we are fanning around!
   * mlfan_curr_index is the index of mlfan_next here, and mlfan_next is not the real next one
   * (i.e. not the future mlfan_curr)...
   */
  *r_mlfan_curr_index = (e2lfan_curr[0] == *r_mlfan_curr_index) ? e2lfan_curr[1] : e2lfan_curr[0];
  *r_mpfan_curr_index = loop_to_poly[*r_mlfan_curr_index];

  BLI_assert(*r_mlfan_curr_index >= 0);
  BLI_assert(*r_mpfan_curr_index >= 0);

  mlfan_next = &mloops[*r_mlfan_curr_index];
  mpfan_next = &mpolys[*r_mpfan_curr_index];
  if (((*r_mlfan_curr)->v == mlfan_next->v && (*r_mlfan_curr)->v == mv_pivot_index) ||
      ((*r_mlfan_curr)->v != mlfan_next->v && (*r_mlfan_curr)->v != mv_pivot_index)) {
    /* We need the previous loop, but current one is our vertex's loop. */
    *r_mlfan_vert_index = *r_mlfan_curr_index;
    if (--(*r_mlfan_curr_index) < mpfan_next->loopstart) {
      *r_mlfan_curr_index = mpfan_next->loopstart + mpfan_next->totloop - 1;
    }
  }
  else {
    /* We need the next loop, which is also our vertex's loop. */
    if (++(*r_mlfan_curr_index) >= mpfan_next->loopstart + mpfan_next->totloop) {
      *r_mlfan_curr_index = mpfan_next->loopstart;
    }
    *r_mlfan_vert_index = *r_mlfan_curr_index;
  }
  *r_mlfan_curr = &mloops[*r_mlfan_curr_index];
  /* And now we are back in sync, mlfan_curr_index is the index of mlfan_curr! Pff! */
}

static void split_loop_nor_single_do(LoopSplitTaskDataCommon *common_data, LoopSplitTaskData *data)
{
  MLoopNorSpaceArray *lnors_spacearr = common_data->lnors_spacearr;
  const short(*clnors_data)[2] = common_data->clnors_data;

  const MVert *mverts = common_data->mverts;
  const MEdge *medges = common_data->medges;
  const float(*polynors)[3] = common_data->polynors;

  MLoopNorSpace *lnor_space = data->lnor_space;
  float(*lnor)[3] = data->lnor;
  const MLoop *ml_curr = data->ml_curr;
  const MLoop *ml_prev = data->ml_prev;
  const int ml_curr_index = data->ml_curr_index;
#if 0 /* Not needed for 'single' loop. */
  const int ml_prev_index = data->ml_prev_index;
  const int *e2l_prev = data->e2l_prev;
#endif
  const int mp_index = data->mp_index;

  /* Simple case (both edges around that vertex are sharp in current polygon),
   * this loop just takes its poly normal.
   */
  copy_v3_v3(*lnor, polynors[mp_index]);

#if 0
  printf("BASIC: handling loop %d / edge %d / vert %d / poly %d\n",
         ml_curr_index,
         ml_curr->e,
         ml_curr->v,
         mp_index);
#endif

  /* If needed, generate this (simple!) lnor space. */
  if (lnors_spacearr) {
    float vec_curr[3], vec_prev[3];

    const unsigned int mv_pivot_index = ml_curr->v; /* The vertex we are "fanning" around! */
    const MVert *mv_pivot = &mverts[mv_pivot_index];
    const MEdge *me_curr = &medges[ml_curr->e];
    const MVert *mv_2 = (me_curr->v1 == mv_pivot_index) ? &mverts[me_curr->v2] :
                                                          &mverts[me_curr->v1];
    const MEdge *me_prev = &medges[ml_prev->e];
    const MVert *mv_3 = (me_prev->v1 == mv_pivot_index) ? &mverts[me_prev->v2] :
                                                          &mverts[me_prev->v1];

    sub_v3_v3v3(vec_curr, mv_2->co, mv_pivot->co);
    normalize_v3(vec_curr);
    sub_v3_v3v3(vec_prev, mv_3->co, mv_pivot->co);
    normalize_v3(vec_prev);

    BKE_lnor_space_define(lnor_space, *lnor, vec_curr, vec_prev, NULL);
    /* We know there is only one loop in this space,
     * no need to create a linklist in this case... */
    BKE_lnor_space_add_loop(lnors_spacearr, lnor_space, ml_curr_index, NULL, true);

    if (clnors_data) {
      BKE_lnor_space_custom_data_to_normal(lnor_space, clnors_data[ml_curr_index], *lnor);
    }
  }
}

static void split_loop_nor_fan_do(LoopSplitTaskDataCommon *common_data, LoopSplitTaskData *data)
{
  MLoopNorSpaceArray *lnors_spacearr = common_data->lnors_spacearr;
  float(*loopnors)[3] = common_data->loopnors;
  short(*clnors_data)[2] = common_data->clnors_data;

  const MVert *mverts = common_data->mverts;
  const MEdge *medges = common_data->medges;
  const MLoop *mloops = common_data->mloops;
  const MPoly *mpolys = common_data->mpolys;
  const int(*edge_to_loops)[2] = common_data->edge_to_loops;
  const int *loop_to_poly = common_data->loop_to_poly;
  const float(*polynors)[3] = common_data->polynors;

  MLoopNorSpace *lnor_space = data->lnor_space;
#if 0 /* Not needed for 'fan' loops. */
  float(*lnor)[3] = data->lnor;
#endif
  const MLoop *ml_curr = data->ml_curr;
  const MLoop *ml_prev = data->ml_prev;
  const int ml_curr_index = data->ml_curr_index;
  const int ml_prev_index = data->ml_prev_index;
  const int mp_index = data->mp_index;
  const int *e2l_prev = data->e2l_prev;

  BLI_Stack *edge_vectors = data->edge_vectors;

  /* Gah... We have to fan around current vertex, until we find the other non-smooth edge,
   * and accumulate face normals into the vertex!
   * Note in case this vertex has only one sharp edges, this is a waste because the normal is the
   * same as the vertex normal, but I do not see any easy way to detect that (would need to count
   * number of sharp edges per vertex, I doubt the additional memory usage would be worth it,
   * especially as it should not be a common case in real-life meshes anyway).
   */
  const unsigned int mv_pivot_index = ml_curr->v; /* The vertex we are "fanning" around! */
  const MVert *mv_pivot = &mverts[mv_pivot_index];

  /* ml_curr would be mlfan_prev if we needed that one. */
  const MEdge *me_org = &medges[ml_curr->e];

  const int *e2lfan_curr;
  float vec_curr[3], vec_prev[3], vec_org[3];
  const MLoop *mlfan_curr;
  float lnor[3] = {0.0f, 0.0f, 0.0f};
  /* mlfan_vert_index: the loop of our current edge might not be the loop of our current vertex! */
  int mlfan_curr_index, mlfan_vert_index, mpfan_curr_index;

  /* We validate clnors data on the fly - cheapest way to do! */
  int clnors_avg[2] = {0, 0};
  short(*clnor_ref)[2] = NULL;
  int clnors_nbr = 0;
  bool clnors_invalid = false;

  /* Temp loop normal stack. */
  BLI_SMALLSTACK_DECLARE(normal, float *);
  /* Temp clnors stack. */
  BLI_SMALLSTACK_DECLARE(clnors, short *);

  e2lfan_curr = e2l_prev;
  mlfan_curr = ml_prev;
  mlfan_curr_index = ml_prev_index;
  mlfan_vert_index = ml_curr_index;
  mpfan_curr_index = mp_index;

  BLI_assert(mlfan_curr_index >= 0);
  BLI_assert(mlfan_vert_index >= 0);
  BLI_assert(mpfan_curr_index >= 0);

  /* Only need to compute previous edge's vector once, then we can just reuse old current one! */
  {
    const MVert *mv_2 = (me_org->v1 == mv_pivot_index) ? &mverts[me_org->v2] : &mverts[me_org->v1];

    sub_v3_v3v3(vec_org, mv_2->co, mv_pivot->co);
    normalize_v3(vec_org);
    copy_v3_v3(vec_prev, vec_org);

    if (lnors_spacearr) {
      BLI_stack_push(edge_vectors, vec_org);
    }
  }

  //  printf("FAN: vert %d, start edge %d\n", mv_pivot_index, ml_curr->e);

  while (true) {
    const MEdge *me_curr = &medges[mlfan_curr->e];
    /* Compute edge vectors.
     * NOTE: We could pre-compute those into an array, in the first iteration, instead of computing
     *       them twice (or more) here. However, time gained is not worth memory and time lost,
     *       given the fact that this code should not be called that much in real-life meshes...
     */
    {
      const MVert *mv_2 = (me_curr->v1 == mv_pivot_index) ? &mverts[me_curr->v2] :
                                                            &mverts[me_curr->v1];

      sub_v3_v3v3(vec_curr, mv_2->co, mv_pivot->co);
      normalize_v3(vec_curr);
    }

    //      printf("\thandling edge %d / loop %d\n", mlfan_curr->e, mlfan_curr_index);

    {
      /* Code similar to accumulate_vertex_normals_poly_v3. */
      /* Calculate angle between the two poly edges incident on this vertex. */
      const float fac = saacos(dot_v3v3(vec_curr, vec_prev));
      /* Accumulate */
      madd_v3_v3fl(lnor, polynors[mpfan_curr_index], fac);

      if (clnors_data) {
        /* Accumulate all clnors, if they are not all equal we have to fix that! */
        short(*clnor)[2] = &clnors_data[mlfan_vert_index];
        if (clnors_nbr) {
          clnors_invalid |= ((*clnor_ref)[0] != (*clnor)[0] || (*clnor_ref)[1] != (*clnor)[1]);
        }
        else {
          clnor_ref = clnor;
        }
        clnors_avg[0] += (*clnor)[0];
        clnors_avg[1] += (*clnor)[1];
        clnors_nbr++;
        /* We store here a pointer to all custom lnors processed. */
        BLI_SMALLSTACK_PUSH(clnors, (short *)*clnor);
      }
    }

    /* We store here a pointer to all loop-normals processed. */
    BLI_SMALLSTACK_PUSH(normal, (float *)(loopnors[mlfan_vert_index]));

    if (lnors_spacearr) {
      /* Assign current lnor space to current 'vertex' loop. */
      BKE_lnor_space_add_loop(lnors_spacearr, lnor_space, mlfan_vert_index, NULL, false);
      if (me_curr != me_org) {
        /* We store here all edges-normalized vectors processed. */
        BLI_stack_push(edge_vectors, vec_curr);
      }
    }

    if (IS_EDGE_SHARP(e2lfan_curr) || (me_curr == me_org)) {
      /* Current edge is sharp and we have finished with this fan of faces around this vert,
       * or this vert is smooth, and we have completed a full turn around it.
       */
      //          printf("FAN: Finished!\n");
      break;
    }

    copy_v3_v3(vec_prev, vec_curr);

    /* Find next loop of the smooth fan. */
    BKE_mesh_loop_manifold_fan_around_vert_next(mloops,
                                                mpolys,
                                                loop_to_poly,
                                                e2lfan_curr,
                                                mv_pivot_index,
                                                &mlfan_curr,
                                                &mlfan_curr_index,
                                                &mlfan_vert_index,
                                                &mpfan_curr_index);

    e2lfan_curr = edge_to_loops[mlfan_curr->e];
  }

  {
    float lnor_len = normalize_v3(lnor);

    /* If we are generating lnor spacearr, we can now define the one for this fan,
     * and optionally compute final lnor from custom data too!
     */
    if (lnors_spacearr) {
      if (UNLIKELY(lnor_len == 0.0f)) {
        /* Use vertex normal as fallback! */
        copy_v3_v3(lnor, loopnors[mlfan_vert_index]);
        lnor_len = 1.0f;
      }

      BKE_lnor_space_define(lnor_space, lnor, vec_org, vec_curr, edge_vectors);

      if (clnors_data) {
        if (clnors_invalid) {
          short *clnor;

          clnors_avg[0] /= clnors_nbr;
          clnors_avg[1] /= clnors_nbr;
          /* Fix/update all clnors of this fan with computed average value. */
          if (G.debug & G_DEBUG) {
            printf("Invalid clnors in this fan!\n");
          }
          while ((clnor = BLI_SMALLSTACK_POP(clnors))) {
            // print_v2("org clnor", clnor);
            clnor[0] = (short)clnors_avg[0];
            clnor[1] = (short)clnors_avg[1];
          }
          // print_v2("new clnors", clnors_avg);
        }
        /* Extra bonus: since small-stack is local to this function,
         * no more need to empty it at all cost! */

        BKE_lnor_space_custom_data_to_normal(lnor_space, *clnor_ref, lnor);
      }
    }

    /* In case we get a zero normal here, just use vertex normal already set! */
    if (LIKELY(lnor_len != 0.0f)) {
      /* Copy back the final computed normal into all related loop-normals. */
      float *nor;

      while ((nor = BLI_SMALLSTACK_POP(normal))) {
        copy_v3_v3(nor, lnor);
      }
    }
    /* Extra bonus: since small-stack is local to this function,
     * no more need to empty it at all cost! */
  }
}

static void loop_split_worker_do(LoopSplitTaskDataCommon *common_data,
                                 LoopSplitTaskData *data,
                                 BLI_Stack *edge_vectors)
{
  BLI_assert(data->ml_curr);
  if (data->e2l_prev) {
    BLI_assert((edge_vectors == NULL) || BLI_stack_is_empty(edge_vectors));
    data->edge_vectors = edge_vectors;
    split_loop_nor_fan_do(common_data, data);
  }
  else {
    /* No need for edge_vectors for 'single' case! */
    split_loop_nor_single_do(common_data, data);
  }
}

static void loop_split_worker(TaskPool *__restrict pool, void *taskdata)
{
  LoopSplitTaskDataCommon *common_data = BLI_task_pool_user_data(pool);
  LoopSplitTaskData *data = taskdata;

  /* Temp edge vectors stack, only used when computing lnor spacearr. */
  BLI_Stack *edge_vectors = common_data->lnors_spacearr ?
                                BLI_stack_new(sizeof(float[3]), __func__) :
                                NULL;

#ifdef DEBUG_TIME
  TIMEIT_START_AVERAGED(loop_split_worker);
#endif

  for (int i = 0; i < LOOP_SPLIT_TASK_BLOCK_SIZE; i++, data++) {
    /* A NULL ml_curr is used to tag ended data! */
    if (data->ml_curr == NULL) {
      break;
    }

    loop_split_worker_do(common_data, data, edge_vectors);
  }

  if (edge_vectors) {
    BLI_stack_free(edge_vectors);
  }

#ifdef DEBUG_TIME
  TIMEIT_END_AVERAGED(loop_split_worker);
#endif
}

/**
 * Check whether given loop is part of an unknown-so-far cyclic smooth fan, or not.
 * Needed because cyclic smooth fans have no obvious 'entry point',
 * and yet we need to walk them once, and only once.
 */
static bool loop_split_generator_check_cyclic_smooth_fan(const MLoop *mloops,
                                                         const MPoly *mpolys,
                                                         const int (*edge_to_loops)[2],
                                                         const int *loop_to_poly,
                                                         const int *e2l_prev,
                                                         BLI_bitmap *skip_loops,
                                                         const MLoop *ml_curr,
                                                         const MLoop *ml_prev,
                                                         const int ml_curr_index,
                                                         const int ml_prev_index,
                                                         const int mp_curr_index)
{
  const unsigned int mv_pivot_index = ml_curr->v; /* The vertex we are "fanning" around! */
  const int *e2lfan_curr;
  const MLoop *mlfan_curr;
  /* mlfan_vert_index: the loop of our current edge might not be the loop of our current vertex! */
  int mlfan_curr_index, mlfan_vert_index, mpfan_curr_index;

  e2lfan_curr = e2l_prev;
  if (IS_EDGE_SHARP(e2lfan_curr)) {
    /* Sharp loop, so not a cyclic smooth fan... */
    return false;
  }

  mlfan_curr = ml_prev;
  mlfan_curr_index = ml_prev_index;
  mlfan_vert_index = ml_curr_index;
  mpfan_curr_index = mp_curr_index;

  BLI_assert(mlfan_curr_index >= 0);
  BLI_assert(mlfan_vert_index >= 0);
  BLI_assert(mpfan_curr_index >= 0);

  BLI_assert(!BLI_BITMAP_TEST(skip_loops, mlfan_vert_index));
  BLI_BITMAP_ENABLE(skip_loops, mlfan_vert_index);

  while (true) {
    /* Find next loop of the smooth fan. */
    BKE_mesh_loop_manifold_fan_around_vert_next(mloops,
                                                mpolys,
                                                loop_to_poly,
                                                e2lfan_curr,
                                                mv_pivot_index,
                                                &mlfan_curr,
                                                &mlfan_curr_index,
                                                &mlfan_vert_index,
                                                &mpfan_curr_index);

    e2lfan_curr = edge_to_loops[mlfan_curr->e];

    if (IS_EDGE_SHARP(e2lfan_curr)) {
      /* Sharp loop/edge, so not a cyclic smooth fan... */
      return false;
    }
    /* Smooth loop/edge... */
    if (BLI_BITMAP_TEST(skip_loops, mlfan_vert_index)) {
      if (mlfan_vert_index == ml_curr_index) {
        /* We walked around a whole cyclic smooth fan without finding any already-processed loop,
         * means we can use initial ml_curr/ml_prev edge as start for this smooth fan. */
        return true;
      }
      /* ... already checked in some previous looping, we can abort. */
      return false;
    }

    /* ... we can skip it in future, and keep checking the smooth fan. */
    BLI_BITMAP_ENABLE(skip_loops, mlfan_vert_index);
  }
}

static void loop_split_generator(TaskPool *pool, LoopSplitTaskDataCommon *common_data)
{
  MLoopNorSpaceArray *lnors_spacearr = common_data->lnors_spacearr;
  float(*loopnors)[3] = common_data->loopnors;

  const MLoop *mloops = common_data->mloops;
  const MPoly *mpolys = common_data->mpolys;
  const int *loop_to_poly = common_data->loop_to_poly;
  const int(*edge_to_loops)[2] = common_data->edge_to_loops;
  const int numLoops = common_data->numLoops;
  const int numPolys = common_data->numPolys;

  const MPoly *mp;
  int mp_index;

  const MLoop *ml_curr;
  const MLoop *ml_prev;
  int ml_curr_index;
  int ml_prev_index;

  BLI_bitmap *skip_loops = BLI_BITMAP_NEW(numLoops, __func__);

  LoopSplitTaskData *data_buff = NULL;
  int data_idx = 0;

  /* Temp edge vectors stack, only used when computing lnor spacearr
   * (and we are not multi-threading). */
  BLI_Stack *edge_vectors = NULL;

#ifdef DEBUG_TIME
  TIMEIT_START_AVERAGED(loop_split_generator);
#endif

  if (!pool) {
    if (lnors_spacearr) {
      edge_vectors = BLI_stack_new(sizeof(float[3]), __func__);
    }
  }

  /* We now know edges that can be smoothed (with their vector, and their two loops),
   * and edges that will be hard! Now, time to generate the normals.
   */
  for (mp = mpolys, mp_index = 0; mp_index < numPolys; mp++, mp_index++) {
    float(*lnors)[3];
    const int ml_last_index = (mp->loopstart + mp->totloop) - 1;
    ml_curr_index = mp->loopstart;
    ml_prev_index = ml_last_index;

    ml_curr = &mloops[ml_curr_index];
    ml_prev = &mloops[ml_prev_index];
    lnors = &loopnors[ml_curr_index];

    for (; ml_curr_index <= ml_last_index; ml_curr++, ml_curr_index++, lnors++) {
      const int *e2l_curr = edge_to_loops[ml_curr->e];
      const int *e2l_prev = edge_to_loops[ml_prev->e];

#if 0
      printf("Checking loop %d / edge %u / vert %u (sharp edge: %d, skiploop: %d)...",
             ml_curr_index,
             ml_curr->e,
             ml_curr->v,
             IS_EDGE_SHARP(e2l_curr),
             BLI_BITMAP_TEST_BOOL(skip_loops, ml_curr_index));
#endif

      /* A smooth edge, we have to check for cyclic smooth fan case.
       * If we find a new, never-processed cyclic smooth fan, we can do it now using that loop/edge
       * as 'entry point', otherwise we can skip it. */

      /* Note: In theory, we could make loop_split_generator_check_cyclic_smooth_fan() store
       * mlfan_vert_index'es and edge indexes in two stacks, to avoid having to fan again around
       * the vert during actual computation of clnor & clnorspace. However, this would complicate
       * the code, add more memory usage, and despite its logical complexity,
       * loop_manifold_fan_around_vert_next() is quite cheap in term of CPU cycles,
       * so really think it's not worth it. */
      if (!IS_EDGE_SHARP(e2l_curr) && (BLI_BITMAP_TEST(skip_loops, ml_curr_index) ||
                                       !loop_split_generator_check_cyclic_smooth_fan(mloops,
                                                                                     mpolys,
                                                                                     edge_to_loops,
                                                                                     loop_to_poly,
                                                                                     e2l_prev,
                                                                                     skip_loops,
                                                                                     ml_curr,
                                                                                     ml_prev,
                                                                                     ml_curr_index,
                                                                                     ml_prev_index,
                                                                                     mp_index))) {
        //              printf("SKIPPING!\n");
      }
      else {
        LoopSplitTaskData *data, data_local;

        //              printf("PROCESSING!\n");

        if (pool) {
          if (data_idx == 0) {
            data_buff = MEM_calloc_arrayN(
                LOOP_SPLIT_TASK_BLOCK_SIZE, sizeof(*data_buff), __func__);
          }
          data = &data_buff[data_idx];
        }
        else {
          data = &data_local;
          memset(data, 0, sizeof(*data));
        }

        if (IS_EDGE_SHARP(e2l_curr) && IS_EDGE_SHARP(e2l_prev)) {
          data->lnor = lnors;
          data->ml_curr = ml_curr;
          data->ml_prev = ml_prev;
          data->ml_curr_index = ml_curr_index;
#if 0 /* Not needed for 'single' loop. */
          data->ml_prev_index = ml_prev_index;
          data->e2l_prev = NULL; /* Tag as 'single' task. */
#endif
          data->mp_index = mp_index;
          if (lnors_spacearr) {
            data->lnor_space = BKE_lnor_space_create(lnors_spacearr);
          }
        }
        /* We *do not need* to check/tag loops as already computed!
         * Due to the fact a loop only links to one of its two edges,
         * a same fan *will never be walked more than once!*
         * Since we consider edges having neighbor polys with inverted
         * (flipped) normals as sharp, we are sure that no fan will be skipped,
         * even only considering the case (sharp curr_edge, smooth prev_edge),
         * and not the alternative (smooth curr_edge, sharp prev_edge).
         * All this due/thanks to link between normals and loop ordering (i.e. winding).
         */
        else {
#if 0 /* Not needed for 'fan' loops. */
          data->lnor = lnors;
#endif
          data->ml_curr = ml_curr;
          data->ml_prev = ml_prev;
          data->ml_curr_index = ml_curr_index;
          data->ml_prev_index = ml_prev_index;
          data->e2l_prev = e2l_prev; /* Also tag as 'fan' task. */
          data->mp_index = mp_index;
          if (lnors_spacearr) {
            data->lnor_space = BKE_lnor_space_create(lnors_spacearr);
          }
        }

        if (pool) {
          data_idx++;
          if (data_idx == LOOP_SPLIT_TASK_BLOCK_SIZE) {
            BLI_task_pool_push(pool, loop_split_worker, data_buff, true, NULL);
            data_idx = 0;
          }
        }
        else {
          loop_split_worker_do(common_data, data, edge_vectors);
        }
      }

      ml_prev = ml_curr;
      ml_prev_index = ml_curr_index;
    }
  }

  /* Last block of data... Since it is calloc'ed and we use first NULL item as stopper,
   * everything is fine. */
  if (pool && data_idx) {
    BLI_task_pool_push(pool, loop_split_worker, data_buff, true, NULL);
  }

  if (edge_vectors) {
    BLI_stack_free(edge_vectors);
  }
  MEM_freeN(skip_loops);

#ifdef DEBUG_TIME
  TIMEIT_END_AVERAGED(loop_split_generator);
#endif
}

#undef INDEX_UNSET
#undef INDEX_INVALID
#undef IS_EDGE_SHARP

void BKE_mesh_normals_loop_custom_set(const MVert *mverts,
                                      const int numVerts,
                                      MEdge *medges,
                                      const int numEdges,
                                      MLoop *mloops,
                                      float (*r_custom_loopnors)[3],
                                      const int numLoops,
                                      MPoly *mpolys,
                                      const float (*polynors)[3],
                                      const int numPolys,
                                      short (*r_clnors_data)[2])
{
  mesh_normals_loop_custom_set(mverts,
                               numVerts,
                               medges,
                               numEdges,
                               mloops,
                               r_custom_loopnors,
                               numLoops,
                               mpolys,
                               polynors,
                               numPolys,
                               r_clnors_data,
                               false);
}

void BKE_mesh_normals_loop_custom_from_vertices_set(const MVert *mverts,
                                                    float (*r_custom_vertnors)[3],
                                                    const int numVerts,
                                                    MEdge *medges,
                                                    const int numEdges,
                                                    MLoop *mloops,
                                                    const int numLoops,
                                                    MPoly *mpolys,
                                                    const float (*polynors)[3],
                                                    const int numPolys,
                                                    short (*r_clnors_data)[2])
{
  mesh_normals_loop_custom_set(mverts,
                               numVerts,
                               medges,
                               numEdges,
                               mloops,
                               r_custom_vertnors,
                               numLoops,
                               mpolys,
                               polynors,
                               numPolys,
                               r_clnors_data,
                               true);
}

#undef LNOR_SPACE_TRIGO_THRESHOLD

/** \} */

/* -------------------------------------------------------------------- */
/** \name Polygon Calculations
 * \{ */

/*
 * COMPUTE POLY NORMAL
 *
 * Computes the normal of a planar
 * polygon See Graphics Gems for
 * computing newell normal.
 */
static void mesh_calc_ngon_normal(const MPoly *mpoly,
                                  const MLoop *loopstart,
                                  const MVert *mvert,
                                  float normal[3])
{
  const int nverts = mpoly->totloop;
  const float *v_prev = mvert[loopstart[nverts - 1].v].co;
  const float *v_curr;

  zero_v3(normal);

  /* Newell's Method */
  for (int i = 0; i < nverts; i++) {
    v_curr = mvert[loopstart[i].v].co;
    add_newell_cross_v3_v3v3(normal, v_prev, v_curr);
    v_prev = v_curr;
  }

  if (UNLIKELY(normalize_v3(normal) == 0.0f)) {
    normal[2] = 1.0f; /* other axis set to 0.0 */
  }
}

void BKE_mesh_calc_poly_normal(const MPoly *mpoly,
                               const MLoop *loopstart,
                               const MVert *mvarray,
                               float r_no[3])
{
  if (mpoly->totloop > 4) {
    mesh_calc_ngon_normal(mpoly, loopstart, mvarray, r_no);
  }
  else if (mpoly->totloop == 3) {
    normal_tri_v3(
        r_no, mvarray[loopstart[0].v].co, mvarray[loopstart[1].v].co, mvarray[loopstart[2].v].co);
  }
  else if (mpoly->totloop == 4) {
    normal_quad_v3(r_no,
                   mvarray[loopstart[0].v].co,
                   mvarray[loopstart[1].v].co,
                   mvarray[loopstart[2].v].co,
                   mvarray[loopstart[3].v].co);
  }
  else { /* horrible, two sided face! */
    r_no[0] = 0.0;
    r_no[1] = 0.0;
    r_no[2] = 1.0;
  }
}
/* duplicate of function above _but_ takes coords rather than mverts */
static void mesh_calc_ngon_normal_coords(const MPoly *mpoly,
                                         const MLoop *loopstart,
                                         const float (*vertex_coords)[3],
                                         float r_normal[3])
{
  const int nverts = mpoly->totloop;
  const float *v_prev = vertex_coords[loopstart[nverts - 1].v];
  const float *v_curr;

  zero_v3(r_normal);

  /* Newell's Method */
  for (int i = 0; i < nverts; i++) {
    v_curr = vertex_coords[loopstart[i].v];
    add_newell_cross_v3_v3v3(r_normal, v_prev, v_curr);
    v_prev = v_curr;
  }

  if (UNLIKELY(normalize_v3(r_normal) == 0.0f)) {
    r_normal[2] = 1.0f; /* other axis set to 0.0 */
  }
}

void BKE_mesh_calc_poly_normal_coords(const MPoly *mpoly,
                                      const MLoop *loopstart,
                                      const float (*vertex_coords)[3],
                                      float r_no[3])
{
  if (mpoly->totloop > 4) {
    mesh_calc_ngon_normal_coords(mpoly, loopstart, vertex_coords, r_no);
  }
  else if (mpoly->totloop == 3) {
    normal_tri_v3(r_no,
                  vertex_coords[loopstart[0].v],
                  vertex_coords[loopstart[1].v],
                  vertex_coords[loopstart[2].v]);
  }
  else if (mpoly->totloop == 4) {
    normal_quad_v3(r_no,
                   vertex_coords[loopstart[0].v],
                   vertex_coords[loopstart[1].v],
                   vertex_coords[loopstart[2].v],
                   vertex_coords[loopstart[3].v]);
  }
  else { /* horrible, two sided face! */
    r_no[0] = 0.0;
    r_no[1] = 0.0;
    r_no[2] = 1.0;
  }
}

static void mesh_calc_ngon_center(const MPoly *mpoly,
                                  const MLoop *loopstart,
                                  const MVert *mvert,
                                  float cent[3])
{
  const float w = 1.0f / (float)mpoly->totloop;

  zero_v3(cent);

  for (int i = 0; i < mpoly->totloop; i++) {
    madd_v3_v3fl(cent, mvert[(loopstart++)->v].co, w);
  }
}

void BKE_mesh_calc_poly_center(const MPoly *mpoly,
                               const MLoop *loopstart,
                               const MVert *mvarray,
                               float r_cent[3])
{
  if (mpoly->totloop == 3) {
    mid_v3_v3v3v3(r_cent,
                  mvarray[loopstart[0].v].co,
                  mvarray[loopstart[1].v].co,
                  mvarray[loopstart[2].v].co);
  }
  else if (mpoly->totloop == 4) {
    mid_v3_v3v3v3v3(r_cent,
                    mvarray[loopstart[0].v].co,
                    mvarray[loopstart[1].v].co,
                    mvarray[loopstart[2].v].co,
                    mvarray[loopstart[3].v].co);
  }
  else {
    mesh_calc_ngon_center(mpoly, loopstart, mvarray, r_cent);
  }
}

static float UNUSED_FUNCTION(mesh_calc_poly_volume_centroid)(const MPoly *mpoly,
                                                             const MLoop *loopstart,
                                                             const MVert *mvarray,
                                                             float r_cent[3])
{
  const float *v_pivot, *v_step1;
  float total_volume = 0.0f;

  zero_v3(r_cent);

  v_pivot = mvarray[loopstart[0].v].co;
  v_step1 = mvarray[loopstart[1].v].co;

  for (int i = 2; i < mpoly->totloop; i++) {
    const float *v_step2 = mvarray[loopstart[i].v].co;

    /* Calculate the 6x volume of the tetrahedron formed by the 3 vertices
     * of the triangle and the origin as the fourth vertex */
    const float tetra_volume = volume_tri_tetrahedron_signed_v3_6x(v_pivot, v_step1, v_step2);
    total_volume += tetra_volume;

    /* Calculate the centroid of the tetrahedron formed by the 3 vertices
     * of the triangle and the origin as the fourth vertex.
     * The centroid is simply the average of the 4 vertices.
     *
     * Note that the vector is 4x the actual centroid
     * so the division can be done once at the end. */
    for (uint j = 0; j < 3; j++) {
      r_cent[j] += tetra_volume * (v_pivot[j] + v_step1[j] + v_step2[j]);
    }

    v_step1 = v_step2;
  }

  return total_volume;
}

/**
 * A version of mesh_calc_poly_volume_centroid that takes an initial reference center,
 * use this to increase numeric stability as the quality of the result becomes
 * very low quality as the value moves away from 0.0, see: T65986.
 */
static float mesh_calc_poly_volume_centroid_with_reference_center(const MPoly *mpoly,
                                                                  const MLoop *loopstart,
                                                                  const MVert *mvarray,
                                                                  const float reference_center[3],
                                                                  float r_cent[3])
{
  /* See: mesh_calc_poly_volume_centroid for comments. */
  float v_pivot[3], v_step1[3];
  float total_volume = 0.0f;
  zero_v3(r_cent);
  sub_v3_v3v3(v_pivot, mvarray[loopstart[0].v].co, reference_center);
  sub_v3_v3v3(v_step1, mvarray[loopstart[1].v].co, reference_center);
  for (int i = 2; i < mpoly->totloop; i++) {
    float v_step2[3];
    sub_v3_v3v3(v_step2, mvarray[loopstart[i].v].co, reference_center);
    const float tetra_volume = volume_tri_tetrahedron_signed_v3_6x(v_pivot, v_step1, v_step2);
    total_volume += tetra_volume;
    for (uint j = 0; j < 3; j++) {
      r_cent[j] += tetra_volume * (v_pivot[j] + v_step1[j] + v_step2[j]);
    }
    copy_v3_v3(v_step1, v_step2);
  }
  return total_volume;
}

/**
 * \note
 * - Results won't be correct if polygon is non-planar.
 * - This has the advantage over #mesh_calc_poly_volume_centroid
 *   that it doesn't depend on solid geometry, instead it weights the surface by volume.
 */
static float mesh_calc_poly_area_centroid(const MPoly *mpoly,
                                          const MLoop *loopstart,
                                          const MVert *mvarray,
                                          float r_cent[3])
{
  float total_area = 0.0f;
  float v1[3], v2[3], v3[3], normal[3], tri_cent[3];

  BKE_mesh_calc_poly_normal(mpoly, loopstart, mvarray, normal);
  copy_v3_v3(v1, mvarray[loopstart[0].v].co);
  copy_v3_v3(v2, mvarray[loopstart[1].v].co);
  zero_v3(r_cent);

  for (int i = 2; i < mpoly->totloop; i++) {
    copy_v3_v3(v3, mvarray[loopstart[i].v].co);

    float tri_area = area_tri_signed_v3(v1, v2, v3, normal);
    total_area += tri_area;

    mid_v3_v3v3v3(tri_cent, v1, v2, v3);
    madd_v3_v3fl(r_cent, tri_cent, tri_area);

    copy_v3_v3(v2, v3);
  }

  mul_v3_fl(r_cent, 1.0f / total_area);

  return total_area;
}

void BKE_mesh_calc_poly_angles(const MPoly *mpoly,
                               const MLoop *loopstart,
                               const MVert *mvarray,
                               float angles[])
{
  float nor_prev[3];
  float nor_next[3];

  int i_this = mpoly->totloop - 1;
  int i_next = 0;

  sub_v3_v3v3(nor_prev, mvarray[loopstart[i_this - 1].v].co, mvarray[loopstart[i_this].v].co);
  normalize_v3(nor_prev);

  while (i_next < mpoly->totloop) {
    sub_v3_v3v3(nor_next, mvarray[loopstart[i_this].v].co, mvarray[loopstart[i_next].v].co);
    normalize_v3(nor_next);
    angles[i_this] = angle_normalized_v3v3(nor_prev, nor_next);

    /* step */
    copy_v3_v3(nor_prev, nor_next);
    i_this = i_next;
    i_next++;
  }
}

void BKE_mesh_poly_edgehash_insert(EdgeHash *ehash, const MPoly *mp, const MLoop *mloop)
{
  const MLoop *ml, *ml_next;
  int i = mp->totloop;

  ml_next = mloop;      /* first loop */
  ml = &ml_next[i - 1]; /* last loop */

  while (i-- != 0) {
    BLI_edgehash_reinsert(ehash, ml->v, ml_next->v, NULL);

    ml = ml_next;
    ml_next++;
  }
}

void BKE_mesh_poly_edgebitmap_insert(unsigned int *edge_bitmap,
                                     const MPoly *mp,
                                     const MLoop *mloop)
{
  const MLoop *ml;
  int i = mp->totloop;

  ml = mloop;

  while (i-- != 0) {
    BLI_BITMAP_ENABLE(edge_bitmap, ml->e);
    ml++;
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Mesh Center Calculation
 * \{ */

bool BKE_mesh_center_median(const Mesh *me, float r_cent[3])
{
  int i = me->totvert;
  const MVert *mvert;
  zero_v3(r_cent);
  for (mvert = me->mvert; i--; mvert++) {
    add_v3_v3(r_cent, mvert->co);
  }
  /* otherwise we get NAN for 0 verts */
  if (me->totvert) {
    mul_v3_fl(r_cent, 1.0f / (float)me->totvert);
  }
  return (me->totvert != 0);
}

/**
 * Calculate the center from polygons,
 * use when we want to ignore vertex locations that don't have connected faces.
 */
bool BKE_mesh_center_median_from_polys(const Mesh *me, float r_cent[3])
{
  int i = me->totpoly;
  int tot = 0;
  const MPoly *mpoly = me->mpoly;
  const MLoop *mloop = me->mloop;
  const MVert *mvert = me->mvert;
  zero_v3(r_cent);
  for (mpoly = me->mpoly; i--; mpoly++) {
    int loopend = mpoly->loopstart + mpoly->totloop;
    for (int j = mpoly->loopstart; j < loopend; j++) {
      add_v3_v3(r_cent, mvert[mloop[j].v].co);
    }
    tot += mpoly->totloop;
  }
  /* otherwise we get NAN for 0 verts */
  if (me->totpoly) {
    mul_v3_fl(r_cent, 1.0f / (float)tot);
  }
  return (me->totpoly != 0);
}

bool BKE_mesh_center_bounds(const Mesh *me, float r_cent[3])
{
  float min[3], max[3];
  INIT_MINMAX(min, max);
  if (BKE_mesh_minmax(me, min, max)) {
    mid_v3_v3v3(r_cent, min, max);
    return true;
  }

  return false;
}

bool BKE_mesh_center_of_surface(const Mesh *me, float r_cent[3])
{
  int i = me->totpoly;
  MPoly *mpoly;
  float poly_area;
  float total_area = 0.0f;
  float poly_cent[3];

  zero_v3(r_cent);

  /* calculate a weighted average of polygon centroids */
  for (mpoly = me->mpoly; i--; mpoly++) {
    poly_area = mesh_calc_poly_area_centroid(
        mpoly, me->mloop + mpoly->loopstart, me->mvert, poly_cent);

    madd_v3_v3fl(r_cent, poly_cent, poly_area);
    total_area += poly_area;
  }
  /* otherwise we get NAN for 0 polys */
  if (me->totpoly) {
    mul_v3_fl(r_cent, 1.0f / total_area);
  }

  /* zero area faces cause this, fallback to median */
  if (UNLIKELY(!is_finite_v3(r_cent))) {
    return BKE_mesh_center_median(me, r_cent);
  }

  return (me->totpoly != 0);
}

/**
 * \note Mesh must be manifold with consistent face-winding,
 * see #mesh_calc_poly_volume_centroid for details.
 */
bool BKE_mesh_center_of_volume(const Mesh *me, float r_cent[3])
{
  int i = me->totpoly;
  MPoly *mpoly;
  float poly_volume;
  float total_volume = 0.0f;
  float poly_cent[3];

  /* Use an initial center to avoid numeric instability of geometry far away from the center. */
  float init_cent[3];
  const bool init_cent_result = BKE_mesh_center_median_from_polys(me, init_cent);

  zero_v3(r_cent);

  /* calculate a weighted average of polyhedron centroids */
  for (mpoly = me->mpoly; i--; mpoly++) {
    poly_volume = mesh_calc_poly_volume_centroid_with_reference_center(
        mpoly, me->mloop + mpoly->loopstart, me->mvert, init_cent, poly_cent);

    /* poly_cent is already volume-weighted, so no need to multiply by the volume */
    add_v3_v3(r_cent, poly_cent);
    total_volume += poly_volume;
  }
  /* otherwise we get NAN for 0 polys */
  if (total_volume != 0.0f) {
    /* multiply by 0.25 to get the correct centroid */
    /* no need to divide volume by 6 as the centroid is weighted by 6x the volume,
     * so it all cancels out. */
    mul_v3_fl(r_cent, 0.25f / total_volume);
  }

  /* this can happen for non-manifold objects, fallback to median */
  if (UNLIKELY(!is_finite_v3(r_cent))) {
    copy_v3_v3(r_cent, init_cent);
    return init_cent_result;
  }
  add_v3_v3(r_cent, init_cent);
  return (me->totpoly != 0);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Mesh Volume Calculation
 * \{ */

static bool mesh_calc_center_centroid_ex(const MVert *mverts,
                                         int UNUSED(mverts_num),
                                         const MLoopTri *looptri,
                                         int looptri_num,
                                         const MLoop *mloop,
                                         float r_center[3])
{

  zero_v3(r_center);

  if (looptri_num == 0) {
    return false;
  }

  float totweight = 0.0f;
  const MLoopTri *lt;
  int i;
  for (i = 0, lt = looptri; i < looptri_num; i++, lt++) {
    const MVert *v1 = &mverts[mloop[lt->tri[0]].v];
    const MVert *v2 = &mverts[mloop[lt->tri[1]].v];
    const MVert *v3 = &mverts[mloop[lt->tri[2]].v];
    float area;

    area = area_tri_v3(v1->co, v2->co, v3->co);
    madd_v3_v3fl(r_center, v1->co, area);
    madd_v3_v3fl(r_center, v2->co, area);
    madd_v3_v3fl(r_center, v3->co, area);
    totweight += area;
  }
  if (totweight == 0.0f) {
    return false;
  }

  mul_v3_fl(r_center, 1.0f / (3.0f * totweight));

  return true;
}

/**
 * Calculate the volume and center.
 *
 * \param r_volume: Volume (unsigned).
 * \param r_center: Center of mass.
 */
void BKE_mesh_calc_volume(const MVert *mverts,
                          const int mverts_num,
                          const MLoopTri *looptri,
                          const int looptri_num,
                          const MLoop *mloop,
                          float *r_volume,
                          float r_center[3])
{
  const MLoopTri *lt;
  float center[3];
  float totvol;
  int i;

  if (r_volume) {
    *r_volume = 0.0f;
  }
  if (r_center) {
    zero_v3(r_center);
  }

  if (looptri_num == 0) {
    return;
  }

  if (!mesh_calc_center_centroid_ex(mverts, mverts_num, looptri, looptri_num, mloop, center)) {
    return;
  }

  totvol = 0.0f;

  for (i = 0, lt = looptri; i < looptri_num; i++, lt++) {
    const MVert *v1 = &mverts[mloop[lt->tri[0]].v];
    const MVert *v2 = &mverts[mloop[lt->tri[1]].v];
    const MVert *v3 = &mverts[mloop[lt->tri[2]].v];
    float vol;

    vol = volume_tetrahedron_signed_v3(center, v1->co, v2->co, v3->co);
    if (r_volume) {
      totvol += vol;
    }
    if (r_center) {
      /* averaging factor 1/3 is applied in the end */
      madd_v3_v3fl(r_center, v1->co, vol);
      madd_v3_v3fl(r_center, v2->co, vol);
      madd_v3_v3fl(r_center, v3->co, vol);
    }
  }

  /* Note: Depending on arbitrary centroid position,
   * totvol can become negative even for a valid mesh.
   * The true value is always the positive value.
   */
  if (r_volume) {
    *r_volume = fabsf(totvol);
  }
  if (r_center) {
    /* Note: Factor 1/3 is applied once for all vertices here.
     * This also automatically negates the vector if totvol is negative.
     */
    if (totvol != 0.0f) {
      mul_v3_fl(r_center, (1.0f / 3.0f) / totvol);
    }
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name NGon Tessellation (NGon/Tessface Conversion)
 * \{ */

/**
 * Convert a triangle or quadrangle of loop/poly data to tessface data
 */
void BKE_mesh_loops_to_mface_corners(
    CustomData *fdata,
    CustomData *ldata,
    CustomData *UNUSED(pdata),
    unsigned int lindex[4],
    int findex,
    const int UNUSED(polyindex),
    const int mf_len, /* 3 or 4 */

    /* cache values to avoid lookups every time */
    const int numUV,         /* CustomData_number_of_layers(ldata, CD_MLOOPUV) */
    const int numCol,        /* CustomData_number_of_layers(ldata, CD_MLOOPCOL) */
    const bool hasPCol,      /* CustomData_has_layer(ldata, CD_PREVIEW_MLOOPCOL) */
    const bool hasOrigSpace, /* CustomData_has_layer(ldata, CD_ORIGSPACE_MLOOP) */
    const bool hasLNor       /* CustomData_has_layer(ldata, CD_NORMAL) */
)
{
  MTFace *texface;
  MCol *mcol;
  MLoopCol *mloopcol;
  MLoopUV *mloopuv;
  int i, j;

  for (i = 0; i < numUV; i++) {
    texface = CustomData_get_n(fdata, CD_MTFACE, findex, i);

    for (j = 0; j < mf_len; j++) {
      mloopuv = CustomData_get_n(ldata, CD_MLOOPUV, (int)lindex[j], i);
      copy_v2_v2(texface->uv[j], mloopuv->uv);
    }
  }

  for (i = 0; i < numCol; i++) {
    mcol = CustomData_get_n(fdata, CD_MCOL, findex, i);

    for (j = 0; j < mf_len; j++) {
      mloopcol = CustomData_get_n(ldata, CD_MLOOPCOL, (int)lindex[j], i);
      MESH_MLOOPCOL_TO_MCOL(mloopcol, &mcol[j]);
    }
  }

  if (hasPCol) {
    mcol = CustomData_get(fdata, findex, CD_PREVIEW_MCOL);

    for (j = 0; j < mf_len; j++) {
      mloopcol = CustomData_get(ldata, (int)lindex[j], CD_PREVIEW_MLOOPCOL);
      MESH_MLOOPCOL_TO_MCOL(mloopcol, &mcol[j]);
    }
  }

  if (hasOrigSpace) {
    OrigSpaceFace *of = CustomData_get(fdata, findex, CD_ORIGSPACE);
    OrigSpaceLoop *lof;

    for (j = 0; j < mf_len; j++) {
      lof = CustomData_get(ldata, (int)lindex[j], CD_ORIGSPACE_MLOOP);
      copy_v2_v2(of->uv[j], lof->uv);
    }
  }

  if (hasLNor) {
    short(*tlnors)[3] = CustomData_get(fdata, findex, CD_TESSLOOPNORMAL);

    for (j = 0; j < mf_len; j++) {
      normal_float_to_short_v3(tlnors[j], CustomData_get(ldata, (int)lindex[j], CD_NORMAL));
    }
  }
}

/**
 * Convert all CD layers from loop/poly to tessface data.
 *
 * \param loopindices: is an array of an int[4] per tessface,
 * mapping tessface's verts to loops indices.
 *
 * \note when mface is not NULL, mface[face_index].v4
 * is used to test quads, else, loopindices[face_index][3] is used.
 */
void BKE_mesh_loops_to_tessdata(CustomData *fdata,
                                CustomData *ldata,
                                MFace *mface,
                                const int *polyindices,
                                unsigned int (*loopindices)[4],
                                const int num_faces)
{
  /* Note: performances are sub-optimal when we get a NULL mface,
   *       we could be ~25% quicker with dedicated code...
   *       Issue is, unless having two different functions with nearly the same code,
   *       there's not much ways to solve this. Better imho to live with it for now. :/ --mont29
   */
  const int numUV = CustomData_number_of_layers(ldata, CD_MLOOPUV);
  const int numCol = CustomData_number_of_layers(ldata, CD_MLOOPCOL);
  const bool hasPCol = CustomData_has_layer(ldata, CD_PREVIEW_MLOOPCOL);
  const bool hasOrigSpace = CustomData_has_layer(ldata, CD_ORIGSPACE_MLOOP);
  const bool hasLoopNormal = CustomData_has_layer(ldata, CD_NORMAL);
  const bool hasLoopTangent = CustomData_has_layer(ldata, CD_TANGENT);
  int findex, i, j;
  const int *pidx;
  unsigned int(*lidx)[4];

  for (i = 0; i < numUV; i++) {
    MTFace *texface = CustomData_get_layer_n(fdata, CD_MTFACE, i);
    MLoopUV *mloopuv = CustomData_get_layer_n(ldata, CD_MLOOPUV, i);

    for (findex = 0, pidx = polyindices, lidx = loopindices; findex < num_faces;
         pidx++, lidx++, findex++, texface++) {
      for (j = (mface ? mface[findex].v4 : (*lidx)[3]) ? 4 : 3; j--;) {
        copy_v2_v2(texface->uv[j], mloopuv[(*lidx)[j]].uv);
      }
    }
  }

  for (i = 0; i < numCol; i++) {
    MCol(*mcol)[4] = CustomData_get_layer_n(fdata, CD_MCOL, i);
    MLoopCol *mloopcol = CustomData_get_layer_n(ldata, CD_MLOOPCOL, i);

    for (findex = 0, lidx = loopindices; findex < num_faces; lidx++, findex++, mcol++) {
      for (j = (mface ? mface[findex].v4 : (*lidx)[3]) ? 4 : 3; j--;) {
        MESH_MLOOPCOL_TO_MCOL(&mloopcol[(*lidx)[j]], &(*mcol)[j]);
      }
    }
  }

  if (hasPCol) {
    MCol(*mcol)[4] = CustomData_get_layer(fdata, CD_PREVIEW_MCOL);
    MLoopCol *mloopcol = CustomData_get_layer(ldata, CD_PREVIEW_MLOOPCOL);

    for (findex = 0, lidx = loopindices; findex < num_faces; lidx++, findex++, mcol++) {
      for (j = (mface ? mface[findex].v4 : (*lidx)[3]) ? 4 : 3; j--;) {
        MESH_MLOOPCOL_TO_MCOL(&mloopcol[(*lidx)[j]], &(*mcol)[j]);
      }
    }
  }

  if (hasOrigSpace) {
    OrigSpaceFace *of = CustomData_get_layer(fdata, CD_ORIGSPACE);
    OrigSpaceLoop *lof = CustomData_get_layer(ldata, CD_ORIGSPACE_MLOOP);

    for (findex = 0, lidx = loopindices; findex < num_faces; lidx++, findex++, of++) {
      for (j = (mface ? mface[findex].v4 : (*lidx)[3]) ? 4 : 3; j--;) {
        copy_v2_v2(of->uv[j], lof[(*lidx)[j]].uv);
      }
    }
  }

  if (hasLoopNormal) {
    short(*fnors)[4][3] = CustomData_get_layer(fdata, CD_TESSLOOPNORMAL);
    float(*lnors)[3] = CustomData_get_layer(ldata, CD_NORMAL);

    for (findex = 0, lidx = loopindices; findex < num_faces; lidx++, findex++, fnors++) {
      for (j = (mface ? mface[findex].v4 : (*lidx)[3]) ? 4 : 3; j--;) {
        normal_float_to_short_v3((*fnors)[j], lnors[(*lidx)[j]]);
      }
    }
  }

  if (hasLoopTangent) {
    /* need to do for all uv maps at some point */
    float(*ftangents)[4] = CustomData_get_layer(fdata, CD_TANGENT);
    float(*ltangents)[4] = CustomData_get_layer(ldata, CD_TANGENT);

    for (findex = 0, pidx = polyindices, lidx = loopindices; findex < num_faces;
         pidx++, lidx++, findex++) {
      int nverts = (mface ? mface[findex].v4 : (*lidx)[3]) ? 4 : 3;
      for (j = nverts; j--;) {
        copy_v4_v4(ftangents[findex * 4 + j], ltangents[(*lidx)[j]]);
      }
    }
  }
}

void BKE_mesh_tangent_loops_to_tessdata(CustomData *fdata,
                                        CustomData *ldata,
                                        MFace *mface,
                                        const int *polyindices,
                                        unsigned int (*loopindices)[4],
                                        const int num_faces,
                                        const char *layer_name)
{
  /* Note: performances are sub-optimal when we get a NULL mface,
   *       we could be ~25% quicker with dedicated code...
   *       Issue is, unless having two different functions with nearly the same code,
   *       there's not much ways to solve this. Better imho to live with it for now. :/ --mont29
   */

  float(*ftangents)[4] = NULL;
  float(*ltangents)[4] = NULL;

  int findex, j;
  const int *pidx;
  unsigned int(*lidx)[4];

  if (layer_name) {
    ltangents = CustomData_get_layer_named(ldata, CD_TANGENT, layer_name);
  }
  else {
    ltangents = CustomData_get_layer(ldata, CD_TANGENT);
  }

  if (ltangents) {
    /* need to do for all uv maps at some point */
    if (layer_name) {
      ftangents = CustomData_get_layer_named(fdata, CD_TANGENT, layer_name);
    }
    else {
      ftangents = CustomData_get_layer(fdata, CD_TANGENT);
    }
    if (ftangents) {
      for (findex = 0, pidx = polyindices, lidx = loopindices; findex < num_faces;
           pidx++, lidx++, findex++) {
        int nverts = (mface ? mface[findex].v4 : (*lidx)[3]) ? 4 : 3;
        for (j = nverts; j--;) {
          copy_v4_v4(ftangents[findex * 4 + j], ltangents[(*lidx)[j]]);
        }
      }
    }
  }
}

/**
 * Flip a single MLoop's #MDisps structure,
 * low level function to be called from face-flipping code which re-arranged the mdisps themselves.
 */
void BKE_mesh_mdisp_flip(MDisps *md, const bool use_loop_mdisp_flip)
{
  if (UNLIKELY(!md->totdisp || !md->disps)) {
    return;
  }

  const int sides = (int)sqrt(md->totdisp);
  float(*co)[3] = md->disps;

  for (int x = 0; x < sides; x++) {
    float *co_a, *co_b;

    for (int y = 0; y < x; y++) {
      co_a = co[y * sides + x];
      co_b = co[x * sides + y];

      swap_v3_v3(co_a, co_b);
      SWAP(float, co_a[0], co_a[1]);
      SWAP(float, co_b[0], co_b[1]);

      if (use_loop_mdisp_flip) {
        co_a[2] *= -1.0f;
        co_b[2] *= -1.0f;
      }
    }

    co_a = co[x * sides + x];

    SWAP(float, co_a[0], co_a[1]);

    if (use_loop_mdisp_flip) {
      co_a[2] *= -1.0f;
    }
  }
}

/**
 * Flip (invert winding of) the given \a mpoly, i.e. reverse order of its loops
 * (keeping the same vertex as 'start point').
 *
 * \param mpoly: the polygon to flip.
 * \param mloop: the full loops array.
 * \param ldata: the loops custom data.
 */
void BKE_mesh_polygon_flip_ex(MPoly *mpoly,
                              MLoop *mloop,
                              CustomData *ldata,
                              float (*lnors)[3],
                              MDisps *mdisp,
                              const bool use_loop_mdisp_flip)
{
  int loopstart = mpoly->loopstart;
  int loopend = loopstart + mpoly->totloop - 1;
  const bool loops_in_ldata = (CustomData_get_layer(ldata, CD_MLOOP) == mloop);

  if (mdisp) {
    for (int i = loopstart; i <= loopend; i++) {
      BKE_mesh_mdisp_flip(&mdisp[i], use_loop_mdisp_flip);
    }
  }

  /* Note that we keep same start vertex for flipped face. */

  /* We also have to update loops edge
   * (they will get their original 'other edge', that is,
   * the original edge of their original previous loop)... */
  unsigned int prev_edge_index = mloop[loopstart].e;
  mloop[loopstart].e = mloop[loopend].e;

  for (loopstart++; loopend > loopstart; loopstart++, loopend--) {
    mloop[loopend].e = mloop[loopend - 1].e;
    SWAP(unsigned int, mloop[loopstart].e, prev_edge_index);

    if (!loops_in_ldata) {
      SWAP(MLoop, mloop[loopstart], mloop[loopend]);
    }
    if (lnors) {
      swap_v3_v3(lnors[loopstart], lnors[loopend]);
    }
    CustomData_swap(ldata, loopstart, loopend);
  }
  /* Even if we did not swap the other 'pivot' loop, we need to set its swapped edge. */
  if (loopstart == loopend) {
    mloop[loopstart].e = prev_edge_index;
  }
}

void BKE_mesh_polygon_flip(MPoly *mpoly, MLoop *mloop, CustomData *ldata)
{
  MDisps *mdisp = CustomData_get_layer(ldata, CD_MDISPS);
  BKE_mesh_polygon_flip_ex(mpoly, mloop, ldata, NULL, mdisp, true);
}

/**
 * Flip (invert winding of) all polygons (used to inverse their normals).
 *
 * \note Invalidates tessellation, caller must handle that.
 */
void BKE_mesh_polygons_flip(MPoly *mpoly, MLoop *mloop, CustomData *ldata, int totpoly)
{
  MDisps *mdisp = CustomData_get_layer(ldata, CD_MDISPS);
  MPoly *mp;
  int i;

  for (mp = mpoly, i = 0; i < totpoly; mp++, i++) {
    BKE_mesh_polygon_flip_ex(mp, mloop, ldata, NULL, mdisp, true);
  }
}

/* -------------------------------------------------------------------- */
/** \name Mesh Flag Flushing
 * \{ */

/* update the hide flag for edges and faces from the corresponding
 * flag in verts */
void BKE_mesh_flush_hidden_from_verts_ex(const MVert *mvert,
                                         const MLoop *mloop,
                                         MEdge *medge,
                                         const int totedge,
                                         MPoly *mpoly,
                                         const int totpoly)
{
  int i, j;

  for (i = 0; i < totedge; i++) {
    MEdge *e = &medge[i];
    if (mvert[e->v1].flag & ME_HIDE || mvert[e->v2].flag & ME_HIDE) {
      e->flag |= ME_HIDE;
    }
    else {
      e->flag &= ~ME_HIDE;
    }
  }
  for (i = 0; i < totpoly; i++) {
    MPoly *p = &mpoly[i];
    p->flag &= (char)~ME_HIDE;
    for (j = 0; j < p->totloop; j++) {
      if (mvert[mloop[p->loopstart + j].v].flag & ME_HIDE) {
        p->flag |= ME_HIDE;
      }
    }
  }
}
void BKE_mesh_flush_hidden_from_verts(Mesh *me)
{
  BKE_mesh_flush_hidden_from_verts_ex(
      me->mvert, me->mloop, me->medge, me->totedge, me->mpoly, me->totpoly);
}

void BKE_mesh_flush_hidden_from_polys_ex(MVert *mvert,
                                         const MLoop *mloop,
                                         MEdge *medge,
                                         const int UNUSED(totedge),
                                         const MPoly *mpoly,
                                         const int totpoly)
{
  int i = totpoly;
  for (const MPoly *mp = mpoly; i--; mp++) {
    if (mp->flag & ME_HIDE) {
      const MLoop *ml;
      int j = mp->totloop;
      for (ml = &mloop[mp->loopstart]; j--; ml++) {
        mvert[ml->v].flag |= ME_HIDE;
        medge[ml->e].flag |= ME_HIDE;
      }
    }
  }

  i = totpoly;
  for (const MPoly *mp = mpoly; i--; mp++) {
    if ((mp->flag & ME_HIDE) == 0) {
      const MLoop *ml;
      int j = mp->totloop;
      for (ml = &mloop[mp->loopstart]; j--; ml++) {
        mvert[ml->v].flag &= (char)~ME_HIDE;
        medge[ml->e].flag &= (short)~ME_HIDE;
      }
    }
  }
}
void BKE_mesh_flush_hidden_from_polys(Mesh *me)
{
  BKE_mesh_flush_hidden_from_polys_ex(
      me->mvert, me->mloop, me->medge, me->totedge, me->mpoly, me->totpoly);
}

/**
 * simple poly -> vert/edge selection.
 */
void BKE_mesh_flush_select_from_polys_ex(MVert *mvert,
                                         const int totvert,
                                         const MLoop *mloop,
                                         MEdge *medge,
                                         const int totedge,
                                         const MPoly *mpoly,
                                         const int totpoly)
{
  MVert *mv;
  MEdge *med;
  const MPoly *mp;

  int i = totvert;
  for (mv = mvert; i--; mv++) {
    mv->flag &= (char)~SELECT;
  }

  i = totedge;
  for (med = medge; i--; med++) {
    med->flag &= ~SELECT;
  }

  i = totpoly;
  for (mp = mpoly; i--; mp++) {
    /* assume if its selected its not hidden and none of its verts/edges are hidden
     * (a common assumption)*/
    if (mp->flag & ME_FACE_SEL) {
      const MLoop *ml;
      int j;
      j = mp->totloop;
      for (ml = &mloop[mp->loopstart]; j--; ml++) {
        mvert[ml->v].flag |= SELECT;
        medge[ml->e].flag |= SELECT;
      }
    }
  }
}
void BKE_mesh_flush_select_from_polys(Mesh *me)
{
  BKE_mesh_flush_select_from_polys_ex(
      me->mvert, me->totvert, me->mloop, me->medge, me->totedge, me->mpoly, me->totpoly);
}

void BKE_mesh_flush_select_from_verts_ex(const MVert *mvert,
                                         const int UNUSED(totvert),
                                         const MLoop *mloop,
                                         MEdge *medge,
                                         const int totedge,
                                         MPoly *mpoly,
                                         const int totpoly)
{
  MEdge *med;
  MPoly *mp;

  /* edges */
  int i = totedge;
  for (med = medge; i--; med++) {
    if ((med->flag & ME_HIDE) == 0) {
      if ((mvert[med->v1].flag & SELECT) && (mvert[med->v2].flag & SELECT)) {
        med->flag |= SELECT;
      }
      else {
        med->flag &= ~SELECT;
      }
    }
  }

  /* polys */
  i = totpoly;
  for (mp = mpoly; i--; mp++) {
    if ((mp->flag & ME_HIDE) == 0) {
      bool ok = true;
      const MLoop *ml;
      int j;
      j = mp->totloop;
      for (ml = &mloop[mp->loopstart]; j--; ml++) {
        if ((mvert[ml->v].flag & SELECT) == 0) {
          ok = false;
          break;
        }
      }

      if (ok) {
        mp->flag |= ME_FACE_SEL;
      }
      else {
        mp->flag &= (char)~ME_FACE_SEL;
      }
    }
  }
}
void BKE_mesh_flush_select_from_verts(Mesh *me)
{
  BKE_mesh_flush_select_from_verts_ex(
      me->mvert, me->totvert, me->mloop, me->medge, me->totedge, me->mpoly, me->totpoly);
}