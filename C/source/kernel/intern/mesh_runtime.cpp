#include "atomic_ops.h"

#include "MEM_guardedalloc.cuh"
#include "memarena.h"

#include "mesh_types.h"
#include "meshdata_types.cuh"
#include "object_types.cuh"

#include "B_math.h"
#include "threads.h"
#include "task.h"

#include "bvhutils.h"
#include "BKE_lib_id.h"
#include "BKE_mesh.h"
#include "mesh_runtime.h"
#include "BKE_shrinkwrap.h"
#include "BKE_subdiv_ccg.h"

/* -------------------------------------------------------------------- */
/** \name Mesh Runtime Struct Utils
 * \{ */

/**
 * Default values defined at read time.
 */
void BKE_mesh_runtime_reset(Mesh *mesh)
{
    memset(&mesh->runtime, 0, sizeof(mesh->runtime));
    mesh->runtime.eval_mutex = MEM_mallocN(sizeof(ThreadMutex), "mesh runtime eval_mutex");
    BLI_mutex_init((ThreadMutex*)mesh->runtime.eval_mutex);
}

/* Clear all pointers which we don't want to be shared on copying the datablock.
 * However, keep all the flags which defines what the mesh is (for example, that
 * it's deformed only, or that its custom data layers are out of date.) */
void BKE_mesh_runtime_reset_on_copy(Mesh *mesh, const int UNUSED(flag))
{
    Mesh_Runtime *runtime = &mesh->runtime;

    runtime->mesh_eval = NULL;
    runtime->edit_data = NULL;
    runtime->batch_cache = NULL;
    runtime->subdiv_ccg = NULL;
    memset(&runtime->looptris, 0, sizeof(runtime->looptris));
    runtime->bvh_cache = NULL;
    runtime->shrinkwrap_data = NULL;

    mesh->runtime.eval_mutex = (ThreadMutex*)MEM_mallocN(sizeof(ThreadMutex), "mesh runtime eval_mutex");
    BLI_mutex_init((ThreadMutex*)mesh->runtime.eval_mutex);
}

static uint* pf_tri_add(PolyFill* pf)
{
    return pf->tris[pf->tris_num++];
}


static bool pf_ear_tip_check(PolyFill* pf, PolyIndex* pi_ear_tip)
{
#ifndef USE_KDTREE
    /* localize */
    const float(*coords)[2] = pf->coords;
    PolyIndex* pi_curr;

    const float* v1, * v2, * v3;
#endif

#if defined(USE_CONVEX_SKIP) && !defined(USE_KDTREE)
    uint coords_num_concave_checked = 0;
#endif

#ifdef USE_CONVEX_SKIP

#  ifdef USE_CONVEX_SKIP_TEST
    /* check if counting is wrong */
    {
        uint coords_num_concave_test = 0;
        PolyIndex* pi_iter = pi_ear_tip;
        do {
            if (pi_iter->sign != CONVEX) {
                coords_num_concave_test += 1;
            }
        } while ((pi_iter = pi_iter->next) != pi_ear_tip);
        BLI_assert(coords_num_concave_test == pf->coords_num_concave);
    }
#  endif

    /* fast-path for circles */
    if (pf->coords_num_concave == 0) {
        return true;
    }
#endif

    if (UNLIKELY(pi_ear_tip->sign == CONCAVE)) {
        return false;
    }

#ifdef USE_KDTREE
    {
        const uint ind[3] = { pi_ear_tip->index, pi_ear_tip->next->index, pi_ear_tip->prev->index };

        if (kdtree2d_isect_tri(&pf->kdtree, ind)) {
            return false;
        }
    }
#else

    v1 = coords[pi_ear_tip->prev->index];
    v2 = coords[pi_ear_tip->index];
    v3 = coords[pi_ear_tip->next->index];

    /* Check if any point is inside the triangle formed by previous, current and next vertices.
     * Only consider vertices that are not part of this triangle,
     * or else we'll always find one inside. */

    for (pi_curr = pi_ear_tip->next->next; pi_curr != pi_ear_tip->prev; pi_curr = pi_curr->next) {
        /* Concave vertices can obviously be inside the candidate ear,
         * but so can tangential vertices if they coincide with one of the triangle's vertices. */
        if (pi_curr->sign != CONVEX) {
            const float* v = coords[pi_curr->index];
            /* Because the polygon has clockwise winding order,
             * the area sign will be positive if the point is strictly inside.
             * It will be 0 on the edge, which we want to include as well. */

             /* NOTE: check (v3, v1) first since it fails _far_ more often than the other 2 checks
              * (those fail equally).
              * It's logical - the chance is low that points exist on the
              * same side as the ear we're clipping off. */
            if ((span_tri_v2_sign(v3, v1, v) != CONCAVE) && (span_tri_v2_sign(v1, v2, v) != CONCAVE) && (span_tri_v2_sign(v2, v3, v) != CONCAVE)) 
            {
                return false;
            }

#  ifdef USE_CONVEX_SKIP
            coords_num_concave_checked += 1;
            if (coords_num_concave_checked == pf->coords_num_concave) {
                break;
            }
#  endif
        }
    }
#endif /* USE_KDTREE */

    return true;
}

void BKE_mesh_runtime_clear_cache(Mesh *mesh)
{
    if (mesh->runtime.eval_mutex != NULL) 
    {
        BLI_mutex_end((ThreadMutex*)mesh->runtime.eval_mutex);
        MEM_freeN(mesh->runtime.eval_mutex);
        mesh->runtime.eval_mutex = NULL;
    }
    if (mesh->runtime.mesh_eval != NULL) 
    {
        mesh->runtime.mesh_eval->edit_mesh = NULL;
        BKE_id_free(NULL, mesh->runtime.mesh_eval);
        mesh->runtime.mesh_eval = NULL;
    }
    BKE_mesh_runtime_clear_geometry(mesh);
    BKE_mesh_batch_cache_free(mesh);
    BKE_mesh_runtime_clear_edit_data(mesh);
}


static PolyIndex* pf_ear_tip_find(PolyFill* pf
#ifdef USE_CLIP_EVEN
    ,
    PolyIndex* pi_ear_init
#endif
#ifdef USE_CLIP_SWEEP
    ,
    bool reverse
#endif
)
{
    /* localize */
    const uint coords_num = pf->coords_num;
    PolyIndex* pi_ear;

    uint i;

#ifdef USE_CLIP_EVEN
    pi_ear = pi_ear_init;
#else
    pi_ear = pf->indices;
#endif

    i = coords_num;
    while (i--) {
        if (pf_ear_tip_check(pf, pi_ear)) {
            return pi_ear;
        }
#ifdef USE_CLIP_SWEEP
        pi_ear = reverse ? pi_ear->prev : pi_ear->next;
#else
        pi_ear = pi_ear->next;
#endif
    }

    /* Desperate mode: if no vertex is an ear tip,
     * we are dealing with a degenerate polygon (e.g. nearly collinear).
     * Note that the input was not necessarily degenerate,
     * but we could have made it so by clipping some valid ears.
     *
     * Idea taken from Martin Held, "FIST: Fast industrial-strength triangulation of polygons",
     * Algorithmica (1998),
     * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.115.291
     *
     * Return a convex or tangential vertex if one exists.
     */

#ifdef USE_CLIP_EVEN
    pi_ear = pi_ear_init;
#else
    pi_ear = pf->indices;
#endif

    i = coords_num;
    while (i--) {
        if (pi_ear->sign != CONCAVE) {
            return pi_ear;
        }
        pi_ear = pi_ear->next;
    }

    /* If all vertices are concave, just return the last one. */
    return pi_ear;
}

static void pf_coord_remove(PolyFill* pf, PolyIndex* pi)
{
#ifdef USE_KDTREE
    /* avoid double lookups, since convex coords are ignored when testing intersections */
    if (pf->kdtree.node_num) {
        kdtree2d_node_remove(&pf->kdtree, pi->index);
    }
#endif

    pi->next->prev = pi->prev;
    pi->prev->next = pi->next;

    if (pf->indices == pi) 
    {
        pf->indices = pi->next;
    }
#ifdef DEBUG
    pi->index = (uint)-1;
    pi->next = pi->prev = NULL;
#endif

    pf->coords_num -= 1;
}

static void pf_ear_tip_cut(PolyFill* pf, PolyIndex* pi_ear_tip)
{
    uint* tri = pf_tri_add(pf);

    tri[0] = pi_ear_tip->prev->index;
    tri[1] = pi_ear_tip->index;
    tri[2] = pi_ear_tip->next->index;

    pf_coord_remove(pf, pi_ear_tip);
}

static void pf_triangulate(PolyFill* pf)
{
    PolyIndex* pi_ear;

#ifdef USE_CLIP_EVEN
    PolyIndex* pi_ear_init = pf->indices;
#endif
#ifdef USE_CLIP_SWEEP
    bool reverse = false;
#endif

    while (pf->coords_num > 3) {
        PolyIndex* pi_prev, * pi_next;
        unsigned char sign_orig_prev, sign_orig_next;

        pi_ear = pf_ear_tip_find(pf
#ifdef USE_CLIP_EVEN
            ,pi_ear_init
#endif
#ifdef USE_CLIP_SWEEP
            ,reverse
#endif
        );

#ifdef USE_CONVEX_SKIP
        if (pi_ear->sign != CONVEX) {
            pf->coords_num_concave -= 1;
        }
#endif

        pi_prev = pi_ear->prev;
        pi_next = pi_ear->next;

        pf_ear_tip_cut(pf, pi_ear);

        /* The type of the two vertices adjacent to the clipped vertex may have changed. */
        sign_orig_prev = pi_prev->sign;
        sign_orig_next = pi_next->sign;

        /* check if any verts became convex the (else if)
         * case is highly unlikely but may happen with degenerate polygons */
        if (sign_orig_prev != CONVEX) {
            pf_coord_sign_calc(pf, pi_prev);
#ifdef USE_CONVEX_SKIP
            if (pi_prev->sign == CONVEX) {
                pf->coords_num_concave -= 1;
#  ifdef USE_KDTREE
                kdtree2d_node_remove(&pf->kdtree, pi_prev->index);
#  endif
            }
#endif
        }
        if (sign_orig_next != CONVEX) {
            pf_coord_sign_calc(pf, pi_next);
#ifdef USE_CONVEX_SKIP
            if (pi_next->sign == CONVEX) {
                pf->coords_num_concave -= 1;
#  ifdef USE_KDTREE
                kdtree2d_node_remove(&pf->kdtree, pi_next->index);
#  endif
            }
#endif
        }

#ifdef USE_CLIP_EVEN
#  ifdef USE_CLIP_SWEEP
        pi_ear_init = reverse ? pi_prev->prev : pi_next->next;
#  else
        pi_ear_init = pi_next->next;
#  endif
#endif

#ifdef USE_CLIP_EVEN
#  ifdef USE_CLIP_SWEEP
        if (pi_ear_init->sign != CONVEX) {
            /* take the extra step since this ear isn't a good candidate */
            pi_ear_init = reverse ? pi_ear_init->prev : pi_ear_init->next;
            reverse = !reverse;
        }
#  endif
#elseif USE_CLIP_SWEEP
        if ((reverse ? pi_prev->prev : pi_next->next)->sign != CONVEX) {
            reverse = !reverse;
        }
#endif
    }

    if (pf->coords_num == 3) {
        uint* tri = pf_tri_add(pf);
        pi_ear = pf->indices;
        tri[0] = pi_ear->index;
        pi_ear = pi_ear->next;
        tri[1] = pi_ear->index;
        pi_ear = pi_ear->next;
        tri[2] = pi_ear->index;
    }
}

static void polyfill_calc(PolyFill* pf)
{
#ifdef USE_KDTREE
#  ifdef USE_CONVEX_SKIP
    if (pf->coords_num_concave)
#  endif
    {
        kdtree2d_new(&pf->kdtree, pf->coords_num_concave, pf->coords);
        kdtree2d_init(&pf->kdtree, pf->coords_num, pf->indices);
        kdtree2d_balance(&pf->kdtree);
        kdtree2d_init_mapping(&pf->kdtree);
    }
#endif

    pf_triangulate(pf);
}

signed char signum_enum(float a)
{
    if (UNLIKELY(a == 0.0f)) {
        return 0;
    }
    if (a > 0.0f) {
        return 1;
    }

    return -1;
}

float area_tri_signed_v2_alt_2x(const float v1[2], const float v2[2], const float v3[2])
{
    float d2[2], d3[2];
    sub_v2_v2v2(d2, v2, v1);
    sub_v2_v2v2(d3, v3, v1);
    return (d2[0] * d3[1]) - (d3[0] * d2[1]);
}

signed char span_tri_v2_sign(const float v1[2], const float v2[2], const float v3[2])
{
    return signum_enum(area_tri_signed_v2_alt_2x(v3, v2, v1));
}

/**
 * \return CONCAVE, TANGENTIAL or CONVEX
 */
void pf_coord_sign_calc(PolyFill* pf, PolyIndex* pi)
{
    /* localize */
    const float(*coords)[2] = pf->coords;

    pi->sign = span_tri_v2_sign(coords[pi->prev->index], coords[pi->index], coords[pi->next->index]);
}

/**
 * Initializes the #PolyFill structure before tessellating with #polyfill_calc.
 */
static void polyfill_prepare(PolyFill* pf,
    const float(*coords)[2],
    const uint coords_num,
    int coords_sign,
    uint(*r_tris)[3],
    PolyIndex* r_indices)
{
    /* localize */
    PolyIndex* indices = r_indices;

    uint i;

    /* assign all polyfill members here */
    pf->indices = r_indices;
    pf->coords = coords;
    pf->coords_num = coords_num;
#ifdef USE_CONVEX_SKIP
    pf->coords_num_concave = 0;
#endif
    pf->tris = r_tris;
    pf->tris_num = 0;

    if (coords_sign == 0) {
        coords_sign = (cross_poly_v2(coords, coords_num) >= 0.0f) ? 1 : -1;
    }
    else {
        /* check we're passing in correct args */
#ifdef USE_STRICT_ASSERT
#  ifndef NDEBUG
        if (coords_sign == 1) {
            BLI_assert(cross_poly_v2(coords, coords_num) >= 0.0f);
        }
        else {
            BLI_assert(cross_poly_v2(coords, coords_num) <= 0.0f);
        }
#  endif
#endif
    }

    if (coords_sign == 1) {
        for (i = 0; i < coords_num; i++) {
            indices[i].next = &indices[i + 1];
            indices[i].prev = &indices[i - 1];
            indices[i].index = i;
        }
    }
    else {
        /* reversed */
        uint n = coords_num - 1;
        for (i = 0; i < coords_num; i++) {
            indices[i].next = &indices[i + 1];
            indices[i].prev = &indices[i - 1];
            indices[i].index = (n - i);
        }
    }
    indices[0].prev = &indices[coords_num - 1];
    indices[coords_num - 1].next = &indices[0];

    for (i = 0; i < coords_num; i++) 
    {
        PolyIndex* pi = &indices[i];
        pf_coord_sign_calc(pf, pi);
#ifdef USE_CONVEX_SKIP
        if (pi->sign != CONVEX) {
            pf->coords_num_concave += 1;
        }
#endif
    }
}

void BLI_polyfill_calc_arena(const float(*coords)[2],
    const uint coords_num, const int coords_sign,
    uint(*r_tris)[3], struct MemArena* arena)
{
    PolyFill pf;
    PolyIndex* indices = (PolyIndex*)BLI_memarena_alloc(arena, sizeof(*indices) * coords_num);
    polyfill_prepare(&pf, coords, coords_num, coords_sign, r_tris, indices);

#ifdef USE_KDTREE
    if (pf.coords_num_concave) {
        pf.kdtree.nodes = BLI_memarena_alloc(arena, sizeof(*pf.kdtree.nodes) * pf.coords_num_concave);
        pf.kdtree.nodes_map = memset(
            BLI_memarena_alloc(arena, sizeof(*pf.kdtree.nodes_map) * coords_num),
            0xff,
            sizeof(*pf.kdtree.nodes_map) * coords_num);
    }
    else {
        pf.kdtree.node_num = 0;
    }
#endif

    polyfill_calc(&pf);
}

/* This is a ported copy of DM_ensure_looptri_data(dm) */
/**
 * Ensure the array is large enough
 *
 * \note This function must always be thread-protected by caller.
 * It should only be used by internal code.
 */
void mesh_ensure_looptri_data(Mesh *mesh)
{
  const unsigned int totpoly = mesh->totpoly;
  const int looptris_len = poly_to_tri_count(totpoly, mesh->totloop);

  BLI_assert(mesh->runtime.looptris.array_wip == NULL);

  SWAP(MLoopTri *, mesh->runtime.looptris.array, mesh->runtime.looptris.array_wip);

  if ((looptris_len > mesh->runtime.looptris.len_alloc) ||
      (looptris_len < mesh->runtime.looptris.len_alloc * 2) || (totpoly == 0)) {
    MEM_SAFE_FREE(mesh->runtime.looptris.array_wip);
    mesh->runtime.looptris.len_alloc = 0;
    mesh->runtime.looptris.len = 0;
  }

  if (totpoly) {
    if (mesh->runtime.looptris.array_wip == NULL) {
      mesh->runtime.looptris.array_wip = (MLoopTri*)MEM_malloc_arrayN(looptris_len, sizeof(*mesh->runtime.looptris.array_wip), __func__);
      mesh->runtime.looptris.len_alloc = looptris_len;
    }

    mesh->runtime.looptris.len = looptris_len;
  }
}


bool is_quad_flip_v3_first_third_fast_with_normal(const float v1[3],
    const float v2[3],
    const float v3[3],
    const float v4[3],
    const float normal[3])
{
    float dir_v3v1[3], tangent[3];
    sub_v3_v3v3(dir_v3v1, v3, v1);
    cross_v3_v3v3(tangent, dir_v3v1, normal);
    const float dot = dot_v3v3(v1, tangent);
    return (dot_v3v3(v4, tangent) >= dot) || (dot_v3v3(v2, tangent) <= dot);
}

void mesh_calc_tessellation_for_face_impl(const MLoop* mloop,
    const MPoly* mpoly,
    const MVert* mvert,
    uint poly_index,
    MLoopTri* mlt,
    MemArena** pf_arena_p,
    const bool face_normal,
    const float normal_precalc[3])
{
    const uint mp_loopstart = (uint)mpoly[poly_index].loopstart;
    const uint mp_totloop = (uint)mpoly[poly_index].totloop;

#define ML_TO_MLT(i1, i2, i3) { ARRAY_SET_ITEMS(mlt->tri, mp_loopstart + i1, mp_loopstart + i2, mp_loopstart + i3); mlt->poly = poly_index; } ((void)0)

    switch (mp_totloop) {
    case 3: {
        ML_TO_MLT(0, 1, 2);
        break;
    }
    case 4: {
        ML_TO_MLT(0, 1, 2);
        MLoopTri* mlt_a = mlt++;
        ML_TO_MLT(0, 2, 3);
        MLoopTri* mlt_b = mlt;

        if ((face_normal ? is_quad_flip_v3_first_third_fast_with_normal(
            /* Simpler calculation (using the normal). */
            mvert[mloop[mlt_a->tri[0]].v].co,
            mvert[mloop[mlt_a->tri[1]].v].co,
            mvert[mloop[mlt_a->tri[2]].v].co,
            mvert[mloop[mlt_b->tri[2]].v].co,
            normal_precalc) :
            is_quad_flip_v3_first_third_fast(
                /* Expensive calculation (no normal). */
                mvert[mloop[mlt_a->tri[0]].v].co,
                mvert[mloop[mlt_a->tri[1]].v].co,
                mvert[mloop[mlt_a->tri[2]].v].co,
                mvert[mloop[mlt_b->tri[2]].v].co))) {
            /* Flip out of degenerate 0-2 state. */
            mlt_a->tri[2] = mlt_b->tri[2];
            mlt_b->tri[0] = mlt_a->tri[1];
        }
        break;
    }
    default: {
        const MLoop* ml;
        float axis_mat[3][3];

        /* Calculate `axis_mat` to project verts to 2D. */
        if (face_normal == false) {
            float normal[3];
            const float* co_curr, * co_prev;

            zero_v3(normal);

            /* Calc normal, flipped: to get a positive 2D cross product. */
            ml = mloop + mp_loopstart;
            co_prev = mvert[ml[mp_totloop - 1].v].co;
            for (uint j = 0; j < mp_totloop; j++, ml++) {
                co_curr = mvert[ml->v].co;
                add_newell_cross_v3_v3v3(normal, co_prev, co_curr);
                co_prev = co_curr;
            }
            if (UNLIKELY(normalize_v3(normal) == 0.0f)) {
                normal[2] = 1.0f;
            }
            axis_dominant_v3_to_m3_negate(axis_mat, normal);
        }
        else {
            axis_dominant_v3_to_m3_negate(axis_mat, normal_precalc);
        }

        const uint totfilltri = mp_totloop - 2;

        MemArena* pf_arena = *pf_arena_p;
        if (pf_arena == nullptr) 
        {
            pf_arena = *pf_arena_p = BLI_memarena_new(BLI_MEMARENA_STD_BUFSIZE, __func__);
        }

        uint(*tris)[3] = static_cast<uint(*)[3]>(BLI_memarena_alloc(pf_arena, sizeof(*tris) * (size_t)totfilltri));
        float(*projverts)[2] = static_cast<float(*)[2]>(BLI_memarena_alloc(pf_arena, sizeof(*projverts) * (size_t)mp_totloop));

        ml = mloop + mp_loopstart;
        for (uint j = 0; j < mp_totloop; j++, ml++) {
            mul_v2_m3v3(projverts[j], axis_mat, mvert[ml->v].co);
        }

        BLI_polyfill_calc_arena(projverts, mp_totloop, 1, tris, pf_arena);

        /* Apply fill. */
        for (uint j = 0; j < totfilltri; j++, mlt++) {
            const uint* tri = tris[j];
            ML_TO_MLT(tri[0], tri[1], tri[2]);
        }

        BLI_memarena_clear(pf_arena);

        break;
    }
    }
#undef ML_TO_MLT
}

static void mesh_calc_tessellation_for_face(const MLoop* mloop,
    const MPoly* mpoly, const MVert* mvert,
    uint poly_index, MLoopTri* mlt, MemArena** pf_arena_p)
{
    mesh_calc_tessellation_for_face_impl(
        mloop, mpoly, mvert, poly_index, mlt, pf_arena_p, false, nullptr);
}


static void mesh_calc_tessellation_for_face_with_normal(const MLoop* mloop,
    const MPoly* mpoly, const MVert* mvert,
    uint poly_index, MLoopTri* mlt, MemArena** pf_arena_p,
    const float normal_precalc[3])
{
    mesh_calc_tessellation_for_face_impl(
        mloop, mpoly, mvert, poly_index, mlt, pf_arena_p, true, normal_precalc);
}

static void mesh_recalc_looptri__single_threaded(const MLoop* mloop, const MPoly* mpoly, const MVert* mvert,
                                                 int totloop, int totpoly,  MLoopTri* mlooptri, const float(*poly_normals)[3])
{
    MemArena* pf_arena = nullptr;
    const MPoly* mp = mpoly;
    uint tri_index = 0;

    if (poly_normals != nullptr) 
    {
        for (uint poly_index = 0; poly_index < (uint)totpoly; poly_index++, mp++) 
        {
            mesh_calc_tessellation_for_face_with_normal(mloop,  mpoly, mvert,
                poly_index, &mlooptri[tri_index], &pf_arena, poly_normals[poly_index]);
            tri_index += (uint)(mp->totloop - 2);
        }
    }
    else 
    {
        for (uint poly_index = 0; poly_index < (uint)totpoly; poly_index++, mp++) 
        {
            mesh_calc_tessellation_for_face(mloop, mpoly, mvert, poly_index, &mlooptri[tri_index], &pf_arena);
            tri_index += (uint)(mp->totloop - 2);
        }
    }

    if (pf_arena) 
    {
        BLI_memarena_free(pf_arena);
        pf_arena = nullptr;
    }
    BLI_assert(tri_index == (uint)poly_to_tri_count(totpoly, totloop));
    UNUSED_VARS_NDEBUG(totloop);
}

static void mesh_calc_tessellation_for_face_fn(void* __restrict userdata, const int index, const TaskParallelTLS* __restrict tls)
{
    const TessellationUserData* data = static_cast<const TessellationUserData*>(userdata);
    TessellationUserTLS* tls_data = static_cast<TessellationUserTLS*>(tls->userdata_chunk);
    const int tri_index = poly_to_tri_count(index, data->mpoly[index].loopstart);
    mesh_calc_tessellation_for_face_impl(data->mloop,
        data->mpoly, data->mvert,
        (uint)index, &data->mlooptri[tri_index],
        &tls_data->pf_arena, false, nullptr);
}

static void mesh_calc_tessellation_for_face_with_normal_fn(void* __restrict userdata, const int index, const TaskParallelTLS* __restrict tls)
{
    const TessellationUserData* data = static_cast<const TessellationUserData*>(userdata);
    TessellationUserTLS* tls_data = static_cast<TessellationUserTLS*>(tls->userdata_chunk);
    const int tri_index = poly_to_tri_count(index, data->mpoly[index].loopstart);
    mesh_calc_tessellation_for_face_impl(data->mloop,
        data->mpoly, data->mvert,
        (uint)index, &data->mlooptri[tri_index],
        &tls_data->pf_arena, true, data->poly_normals[index]);
}

static void mesh_calc_tessellation_for_face_free_fn(const void* __restrict UNUSED(userdata),
    void* __restrict tls_v)
{
    TessellationUserTLS* tls_data = static_cast<TessellationUserTLS*>(tls_v);
    if (tls_data->pf_arena) 
    {
        BLI_memarena_free(tls_data->pf_arena);
    }
}

static void mesh_recalc_looptri__multi_threaded(const MLoop* mloop,
    const MPoly* mpoly,
    const MVert* mvert,
    int UNUSED(totloop),
    int totpoly,
    MLoopTri* mlooptri,
    const float(*poly_normals)[3])
{
    struct TessellationUserTLS tls_data_dummy = { nullptr };

    struct TessellationUserData data {};
    data.mloop = mloop;
    data.mpoly = mpoly;
    data.mvert = mvert;
    data.mlooptri = mlooptri;
    data.poly_normals = poly_normals;

    TaskParallelSettings settings;
    BLI_parallel_range_settings_defaults(&settings);

    settings.userdata_chunk = &tls_data_dummy;
    settings.userdata_chunk_size = sizeof(tls_data_dummy);

    settings.func_free = mesh_calc_tessellation_for_face_free_fn;

    BLI_task_parallel_range(0, totpoly, &data,
        poly_normals ? mesh_calc_tessellation_for_face_with_normal_fn :
        mesh_calc_tessellation_for_face_fn, &settings);
}

#define MESH_FACE_TESSELLATE_THREADED_LIMIT 4096
void BKE_mesh_recalc_looptri(const MLoop* mloop, const MPoly* mpoly,
    const MVert* mvert, int totloop, int totpoly, MLoopTri* mlooptri)
{
    if (totloop < MESH_FACE_TESSELLATE_THREADED_LIMIT) 
    {
        mesh_recalc_looptri__single_threaded(mloop, mpoly, mvert, totloop, totpoly, mlooptri, nullptr);
    }
    else 
    {
        mesh_recalc_looptri__multi_threaded(mloop, mpoly, mvert, totloop, totpoly, mlooptri, nullptr);
    }
}

/* This is a ported copy of CDDM_recalc_looptri(dm). */
void BKE_mesh_runtime_looptri_recalc(Mesh *mesh)
{
  mesh_ensure_looptri_data(mesh);
  BLI_assert(mesh->totpoly == 0 || mesh->runtime.looptris.array_wip != NULL);

  BKE_mesh_recalc_looptri(mesh->mloop,  mesh->mpoly,
                          mesh->mvert, mesh->totloop,
                          mesh->totpoly,  mesh->runtime.looptris.array_wip);

  BLI_assert(mesh->runtime.looptris.array == NULL);
  atomic_cas_ptr((void **)&mesh->runtime.looptris.array,
                 mesh->runtime.looptris.array,
                 mesh->runtime.looptris.array_wip);
  mesh->runtime.looptris.array_wip = NULL;
}

/* This is a ported copy of dm_getNumLoopTri(dm). */
int BKE_mesh_runtime_looptri_len(const Mesh *mesh)
{
  const int looptri_len = poly_to_tri_count(mesh->totpoly, mesh->totloop);
  BLI_assert(ELEM(mesh->runtime.looptris.len, 0, looptri_len));
  return looptri_len;
}

/* This is a ported copy of dm_getLoopTriArray(dm). */
const MLoopTri *BKE_mesh_runtime_looptri_ensure(Mesh *mesh)
{
  ThreadMutex *mesh_eval_mutex = (ThreadMutex *)mesh->runtime.eval_mutex;
  BLI_mutex_lock(mesh_eval_mutex);

  MLoopTri *looptri = mesh->runtime.looptris.array;

  if (looptri != NULL) {
    BLI_assert(BKE_mesh_runtime_looptri_len(mesh) == mesh->runtime.looptris.len);
  }
  else {
    BKE_mesh_runtime_looptri_recalc(mesh);
    looptri = mesh->runtime.looptris.array;
  }

  BLI_mutex_unlock(mesh_eval_mutex);

  return looptri;
}

/* This is a copy of DM_verttri_from_looptri(). */
void BKE_mesh_runtime_verttri_from_looptri(MVertTri *r_verttri,
                                           const MLoop *mloop,
                                           const MLoopTri *looptri,
                                           int looptri_num)
{
  for (int i = 0; i < looptri_num; i++) {
    r_verttri[i].tri[0] = mloop[looptri[i].tri[0]].v;
    r_verttri[i].tri[1] = mloop[looptri[i].tri[1]].v;
    r_verttri[i].tri[2] = mloop[looptri[i].tri[2]].v;
  }
}

bool BKE_mesh_runtime_ensure_edit_data(struct Mesh *mesh)
{
  if (mesh->runtime.edit_data != NULL) {
    return false;
  }

  mesh->runtime.edit_data = (EditMeshData*)MEM_callocN(sizeof(EditMeshData), "EditMeshData");
  return true;
}

bool BKE_mesh_runtime_reset_edit_data(Mesh *mesh)
{
  EditMeshData *edit_data = mesh->runtime.edit_data;
  if (edit_data == NULL) {
    return false;
  }

  MEM_SAFE_FREE(edit_data->polyCos);
  MEM_SAFE_FREE(edit_data->polyNos);
  MEM_SAFE_FREE(edit_data->vertexCos);
  MEM_SAFE_FREE(edit_data->vertexNos);

  return true;
}

bool BKE_mesh_runtime_clear_edit_data(Mesh *mesh)
{
  if (mesh->runtime.edit_data == NULL) {
    return false;
  }
  BKE_mesh_runtime_reset_edit_data(mesh);

  MEM_freeN(mesh->runtime.edit_data);
  mesh->runtime.edit_data = NULL;

  return true;
}

void BKE_mesh_runtime_clear_geometry(Mesh *mesh)
{
  if (mesh->runtime.bvh_cache) {
    bvhcache_free(mesh->runtime.bvh_cache);
    mesh->runtime.bvh_cache = NULL;
  }
  MEM_SAFE_FREE(mesh->runtime.looptris.array);
  /* TODO(sergey): Does this really belong here? */
  if (mesh->runtime.subdiv_ccg != NULL) {
    BKE_subdiv_ccg_destroy(mesh->runtime.subdiv_ccg);
    mesh->runtime.subdiv_ccg = NULL;
  }
  BKE_shrinkwrap_discard_boundary_data(mesh);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Mesh Batch Cache Callbacks
 * \{ */

/* Draw Engine */
void (*BKE_mesh_batch_cache_dirty_tag_cb)(Mesh *me, eMeshBatchDirtyMode mode) = NULL;
void (*BKE_mesh_batch_cache_free_cb)(Mesh *me) = NULL;

void BKE_mesh_batch_cache_dirty_tag(Mesh *me, eMeshBatchDirtyMode mode)
{
  if (me->runtime.batch_cache) {
    BKE_mesh_batch_cache_dirty_tag_cb(me, mode);
  }
}

void BKE_mesh_batch_cache_free(Mesh *me)
{
  if (me->runtime.batch_cache) {
    BKE_mesh_batch_cache_free_cb(me);
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Mesh runtime debug helpers.
 * \{ */
/* evaluated mesh info printing function,
 * to help track down differences output */

#ifndef NDEBUG
//#include "BLI_dynstr.h"

//static void mesh_runtime_debug_info_layers(DynStr *dynstr, CustomData *cd)
//{
//  int type;
//
//  for (type = 0; type < CD_NUMTYPES; type++) {
//    if (CustomData_has_layer(cd, type)) {
//      /* note: doesn't account for multiple layers */
//      const char *name = CustomData_layertype_name(type);
//      const int size = CustomData_sizeof(type);
//      const void *pt = CustomData_get_layer(cd, type);
//      const int pt_size = pt ? (int)(MEM_allocN_len(pt) / size) : 0;
//      const char *structname;
//      int structnum;
//      //CustomData_file_write_info(type, &structname, &structnum);
//      BLI_dynstr_appendf(dynstr, "        dict(name='%s', struct='%s', type=%d, ptr='%p', elem=%d, length=%d),\n",
//          name, structname, type, (const void *)pt, size, pt_size);
//    }
//  }
//}

//char *BKE_mesh_runtime_debug_info(Mesh *me_eval)
//{
//  DynStr *dynstr = BLI_dynstr_new();
//  char *ret;
//
//  BLI_dynstr_append(dynstr, "{\n");
//  BLI_dynstr_appendf(dynstr, "    'ptr': '%p',\n", (void *)me_eval);
//#  if 0
//  const char *tstr;
//  switch (me_eval->type) {
//    case DM_TYPE_CDDM:
//      tstr = "DM_TYPE_CDDM";
//      break;
//    case DM_TYPE_CCGDM:
//      tstr = "DM_TYPE_CCGDM";
//      break;
//    default:
//      tstr = "UNKNOWN";
//      break;
//  }
//  BLI_dynstr_appendf(dynstr, "    'type': '%s',\n", tstr);
//#  endif
//  BLI_dynstr_appendf(dynstr, "    'totvert': %d,\n", me_eval->totvert);
//  BLI_dynstr_appendf(dynstr, "    'totedge': %d,\n", me_eval->totedge);
//  BLI_dynstr_appendf(dynstr, "    'totface': %d,\n", me_eval->totface);
//  BLI_dynstr_appendf(dynstr, "    'totpoly': %d,\n", me_eval->totpoly);
//  BLI_dynstr_appendf(dynstr, "    'deformed_only': %d,\n", me_eval->runtime.deformed_only);
//
//  //BLI_dynstr_append(dynstr, "    'vertexLayers': (\n");
//  //mesh_runtime_debug_info_layers(dynstr, &me_eval->vdata);
//  //BLI_dynstr_append(dynstr, "    ),\n");
//
//  //BLI_dynstr_append(dynstr, "    'edgeLayers': (\n");
//  //mesh_runtime_debug_info_layers(dynstr, &me_eval->edata);
//  //BLI_dynstr_append(dynstr, "    ),\n");
//
//  //BLI_dynstr_append(dynstr, "    'loopLayers': (\n");
//  //mesh_runtime_debug_info_layers(dynstr, &me_eval->ldata);
//  //BLI_dynstr_append(dynstr, "    ),\n");
//
//  //BLI_dynstr_append(dynstr, "    'polyLayers': (\n");
//  //mesh_runtime_debug_info_layers(dynstr, &me_eval->pdata);
//  //BLI_dynstr_append(dynstr, "    ),\n");
//
//  //BLI_dynstr_append(dynstr, "    'tessFaceLayers': (\n");
//  //mesh_runtime_debug_info_layers(dynstr, &me_eval->fdata);
//  //BLI_dynstr_append(dynstr, "    ),\n");
//
//  BLI_dynstr_append(dynstr, "}\n");
//
//  ret = BLI_dynstr_get_cstring(dynstr);
//  BLI_dynstr_free(dynstr);
//  return ret;
//}

//void BKE_mesh_runtime_debug_print(Mesh *me_eval)
//{
//  char *str = BKE_mesh_runtime_debug_info(me_eval);
//  puts(str);
//  fflush(stdout);
//  MEM_freeN(str);
//}

/* XXX Should go in customdata file? */
//void BKE_mesh_runtime_debug_print_cdlayers(CustomData *data)
//{
//  int i;
//  const CustomDataLayer *layer;
//
//  printf("{\n");
//
//  for (i = 0, layer = data->layers; i < data->totlayer; i++, layer++) {
//
//    const char *name = CustomData_layertype_name(layer->type);
//    const int size = CustomData_sizeof(layer->type);
//    const char *structname;
//    int structnum;
//    CustomData_file_write_info(layer->type, &structname, &structnum);
//    printf("        dict(name='%s', struct='%s', type=%d, ptr='%p', elem=%d, length=%d),\n",
//           name,
//           structname,
//           layer->type,
//           (const void *)layer->data,
//           size,
//           (int)(MEM_allocN_len(layer->data) / size));
//  }
//
//  printf("}\n");
//}

bool BKE_mesh_runtime_is_valid(Mesh *me_eval)
{
  const bool do_verbose = true;
  const bool do_fixes = false;

  bool is_valid = true;
  bool changed = true;

  //if (do_verbose)
  //{
  //  printf("MESH: %s\n", me_eval->id.name + 2);
  //}

  //is_valid &= BKE_mesh_validate_all_customdata(
  //    &me_eval->vdata,
  //    me_eval->totvert,
  //    &me_eval->edata,
  //    me_eval->totedge,
  //    &me_eval->ldata,
  //    me_eval->totloop,
  //    &me_eval->pdata,
  //    me_eval->totpoly,
  //    false, /* setting mask here isn't useful, gives false positives */
  //    do_verbose,
  //    do_fixes,
  //    &changed);

  //is_valid &= BKE_mesh_validate_arrays(me_eval,
  //                                     me_eval->mvert,
  //                                     me_eval->totvert,
  //                                     me_eval->medge,
  //                                     me_eval->totedge,
  //                                     me_eval->mface,
  //                                     me_eval->totface,
  //                                     me_eval->mloop,
  //                                     me_eval->totloop,
  //                                     me_eval->mpoly,
  //                                     me_eval->totpoly,
  //                                     me_eval->dvert,
  //                                     do_verbose,
  //                                     do_fixes,
  //                                     &changed);

  BLI_assert(changed == false);

  return is_valid;
}

#endif /* NDEBUG */

/** \} */
