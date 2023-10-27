#include "atomic_ops.cuh"

#include "MEM_guardedalloc.cuh"

#include "mesh_types.h"
#include "meshdata_types.cuh"
#include "object_types.cuh"

#include "math_geom.cuh"
#include "task.cuh"
//#include "BLI_timeit.hh"

#include "bvhutils.h"
//#include "BKE_editmesh_cache.h"
#include "BKE_lib_id.h"
#include "BKE_mesh.h"
#include "mesh_runtime.h"
//#include "BKE_shrinkwrap.h"
#include "BKE_subdiv_ccg.h"
#include "BLI_span.hh"
#include "mallocn_intern.cuh"

using blender::Span;

/* -------------------------------------------------------------------- */
/** \name Mesh Runtime Struct Utils
 * \{ */

namespace blender::bke {

    static void edit_data_reset(EditMeshData& edit_data)
    {
        MEM_SAFE_FREE(edit_data.polyCos);
        MEM_SAFE_FREE(edit_data.polyNos);
        MEM_SAFE_FREE(edit_data.vertexCos);
        MEM_SAFE_FREE(edit_data.vertexNos);
    }

    static void free_edit_data(MeshRuntime& mesh_runtime)
    {
        if (mesh_runtime.edit_data) {
            edit_data_reset(*mesh_runtime.edit_data);
            MEM_lockfree_freeN(mesh_runtime.edit_data);
            mesh_runtime.edit_data = nullptr;
        }
    }

    static void free_mesh_eval(MeshRuntime& mesh_runtime)
    {
        if (mesh_runtime.mesh_eval != nullptr) {
            //mesh_runtime.mesh_eval->edit_mesh = nullptr;
            BKE_id_free(nullptr, mesh_runtime.mesh_eval);
            mesh_runtime.mesh_eval = nullptr;
        }
    }

    static void free_subdiv_ccg(MeshRuntime& mesh_runtime)
    {
        /* TODO(sergey): Does this really belong here? */
        if (mesh_runtime.subdiv_ccg != nullptr) {
            BKE_subdiv_ccg_destroy(mesh_runtime.subdiv_ccg);
            mesh_runtime.subdiv_ccg = nullptr;
        }
    }

    static void free_bvh_cache(MeshRuntime& mesh_runtime)
    {
        if (mesh_runtime.bvh_cache) {
            bvhcache_free(mesh_runtime.bvh_cache);
            mesh_runtime.bvh_cache = nullptr;
        }
    }

    static void free_normals(MeshRuntime& mesh_runtime)
    {
        MEM_SAFE_FREE(mesh_runtime.vert_normals);
        MEM_SAFE_FREE(mesh_runtime.poly_normals);
        mesh_runtime.vert_normals_dirty = true;
        mesh_runtime.poly_normals_dirty = true;
    }

    static void free_batch_cache(MeshRuntime& mesh_runtime)
    {
        if (mesh_runtime.batch_cache) 
        {
            BKE_mesh_batch_cache_free(mesh_runtime.batch_cache);
            mesh_runtime.batch_cache = nullptr;
        }
    }

}  // namespace blender::bke

MeshRuntime::~MeshRuntime()
{
    blender::bke::free_mesh_eval(*this);
    blender::bke::free_subdiv_ccg(*this);
    blender::bke::free_bvh_cache(*this);
    blender::bke::free_edit_data(*this);
    blender::bke::free_batch_cache(*this);
    blender::bke::free_normals(*this);
    //if (this->shrinkwrap_data) 
    //{
    //    BKE_shrinkwrap_boundary_data_free(this->shrinkwrap_data);
    //}
}
//
//const blender::bke::LooseEdgeCache& Mesh::loose_edges() const
//{
//    using namespace blender::bke;
//    this->runtime->loose_edges_cache.ensure([&](LooseEdgeCache& r_data) {
//        blender::BitVector<>& loose_edges = r_data.is_loose_bits;
//        loose_edges.resize(0);
//        loose_edges.resize(this->totedge, true);
//
//        int count = this->totedge;
//        for (const MLoop& loop : this->loops()) {
//            if (loose_edges[loop.e]) {
//                loose_edges[loop.e].reset();
//                count--;
//            }
//        }
//
//        r_data.count = count;
//        });
//
//    return this->runtime->loose_edges_cache.data();
//}
//
//void Mesh::loose_edges_tag_none() const
//{
//    using namespace blender::bke;
//    this->runtime->loose_edges_cache.ensure([&](LooseEdgeCache& r_data) {
//        r_data.is_loose_bits.resize(0);
//        r_data.count = 0;
//        });
//}
//
//blender::Span<MLoopTri> Mesh::looptris() const
//{
//    this->runtime->looptris_cache.ensure([&](blender::Array<MLoopTri>& r_data) {
//        const Span<float3> positions = this->vert_positions();
//        const Span<MPoly> polys = this->polys();
//        const Span<MLoop> loops = this->loops();
//
//        r_data.reinitialize(poly_to_tri_count(polys.size(), loops.size()));
//
//        if (BKE_mesh_poly_normals_are_dirty(this)) {
//            BKE_mesh_recalc_looptri(loops.data(),
//                polys.data(),
//                reinterpret_cast<const float(*)[3]>(positions.data()),
//                loops.size(),
//                polys.size(),
//                r_data.data());
//        }
//        else {
//            BKE_mesh_recalc_looptri_with_normals(loops.data(),
//                polys.data(),
//                reinterpret_cast<const float(*)[3]>(positions.data()),
//                loops.size(),
//                polys.size(),
//                r_data.data(),
//                BKE_mesh_poly_normals_ensure(this));
//        }
//        });
//
//    return this->runtime->looptris_cache.data();
//}

int BKE_mesh_runtime_looptri_len(const Mesh* mesh)
{
    /* Allow returning the size without calculating the cache. */
    return poly_to_tri_count(mesh->totpoly, mesh->totloop);
}

//const MLoopTri* BKE_mesh_runtime_looptri_ensure(const Mesh* mesh)
//{
//    return mesh->looptris().data();
//}

void BKE_mesh_runtime_verttri_from_looptri(MVertTri* r_verttri,
                                            const MLoop* mloop,
                                            const MLoopTri* looptri,
                                            int looptri_num)
{
    for (int i = 0; i < looptri_num; i++) 
    {
        r_verttri[i].tri[0] = mloop[looptri[i].tri[0]].v;
        r_verttri[i].tri[1] = mloop[looptri[i].tri[1]].v;
        r_verttri[i].tri[2] = mloop[looptri[i].tri[2]].v;
    }
}

bool BKE_mesh_runtime_ensure_edit_data(struct Mesh* mesh)
{
    if (mesh->runtime->edit_data != nullptr) {
        return false;
    }
    mesh->runtime->edit_data = MEM_cnew<EditMeshData>(__func__);
    return true;
}

//void BKE_mesh_runtime_reset_edit_data(Mesh* mesh)
//{
//    using namespace blender::bke;
//    if (EditMeshData* edit_data = mesh->runtime->edit_data) {
//        edit_data_reset(*edit_data);
//    }
//}

void BKE_mesh_runtime_clear_cache(Mesh* mesh)
{
    using namespace blender::bke;
    free_mesh_eval(*mesh->runtime);
    free_batch_cache(*mesh->runtime);
    free_edit_data(*mesh->runtime);
    BKE_mesh_runtime_clear_geometry(mesh);
}

void BKE_mesh_runtime_clear_geometry(Mesh* mesh)
{
    using namespace blender::bke;
    /* Tagging shared caches dirty will free the allocated data if there is only one user. */
    free_bvh_cache(*mesh->runtime);
    free_normals(*mesh->runtime);
    free_subdiv_ccg(*mesh->runtime);
    //mesh->runtime->bounds_cache.tag_dirty();
    //mesh->runtime->loose_edges_cache.tag_dirty();
    //mesh->runtime->looptris_cache.tag_dirty();
    //mesh->runtime->subsurf_face_dot_tags.clear_and_shrink();
    //mesh->runtime->subsurf_optimal_display_edges.clear_and_shrink();
    //if (mesh->runtime->shrinkwrap_data) 
    //{
    //    BKE_shrinkwrap_boundary_data_free(mesh->runtime->shrinkwrap_data);
    //}
}

void BKE_mesh_tag_edges_split(struct Mesh* mesh)
{
    using namespace blender::bke;
    /* Triangulation didn't change because vertex positions and loop vertex indices didn't change.
     * Face normals didn't change either, but tag those anyway, since there is no API function to
     * only tag vertex normals dirty. */
    free_bvh_cache(*mesh->runtime);
    free_normals(*mesh->runtime);
    free_subdiv_ccg(*mesh->runtime);
    //mesh->runtime->loose_edges_cache.tag_dirty();
    //mesh->runtime->subsurf_face_dot_tags.clear_and_shrink();
    //mesh->runtime->subsurf_optimal_display_edges.clear_and_shrink();
    //if (mesh->runtime->shrinkwrap_data) {
    //    BKE_shrinkwrap_boundary_data_free(mesh->runtime->shrinkwrap_data);
    //}
}

void BKE_mesh_tag_coords_changed(Mesh* mesh)
{
    using namespace blender::bke;
    //BKE_mesh_normals_tag_dirty(mesh);
    free_bvh_cache(*mesh->runtime);
    //mesh->runtime->looptris_cache.tag_dirty();
    //mesh->runtime->bounds_cache.tag_dirty();
}

void BKE_mesh_tag_coords_changed_uniformly(Mesh* mesh)
{
    using namespace blender::bke;
    /* The normals and triangulation didn't change, since all verts moved by the same amount. */
    free_bvh_cache(*mesh->runtime);
    //mesh->runtime->bounds_cache.tag_dirty();
}

void BKE_mesh_tag_topology_changed(struct Mesh* mesh)
{
    BKE_mesh_runtime_clear_geometry(mesh);
}

bool BKE_mesh_is_deformed_only(const Mesh* mesh)
{
    return mesh->runtime->deformed_only;
}

eMeshWrapperType BKE_mesh_wrapper_type(const struct Mesh* mesh)
{
    return mesh->runtime->wrapper_type;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Mesh Batch Cache Callbacks
 * \{ */

 /* Draw Engine */
void (*BKE_mesh_batch_cache_dirty_tag_cb)(Mesh* me, eMeshBatchDirtyMode mode) = nullptr;
void (*BKE_mesh_batch_cache_free_cb)(void* batch_cache) = nullptr;

void BKE_mesh_batch_cache_dirty_tag(Mesh* me, eMeshBatchDirtyMode mode)
{
    if (me->runtime->batch_cache) {
        BKE_mesh_batch_cache_dirty_tag_cb(me, mode);
    }
}
void BKE_mesh_batch_cache_free(void* batch_cache)
{
    BKE_mesh_batch_cache_free_cb(batch_cache);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Mesh Runtime Validation
 * \{ */

#ifndef NDEBUG

#include "BLI_dynstr.h"

static void mesh_debug_info_from_cd_flag(const Mesh* me, DynStr* dynstr)
{
    BLI_dynstr_append(dynstr, "'cd_flag': {");
    if (me->cd_flag & ME_CDFLAG_VERT_BWEIGHT) 
    {
        BLI_dynstr_append(dynstr, "'VERT_BWEIGHT', ");
    }
    if (me->cd_flag & ME_CDFLAG_EDGE_BWEIGHT) 
    {
        BLI_dynstr_append(dynstr, "'EDGE_BWEIGHT', ");
    }
    if (me->cd_flag & ME_CDFLAG_EDGE_CREASE) 
    {
        BLI_dynstr_append(dynstr, "'EDGE_CREASE', ");
    }
    BLI_dynstr_append(dynstr, "},\n");
}

void CustomData_debug_info_from_layers(const CustomData* data, const char* indent, DynStr* dynstr)
{
    for (int type = 0; type < CD_NUMTYPES; type++) 
    {
        if (data->layers && CustomData_has_layer(data, type))
        {
            /* NOTE: doesn't account for multiple layers. */
            const char* name = CustomData_layertype_name(type);
            const int size = CustomData_sizeof(type);
            const void* pt = CustomData_get_layer(data, type);
            const int pt_size = pt ? int(MEM_lockfree_allocN_len(pt) / size) : 0;
            const char* structname;
            int structnum;
            CustomData_file_write_info(type, &structname, &structnum);
            BLI_dynstr_appendf(
                dynstr,
                "%sdict(name='%s', struct='%s', type=%d, ptr='%p', elem=%d, length=%d),\n",
                indent,
                name,
                structname,
                type,
                (const void*)pt,
                size,
                pt_size);
        }
    }
}

char* BKE_mesh_debug_info(const Mesh* me)
{
    DynStr* dynstr = BLI_dynstr_new();
    char* ret;

    const char* indent4 = "    ";
    const char* indent8 = "        ";

    BLI_dynstr_append(dynstr, "{\n");
    BLI_dynstr_appendf(dynstr, "    'ptr': '%p',\n", (void*)me);
    BLI_dynstr_appendf(dynstr, "    'totvert': %d,\n", me->totvert);
    BLI_dynstr_appendf(dynstr, "    'totedge': %d,\n", me->totedge);
    BLI_dynstr_appendf(dynstr, "    'totface': %d,\n", me->totface);
    BLI_dynstr_appendf(dynstr, "    'totpoly': %d,\n", me->totpoly);

    //BLI_dynstr_appendf(dynstr, "    'runtime.deformed_only': %d,\n", me->runtime->deformed_only);
    //BLI_dynstr_appendf(dynstr, "    'runtime.is_original': %d,\n", me->runtime->is_original);

    BLI_dynstr_append(dynstr, "    'vert_layers': (\n");
    CustomData_debug_info_from_layers(&me->vdata, indent8, dynstr);
    BLI_dynstr_append(dynstr, "    ),\n");

    BLI_dynstr_append(dynstr, "    'edge_layers': (\n");
    CustomData_debug_info_from_layers(&me->edata, indent8, dynstr);
    BLI_dynstr_append(dynstr, "    ),\n");

    BLI_dynstr_append(dynstr, "    'loop_layers': (\n");
    CustomData_debug_info_from_layers(&me->ldata, indent8, dynstr);
    BLI_dynstr_append(dynstr, "    ),\n");

    BLI_dynstr_append(dynstr, "    'poly_layers': (\n");
    CustomData_debug_info_from_layers(&me->pdata, indent8, dynstr);
    BLI_dynstr_append(dynstr, "    ),\n");

    BLI_dynstr_append(dynstr, "    'tessface_layers': (\n");
    CustomData_debug_info_from_layers(&me->fdata, indent8, dynstr);
    BLI_dynstr_append(dynstr, "    ),\n");

    BLI_dynstr_append(dynstr, indent4);
    mesh_debug_info_from_cd_flag(me, dynstr);

    BLI_dynstr_append(dynstr, "}\n");

    ret = BLI_dynstr_get_cstring(dynstr);
    printf(ret);
    BLI_dynstr_free(dynstr);
    return ret;
}

void BKE_mesh_debug_print(const Mesh* me)
{
    char* str = BKE_mesh_debug_info(me);
    puts(str);
    fflush(stdout);
    MEM_lockfree_freeN(str);
}

bool BKE_mesh_runtime_is_valid(Mesh* me_eval)
{
    const bool do_verbose = true;
    const bool do_fixes = false;

    bool is_valid = true;
    bool changed = true;

    if (do_verbose) {
        printf("MESH: %s\n", me_eval->id.name + 2);
    }

    //MutableSpan<float3> positions = me_eval->vert_positions_for_write();
    //MutableSpan<MEdge> edges = me_eval->edges_for_write();
    //MutableSpan<MPoly> polys = me_eval->polys_for_write();
    //MutableSpan<MLoop> loops = me_eval->loops_for_write();

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
    //    reinterpret_cast<float(*)[3]>(positions.data()),
    //    positions.size(),
    //    edges.data(),
    //    edges.size(),
    //    //static_cast<MFace*>(CustomData_get_layer_for_write(&me_eval->fdata, CD_MFACE, me_eval->totface)),
    //    me_eval->totface,
    //    loops.data(),
    //    loops.size(),
    //    polys.data(),
    //    polys.size(),
    //    //me_eval->deform_verts_for_write().data(),
    //    do_verbose,
    //    do_fixes,
    //    &changed);

    BLI_assert(changed == false);

    return is_valid;
}

#endif /* NDEBUG */

/** \} */
