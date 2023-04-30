#include "MEM_guardedalloc.cuh"         

#include "cloth_types.cuh"              
#include "meshdata_types.cuh"           
#include "modifier_types.cuh"           
#include "object_force_types.cuh"       
#include "object_types.cuh"             
#include "scene_types.cuh"              

#include "linklist.cuh"                 
#include "B_math.h"                      
#include "utildefines.h"                 

#include "cloth.h"                    
#include "collision.h"                  
#include "effect.h"                      

#include "DEG_depsgraph.h"               
#include "DEG_depsgraph_query.h"         

#include "object_force_types.cuh"        
#include "implicit.h"                   
#include "cloth_settings.h"           
#include "SIM_mass_spring.h"          

/* ��� ���� ���������� ��������:
* ������ ����.
*       ���������� �������� ������ �������, ��� ������� ��� ����� ������� � ������ �����(�������)
* ������ ����.
*       ������������ �������, ����� ��������� ������������ �����������.
* ������ ����.
*       ������������� �������� ������ � ������ ������� �����.
*/


/* Number of off-diagonal non-zero matrix blocks.
 * Basically there is one of these for each vertex-vertex interaction.
 */
static int cloth_count_nondiag_blocks(Cloth* cloth)
{
    LinkNode* link;
    int nondiag = 0;

    for (link = cloth->springs; link; link = link->next) {
        ClothSpring* spring = (ClothSpring*)link->link;
        switch (spring->type) {
        case CLOTH_SPRING_TYPE_BENDING_HAIR:
            /* angular bending combines 3 vertices */
            nondiag += 3;
            break;

        default:
            /* all other springs depend on 2 vertices only */
            nondiag += 1;
            break;
        }
    }

    return nondiag;
}

static bool cloth_get_pressure_weights(ClothModifierData* clmd,
    const MVertTri* vt,
    float* r_weights)
{
    /* We have custom vertex weights for pressure. */
    if (clmd->sim_parms->vgroup_pressure > 0) {
        Cloth* cloth = clmd->clothObject;
        ClothVertex* verts = cloth->verts;

        for (uint j = 0; j < 3; j++) {
            r_weights[j] = verts[vt->tri[j]].pressure_factor;

            /* Skip the entire triangle if it has a zero weight. */
            if (r_weights[j] == 0.0f) {
                return false;
            }
        }
    }

    return true;
}

static void cloth_calc_pressure_gradient(ClothModifierData* clmd,
    const float gradient_vector[3],
    float* r_vertex_pressure)
{
    Cloth* cloth = clmd->clothObject;
    Implicit_Data* data = cloth->implicit;
    uint mvert_num = cloth->mvert_num;
    float pt[3];

    for (uint i = 0; i < mvert_num; i++) {
        SIM_mass_spring_get_position(data, i, pt);
        r_vertex_pressure[i] = dot_v3v3(pt, gradient_vector);
    }
}

static float cloth_calc_volume(ClothModifierData* clmd)
{
    /* Calculate the (closed) cloth volume. */
    Cloth* cloth = clmd->clothObject;
    const MVertTri* tri = cloth->tri;
    Implicit_Data* data = cloth->implicit;
    float weights[3] = { 1.0f, 1.0f, 1.0f };
    float vol = 0;

    for (uint i = 0; i < cloth->primitive_num; i++) {
        const MVertTri* vt = &tri[i];

        if (cloth_get_pressure_weights(clmd, vt, weights)) {
            vol += SIM_tri_tetra_volume_signed_6x(data, vt->tri[0], vt->tri[1], vt->tri[2]);
        }
    }

    /* We need to divide by 6 to get the actual volume. */
    vol = vol / 6.0f;

    return vol;
}

static float cloth_calc_rest_volume(ClothModifierData* clmd)
{
    /* Calculate the (closed) cloth volume. */
    Cloth* cloth = clmd->clothObject;
    const MVertTri* tri = cloth->tri;
    const ClothVertex* v = cloth->verts;
    float weights[3] = { 1.0f, 1.0f, 1.0f };
    float vol = 0;

    for (uint i = 0; i < cloth->primitive_num; i++) {
        const MVertTri* vt = &tri[i];

        if (cloth_get_pressure_weights(clmd, vt, weights)) {
            vol += volume_tri_tetrahedron_signed_v3_6x(
                v[vt->tri[0]].xrest, v[vt->tri[1]].xrest, v[vt->tri[2]].xrest);
        }
    }

    /* We need to divide by 6 to get the actual volume. */
    vol = vol / 6.0f;

    return vol;
}

static float cloth_calc_average_pressure(ClothModifierData* clmd, const float* vertex_pressure)
{
    Cloth* cloth = clmd->clothObject;
    const MVertTri* tri = cloth->tri;
    Implicit_Data* data = cloth->implicit;
    float weights[3] = { 1.0f, 1.0f, 1.0f };
    float total_force = 0;
    float total_area = 0;

    for (uint i = 0; i < cloth->primitive_num; i++) {
        const MVertTri* vt = &tri[i];

        if (cloth_get_pressure_weights(clmd, vt, weights)) {
            float area = SIM_tri_area(data, vt->tri[0], vt->tri[1], vt->tri[2]);

            total_force += (vertex_pressure[vt->tri[0]] + vertex_pressure[vt->tri[1]] +
                vertex_pressure[vt->tri[2]]) *
                area / 3.0f;
            total_area += area;
        }
    }

    return total_force / total_area;
}

int SIM_cloth_solver_init(ClothModifierData* clmd)
{
    Cloth* cloth = clmd->clothObject;
    ClothVertex* verts = cloth->verts;
    const float ZERO[3] = { 0.0f, 0.0f, 0.0f };
    Implicit_Data* id;
    uint i, nondiag;

    nondiag = cloth_count_nondiag_blocks(cloth);
    cloth->implicit = id = SIM_mass_spring_solver_create(cloth->mvert_num, nondiag);

    for (i = 0; i < cloth->mvert_num; i++) {
        SIM_mass_spring_set_vertex_mass(id, i, verts[i].mass);
    }

    for (i = 0; i < cloth->mvert_num; i++) {
        SIM_mass_spring_set_motion_state(id, i, verts[i].x, ZERO);
    }

    return 1;
}

void SIM_cloth_solver_free(ClothModifierData* clmd)
{
    Cloth* cloth = clmd->clothObject;

    if (cloth->implicit) {
        SIM_mass_spring_solver_free(cloth->implicit);
        cloth->implicit = nullptr;
    }
}

void SIM_cloth_solver_set_positions(ClothModifierData* clmd)
{
    Cloth* cloth = clmd->clothObject;
    ClothVertex* verts = cloth->verts;
    uint mvert_num = cloth->mvert_num, i;
    Implicit_Data* id = cloth->implicit;

    for (i = 0; i < mvert_num; i++) 
    {
        SIM_mass_spring_set_rest_transform(id, i, I3);
        SIM_mass_spring_set_motion_state(id, i, verts[i].x, verts[i].v);
    }
}

void SIM_cloth_solver_set_volume(ClothModifierData* clmd)
{
    Cloth* cloth = clmd->clothObject;

    cloth->initial_mesh_volume = cloth_calc_rest_volume(clmd);
}

/* Init constraint matrix
 * This is part of the modified CG method suggested by Baraff/Witkin in
 * "Large Steps in Cloth Simulation" (Siggraph 1998)
 */
static void cloth_setup_constraints(ClothModifierData* clmd)
{
    Cloth* cloth = clmd->clothObject;
    Implicit_Data* data = cloth->implicit;
    ClothVertex* verts = cloth->verts;
    int mvert_num = cloth->mvert_num;
    int v;

    const float ZERO[3] = { 0.0f, 0.0f, 0.0f };

    SIM_mass_spring_clear_constraints(data);

    for (v = 0; v < mvert_num; v++) {
        if (verts[v].flags & CLOTH_VERT_FLAG_PINNED) {
            /* pinned vertex constraints */
            SIM_mass_spring_add_constraint_ndof0(data, v, ZERO); /* velocity is defined externally */
        }

        verts[v].impulse_count = 0;
    }
}

 void cloth_calc_spring_force(ClothModifierData* clmd, ClothSpring* s)
{
    Cloth* cloth = clmd->clothObject;
    ClothSimSettings* parms = clmd->sim_parms;
    Implicit_Data* data = cloth->implicit;
    bool using_angular = parms->bending_model == CLOTH_BENDING_ANGULAR;
    bool resist_compress = (parms->flags & CLOTH_SIMSETTINGS_FLAG_RESIST_SPRING_COMPRESS) &&
        !using_angular;

    s->flags &= ~CLOTH_SPRING_FLAG_NEEDED;

    /* Calculate force of bending springs. */
    if ((s->type & CLOTH_SPRING_TYPE_BENDING) && using_angular) {
#ifdef CLOTH_FORCE_SPRING_BEND
        float k, scaling;

        s->flags |= CLOTH_SPRING_FLAG_NEEDED;

        scaling = parms->bending + s->ang_stiffness * fabsf(parms->max_bend - parms->bending);
        k = scaling * s->restlen *
            0.1f; /* Multiplying by 0.1, just to scale the forces to more reasonable values. */

        SIM_mass_spring_force_spring_angular(
            data, s->ij, s->kl, s->pa, s->pb, s->la, s->lb, s->restang, k, parms->bending_damping);
#endif
    }

    /* Calculate force of structural + shear springs. */
    if (s->type &
        (CLOTH_SPRING_TYPE_STRUCTURAL | CLOTH_SPRING_TYPE_SEWING | CLOTH_SPRING_TYPE_INTERNAL)) {
#ifdef CLOTH_FORCE_SPRING_STRUCTURAL
        float k_tension, scaling_tension;

        s->flags |= CLOTH_SPRING_FLAG_NEEDED;

        scaling_tension = parms->tension +
            s->lin_stiffness * fabsf(parms->max_tension - parms->tension);
        k_tension = scaling_tension / (parms->avg_spring_len + FLT_EPSILON);

        if (s->type & CLOTH_SPRING_TYPE_SEWING) {
            /* TODO: verify, half verified (couldn't see error)
             * sewing springs usually have a large distance at first so clamp the force so we don't get
             * tunneling through collision objects. */
            SIM_mass_spring_force_spring_linear(data,
                s->ij,
                s->kl,
                s->restlen,
                k_tension,
                parms->tension_damp,
                0.0f,
                0.0f,
                false,
                false,
                parms->max_sewing);
        }
        else if (s->type & CLOTH_SPRING_TYPE_STRUCTURAL) {
            float k_compression, scaling_compression;
            scaling_compression = parms->compression +
                s->lin_stiffness * fabsf(parms->max_compression - parms->compression);
            k_compression = scaling_compression / (parms->avg_spring_len + FLT_EPSILON);

            SIM_mass_spring_force_spring_linear(data,
                s->ij,
                s->kl,
                s->restlen,
                k_tension,
                parms->tension_damp,
                k_compression,
                parms->compression_damp,
                resist_compress,
                using_angular,
                0.0f);
        }
        else {
            /* CLOTH_SPRING_TYPE_INTERNAL */
            BLI_assert(s->type & CLOTH_SPRING_TYPE_INTERNAL);

            scaling_tension = parms->internal_tension +
                s->lin_stiffness *
                fabsf(parms->max_internal_tension - parms->internal_tension);
            k_tension = scaling_tension / (parms->avg_spring_len + FLT_EPSILON);
            float scaling_compression = parms->internal_compression +
                s->lin_stiffness * fabsf(parms->max_internal_compression -
                    parms->internal_compression);
            float k_compression = scaling_compression / (parms->avg_spring_len + FLT_EPSILON);

            float k_tension_damp = parms->tension_damp;
            float k_compression_damp = parms->compression_damp;

            if (k_tension == 0.0f) {
                /* No damping so it behaves as if no tension spring was there at all. */
                k_tension_damp = 0.0f;
            }

            if (k_compression == 0.0f) {
                /* No damping so it behaves as if no compression spring was there at all. */
                k_compression_damp = 0.0f;
            }

            SIM_mass_spring_force_spring_linear(data,
                s->ij,
                s->kl,
                s->restlen,
                k_tension,
                k_tension_damp,
                k_compression,
                k_compression_damp,
                resist_compress,
                using_angular,
                0.0f);
        }
#endif
    }
    else if (s->type & CLOTH_SPRING_TYPE_SHEAR) {
#ifdef CLOTH_FORCE_SPRING_SHEAR
        float k, scaling;

        s->flags |= CLOTH_SPRING_FLAG_NEEDED;

        scaling = parms->shear + s->lin_stiffness * fabsf(parms->max_shear - parms->shear);
        k = scaling / (parms->avg_spring_len + FLT_EPSILON);

        SIM_mass_spring_force_spring_linear(data,
            s->ij,
            s->kl,
            s->restlen,
            k,
            parms->shear_damp,
            0.0f,
            0.0f,
            resist_compress,
            false,
            0.0f);
#endif
    }
    else if (s->type & CLOTH_SPRING_TYPE_BENDING) { /* calculate force of bending springs */
#ifdef CLOTH_FORCE_SPRING_BEND
        float kb, cb, scaling;

        s->flags |= CLOTH_SPRING_FLAG_NEEDED;

        scaling = parms->bending + s->lin_stiffness * fabsf(parms->max_bend - parms->bending);
        kb = scaling / (20.0f * (parms->avg_spring_len + FLT_EPSILON));

        /* Fix for T45084 for cloth stiffness must have cb proportional to kb */
        cb = kb * parms->bending_damping;

        SIM_mass_spring_force_spring_bending(data, s->ij, s->kl, s->restlen, kb, cb);
#endif
    }
}

/// <summary>
/// ������������ ����, ����������� �� ����� �����.
/// </summary>
/// <param name="scene">��������� �� ������� �����.</param>
/// <param name="clmd">��������� �� ����������� ������ �����.</param>
/// <param name="effectors">������ ����������, �������� �� �����.</param>
/// <param name="time">������� ����� ���������.</param>
static void cloth_calc_force(Scene* scene, ClothModifierData* clmd, ListBase* effectors, float time)
{
    // ���� ��� � �����������: F, dFdX, dFdV
    Cloth* cloth = clmd->clothObject;
    ClothSimSettings* parms = clmd->sim_parms;
    Implicit_Data* data = cloth->implicit;
    uint i = 0;
    float drag = clmd->sim_parms->Cvi * 0.01f; // �������� ������� � ���������
    float gravity[3] = { 0.0f, 0.0f, 0.0f };
    const MVertTri* tri = cloth->tri;
    uint mvert_num = cloth->mvert_num;
    ClothVertex* vert;

#ifdef CLOTH_FORCE_GRAVITY
    // ���������� ��������� (����������)
    if (scene->physics_settings.flag & PHYS_GLOBAL_GRAVITY)
    {
        // ��������������� ���� ����������
        mul_v3_v3fl(gravity, scene->physics_settings.gravity, 0.001f * clmd->sim_parms->effector_weights->global_gravity);
    }

    vert = cloth->verts;
    for (i = 0; i < cloth->mvert_num; i++, vert++)
    {
        SIM_mass_spring_force_gravity(data, i, vert->mass, gravity);

        // ������������ ������� �������
        if ((!(vert->flags & CLOTH_VERT_FLAG_PINNED)) && (vert->goal > FLT_EPSILON))
        {
            float goal_x[3], goal_v[3];
            float k;

            // ���������� �� time_scale ��� �������������� ��������� delta ��������� ������� ������
            interp_v3_v3v3(goal_x, vert->xold, vert->xconst, time / clmd->sim_parms->time_scale);
            sub_v3_v3v3(goal_v, vert->xconst, vert->xold); // ����������, ���������� �� dt==1

            k = vert->goal * clmd->sim_parms->goalspring / (clmd->sim_parms->avg_spring_len + FLT_EPSILON);

            SIM_mass_spring_force_spring_goal(data, i, goal_x, goal_v, k, clmd->sim_parms->goalfrict * 0.01f);
        }
    }
#endif

#ifdef CLOTH_FORCE_DRAG
    SIM_mass_spring_force_drag(data, drag);
#endif

    // ��������� ��� �������� (���������, ��� ��� ������� �� ����������� ��� �����).
    if ((parms->flags & CLOTH_SIMSETTINGS_FLAG_PRESSURE))
    {
        // ������� � �������� ����� ���������� � ������� ������� �����.
        float pressure_difference = 0.0f;
        float volume_factor = 1.0f;

        float init_vol;
        if (parms->flags & CLOTH_SIMSETTINGS_FLAG_PRESSURE_VOL)
        {
            init_vol = clmd->sim_parms->target_volume;
        }
        else
        {
            init_vol = cloth->initial_mesh_volume;
        }

        // �������� ������������� ���������� ������ �����.
        if (init_vol > 1E-6f) {
            float f;
            float vol = cloth_calc_volume(clmd);

            // ���� ����� ����� ��, �� ���������� ��������.
            volume_factor = init_vol / vol;
            pressure_difference = volume_factor - 1;

            // ���������� �������������� ������������� �������� ��� �������� �����.
            f = fabs(clmd->sim_parms->uniform_pressure_force) + 200.0f;

            // ����� �������� ����� �� ������������ ������������� ��������.
            CLAMP_MAX(pressure_difference, f);
        }

        pressure_difference += clmd->sim_parms->uniform_pressure_force;
        pressure_difference *= clmd->sim_parms->pressure_factor;

        // ���������� ��������� ����������������� ��������, ���� ��������.
        float fluid_density = clmd->sim_parms->fluid_density * 1000; // ��/� -> ��/�3
        float* hydrostatic_pressure = nullptr;

        if (fabs(fluid_density) > 1e-6f)
        {
            float hydrostatic_vector[3];
            copy_v3_v3(hydrostatic_vector, gravity);

            // ����� �������� ��������� ������ �������, ��������� ���������
            // ������� � ���� ��������, ��� ��� ���������� ����������
            // �� ��������� �������.
            if (fluid_density > 0)
            {
                sub_v3_v3(hydrostatic_vector, cloth->average_acceleration);

                // ���������� ����� ����� ����� ��������������� ��������� ��� ������������ ��������� ������.
                fluid_density *= volume_factor;
            }

            mul_v3_fl(hydrostatic_vector, fluid_density);

            // ���������� ������� ����������������� �������� ��� ������ ������� � ��������� �������� ��������.
            hydrostatic_pressure = (float*)MEM_mallocN(sizeof(float) * mvert_num, "hydrostatic pressure gradient");

            cloth_calc_pressure_gradient(clmd, hydrostatic_vector, hydrostatic_pressure);

            pressure_difference -= cloth_calc_average_pressure(clmd, hydrostatic_pressure);
        }

        // ���������� ��������.
        if (hydrostatic_pressure || fabs(pressure_difference) > 1E-6f)
        {
            float weights[3] = { 1.0f, 1.0f, 1.0f };

            for (i = 0; i < cloth->primitive_num; i++)
            {
                const MVertTri* vt = &tri[i];

                if (cloth_get_pressure_weights(clmd, vt, weights))
                {
                    SIM_mass_spring_force_pressure(data, vt->tri[0], vt->tri[1], vt->tri[2], pressure_difference, hydrostatic_pressure, weights);
                }
            }
        }

        if (hydrostatic_pressure)
        {
            MEM_freeN(hydrostatic_pressure);
        }
    }

    // ��������� ������� ���, ����� ��� �����.
    if (effectors)
    {
        bool has_wind = false, has_force = false;

        // ����������� ��� ��� ������ �������, ����� �������� ���������� ����������.
        float(*winvec)[3] = (float(*)[3])MEM_callocN(sizeof(float[3]) * mvert_num * 2, "���� ����������");
        float(*forcevec)[3] = winvec + mvert_num;

        for (i = 0; i < cloth->mvert_num; i++)
        {
            float x[3], v[3];
            EffectedPoint epoint;

            SIM_mass_spring_get_motion_state(data, i, x, v);
            pd_point_from_loc(scene, x, v, i, &epoint);
            effectors_apply(effectors, nullptr, clmd->sim_parms->effector_weights, &epoint, forcevec[i], winvec[i], nullptr);

            has_wind = has_wind || !is_zero_v3(winvec[i]);
            has_force = has_force || !is_zero_v3(forcevec[i]);
        }

        for (i = 0; i < cloth->primitive_num; i++)
        {
            const MVertTri* vt = &tri[i];
            if (has_wind)
            {
                SIM_mass_spring_force_face_wind(data, vt->tri[0], vt->tri[1], vt->tri[2], winvec);
            }
            if (has_force)
            {
                SIM_mass_spring_force_face_extern(data, vt->tri[0], vt->tri[1], vt->tri[2], forcevec);
            }
        }
        MEM_freeN(winvec);
    }

    // ���������� ��� �������.
    for (LinkNode* link = cloth->springs; link; link = link->next)
    {
        ClothSpring* spring = (ClothSpring*)link->link;
        // ��������� ������ �������� ������.
        if (!(spring->flags & CLOTH_SPRING_FLAG_DEACTIVATE)) 
        {
            cloth_calc_spring_force(clmd, spring);
        }
    }
}

/// <summary>
/// ���������� ��������� �������� ������.
/// </summary>
/// <param name="data">��������� �� ������ ����������� �������.</param>
/// <param name="cell_scale">������� ������.</param>
/// <param name="cell_offset">�������� ������.</param>
/// <param name="index">������ �������.</param>
/// <param name="x">������ ��� �������� ��������� �������.</param>
/// <param name="v">������ ��� �������� �������� �������.</param>
 void cloth_get_grid_location(Implicit_Data* data, float cell_scale, const float cell_offset[3], int index, float x[3], float v[3])
{
    SIM_mass_spring_get_position(data, index, x);
    SIM_mass_spring_get_new_velocity(data, index, v);

    mul_v3_fl(x, cell_scale);
    add_v3_v3(x, cell_offset);
}

/// <summary>
/// ������������ ������� ���������.
/// </summary>
/// <param name="clmd">��������� �� ��������� ClothModifierData.</param>
/// <param name="dt">��������� ���.</param>
static void cloth_calc_average_acceleration(ClothModifierData* clmd, float dt)
{
    Cloth* cloth = clmd->clothObject;
    Implicit_Data* data = cloth->implicit;
    int i, mvert_num = cloth->mvert_num;
    float total[3] = { 0.0f, 0.0f, 0.0f };

    for (i = 0; i < mvert_num; i++) 
    {
        float v[3], nv[3];

        SIM_mass_spring_get_velocity(data, i, v);
        SIM_mass_spring_get_new_velocity(data, i, nv);

        sub_v3_v3(nv, v);
        add_v3_v3(total, nv);
    }

    mul_v3_fl(total, 1.0f / dt / mvert_num);

    /* ����������� ������ � �������������� ���������� ��� �������������� ��������������.
     * ��� ���������� �������� ����������� �������� ��������������� ����� � ��������. */
    interp_v3_v3v3(cloth->average_acceleration, total, cloth->average_acceleration, powf(0.25f, dt));
}

///<summary>
/// ������� cloth_solve_collisions ������ �������� � ����� � ��������� ������� � �������� ������.
/// ����� �������� ��������� ������������ � �������� � ������� ���������.
///</summary>
/// <param name="depsgraph">��������� �� ���� ������������, ������� �������� ���������� � ������ ����� ��������� � ��������������.</param>
/// <param name="ob">��������� �� ������, �������� ����������� ����������� �����.</param>
/// <param name="clmd">��������� �� ��������� ������ ������������ �����.</param>
/// <param name="step">��� ������� ���������.</param>
/// <param name="dt">������� ��������� ��������.</param>
static void cloth_solve_collisions(Depsgraph* depsgraph, Object* ob, ClothModifierData* clmd, float step, float dt)
{
    Cloth* cloth = clmd->clothObject; // �������� ������ �����
    Implicit_Data* id = cloth->implicit; // �������� ������� ������ �����
    ClothVertex* verts = cloth->verts; // �������� ������� �����
    int mvert_num = cloth->mvert_num; // �������� ���������� ������ �����
    const float time_multiplier = 1.0f / (clmd->sim_parms->dt * clmd->sim_parms->timescale); // ��������� ��������� �������
    int i;

    // ���� �� �������� �������� ��� ������������, ������� �� �������
    if (!(clmd->coll_parms->flags & (CLOTH_COLLSETTINGS_FLAG_ENABLED | CLOTH_COLLSETTINGS_FLAG_SELF)))
    {
        return;
    }

    // ���� ��� BVH-������ ��� �����, ������� �� �������
    if (!clmd->clothObject->bvhtree) {
        return;
    }

    // ������ ������� �������� ����� � �����
    SIM_mass_spring_solve_positions(id, dt);

    // ���������� ������ �� ������� �������
    for (i = 0; i < mvert_num; i++) {
        SIM_mass_spring_get_new_position(id, i, verts[i].tx);

        // ���������� ������� �������� �������
        sub_v3_v3v3(verts[i].tv, verts[i].tx, verts[i].txold);
        zero_v3(verts[i].dcvel);
    }

    // �������� �������� � �������������� BVH-������
    if (cloth_bvh_collision(depsgraph, ob, clmd, step / clmd->sim_parms->timescale, dt / clmd->sim_parms->timescale))
    {
        // ���������� ��������� ������ ����� ����������� ��������
        for (i = 0; i < mvert_num; i++)
        {
            // ���������� ������������ �������
            if ((clmd->sim_parms->vgroup_mass > 0) && (verts[i].flags & CLOTH_VERT_FLAG_PINNED))
            {
                continue;
            }

            SIM_mass_spring_get_new_velocity(id, i, verts[i].tv);
            madd_v3_v3fl(verts[i].tv, verts[i].dcvel, time_multiplier);
            SIM_mass_spring_set_new_velocity(id, i, verts[i].tv);
        }
    }
}

/// <summary>
/// ������� cloth_clear_result ���������� ���������� �������� �����.
/// </summary>
/// <param name="clmd">��������� �� ��������� ������ ������������ �����.</param>
static void cloth_clear_result(ClothModifierData* clmd)
{
    ClothSolverResult* sres = clmd->solver_result; // ��������� ��������� �� ��������� ��������

    // ����� ����������� ��������
    sres->status = 0;
    sres->max_error = sres->min_error = sres->avg_error = 0.0f;
    sres->max_iterations = sres->min_iterations = 0;
    sres->avg_iterations = 0.0f;
}

/// <summary>
/// ������� cloth_record_result ���������� ���������� �������� �����.
/// </summary>
/// <param name="clmd">��������� �� ��������� ������ ������������ �����.</param>
/// <param name="result">��������� �� ��������� � ������������ ��������.</param>
/// <param name="dt">������� ��������� ��������.</param>
static void cloth_record_result(ClothModifierData* clmd, ImplicitSolverResult* result, float dt)
{
    ClothSolverResult* sres = clmd->solver_result; // ��������� ��������� �� ��������� ��������

    // ���� ��������� ��� ���������������
    if (sres->status)
    {
        // ������ ����� ����� ������ ��� �������� ��������
        if (result->status == SIM_SOLVER_SUCCESS)
        {
            sres->min_error = min_ff(sres->min_error, result->error);
            sres->max_error = max_ff(sres->max_error, result->error);
            sres->avg_error += result->error * dt;
        }

        sres->min_iterations = min_ii(sres->min_iterations, result->iterations);
        sres->max_iterations = max_ii(sres->max_iterations, result->iterations);
        sres->avg_iterations += (float)result->iterations * dt;
    }
    else 
    {
        // ������ ����� ����� ������ ��� �������� ��������
        if (result->status == SIM_SOLVER_SUCCESS)
        {
            sres->min_error = sres->max_error = result->error;
            sres->avg_error += result->error * dt;
        }

        sres->min_iterations = sres->max_iterations = result->iterations;
        sres->avg_iterations += (float)result->iterations * dt;
    }

    sres->status |= result->status;
}

/// <summary>
/// ������� �������� ��������� �������� ����� ��� ������� � ������������� Cloth.
/// </summary>
/// <param name="depsgraph">��������� �� ���� ������������ �����.</param>
/// <param name="ob">��������� �� ������, ��� �������� ���������� ���������.</param>
/// <param name="frame">������� ���� ���������.</param>
/// <param name="clmd">��������� �� ��������� ������ ������������ Cloth.</param>
/// <param name="effectors">��������� �� ������ �������� ��������.</param>
/// <returns>���������� true, ���� ��������� ��������� �������.</returns>
bool SIM_cloth_solve(Depsgraph* depsgraph, Object* ob, float frame, ClothModifierData* clmd, ListBase* effectors)
{
    Scene* scene = depsgraph->scene;

    // ������������� ����������
    uint i = 0;
    float step = 0.0f, tf = clmd->sim_parms->timescale;
    Cloth* cloth = clmd->clothObject;
    ClothVertex* verts = cloth->verts;
    uint mvert_num = cloth->mvert_num;
    const float dt = clmd->sim_parms->dt * clmd->sim_parms->timescale;
    Implicit_Data* id = cloth->implicit;

    // �������� ��� ��������� ����������� ��������� �� ���������������� �������� �������� ������ �������
    bool use_acceleration = (clmd->sim_parms->flags & CLOTH_SIMSETTINGS_FLAG_PRESSURE) && (clmd->sim_parms->fluid_density > 0);

    BKE_sim_debug_data_clear_category("collision");

    if (!clmd->solver_result)
    {
        clmd->solver_result = (ClothSolverResult*)MEM_callocN(sizeof(ClothSolverResult), "cloth solver result");
    }
    cloth_clear_result(clmd);

    // �������� ������� � ��������� ������������� (������������� �������)
    if (clmd->sim_parms->vgroup_mass > 0)
    {
        for (i = 0; i < mvert_num; i++)
        {
            if (verts[i].flags & CLOTH_VERT_FLAG_PINNED)
            {
                float v[3];
                sub_v3_v3v3(v, verts[i].xconst, verts[i].xold);
                mul_v3_fl(v, clmd->sim_parms->stepsPerFrame);
                mul_v3_fl(v, 1.0f / clmd->sim_parms->time_scale);
                SIM_mass_spring_set_velocity(id, i, v);
            }
        }
    }

    if (!use_acceleration)
    {
        zero_v3(cloth->average_acceleration);
    }

    // ���� ���������
    while (step < tf)
    {
        ImplicitSolverResult result;

        // ��������� ����������� ��� ������������� ������
        cloth_setup_constraints(clmd);

        // ��������� ���
        SIM_mass_spring_clear_forces(id);

        // ������ ���
        cloth_calc_force(scene, clmd, effectors, step);

        // ������ ����� ��������� � �������
        SIM_mass_spring_solve_velocities(id, dt, &result);
        cloth_record_result(clmd, &result, dt);

        // ������ ��������� ������������
        cloth_solve_collisions(depsgraph, ob, clmd, step, dt);

        if (use_acceleration)
        {
            cloth_calc_average_acceleration(clmd, dt);
        }

        // ������� ��������� ��� ������� � ���������� ����������
        SIM_mass_spring_solve_positions(id, dt);
        SIM_mass_spring_apply_result(id);
        // ����������� ������������ ������ � ���������� �������
        for (i = 0; i < mvert_num; i++)
        {
            if (clmd->sim_parms->vgroup_mass > 0)
            {
                if (verts[i].flags & CLOTH_VERT_FLAG_PINNED)
                {
                    float x[3];
                    interp_v3_v3v3(x, verts[i].xold, verts[i].xconst, (step + dt) / clmd->sim_parms->time_scale);
                    SIM_mass_spring_set_position(id, i, x);
                }
            }

            SIM_mass_spring_get_motion_state(id, i, verts[i].txold, nullptr);
        }

        step += dt;
    }

    // ����������� ����������� ������� � ������ �����
    for (i = 0; i < mvert_num; i++)
    {
        SIM_mass_spring_get_motion_state(id, i, verts[i].x, verts[i].v);
        copy_v3_v3(verts[i].txold, verts[i].x);
    }

    effectors_free(effectors);

    return true;
}

