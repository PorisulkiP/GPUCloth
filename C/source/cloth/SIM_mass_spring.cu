#include "MEM_guardedalloc.cuh"         

//#include "implicit.cuh"
//#include "cloth_types.cuh"              
#include "meshdata_types.cuh"           
#include "modifier_types.cuh"           
#include "object_force_types.cuh"       
#include "object_types.cuh"             
#include "scene_types.cuh"   
//#include "pointcache.cuh"           

#include "linklist.cuh"                 
//#include "B_math.h"                      
#include "utildefines.h"                 

//#include "cloth.h"      
//#include "effect.h"                      

#include "DEG_depsgraph.cuh"               
#include "DEG_depsgraph_query.cuh"         

//#include "cloth_settings.cuh"           
#include "SIM_mass_spring.cuh"          

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "particle.h"
//#include <kdopbvh.cuh>

#include "mallocn_intern.cuh"
#include "mallocn_lockfree_impl.cu"
#include "leak_detector.cu"
#include "implicit.cu"
//#include "math_matrix.cuh"
#include "intern/math_vector_inline.cu"
#include "intern/math_geom.cu"
#include "intern/kdopbvh.cu"
#include "intern/math_geom_inline.cu"
#include "intern/stack.cu"
#include "intern/pointcache.cu"
#include "intern/math_base_inline.cu"
#include "intern/math_matrix.cu"
#include "intern/collision.cu"
#include "intern/effect.cu"
#include "intern/cloth.cu"
#include <intern/ghash.cu>

/* Три фазы вычисления скорости:
* Первая фаза.
*       Вычисление динамики каждой частицы, при падении под силой тяжести в вязкой среде(воздухе)
* Вторая фаза.
*       Минимизируем энергию, чтобы применить межчастичные ограничения.
* Третья фаза.
*       Корректировка скорости частиц с учётом второго этапа.
*/

__host__ __device__ void cloth_calc_spring_force(const ClothModifierData* clmd, ClothSpring* s);
__host__ __device__  void SIM_mass_spring_force_pressure(Implicit_Data* data, int v1, int v2, int v3, float common_pressure, const float* vertex_pressure, const float weights[3]);

/* Number of off-diagonal non-zero matrix blocks.
 * Basically there is one of these for each vertex-vertex interaction.
 */
__host__ __device__ static int cloth_count_nondiag_blocks(const Cloth* cloth)
{
	int nondiag = 0;

    for (const LinkNode* link = cloth->springs; link; link = link->next) {
	    const auto spring = static_cast<ClothSpring*>(link->link);
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

__host__ __device__ bool cloth_get_pressure_weights(const ClothModifierData* clmd, const MVertTri* vt, float* r_weights)
{
    /* We have custom vertex weights for pressure. */
    if (clmd->sim_parms->vgroup_pressure > 0) {
	    const Cloth* cloth = clmd->clothObject;
	    const ClothVertex* verts = cloth->verts;

        for (uint j = 0; j < 3; j++) 
        {
            r_weights[j] = verts[vt->tri[j]].pressure_factor;

            /* Skip the entire triangle if it has a zero weight. */
            if (r_weights[j] == 0.0f) 
            {
                return false;
            }
        }
    }

    return true;
}

__host__ __device__ void cloth_calc_pressure_gradient(const ClothModifierData* clmd,
                                         const float gradient_vector[3],
                                         float* r_vertex_pressure)
{
	const Cloth* cloth = clmd->clothObject;
    Implicit_Data* data = cloth->implicit;
	const uint mvert_num = cloth->mvert_num;
    float pt[3];

    for (uint i = 0; i < mvert_num; i++) 
    {
        SIM_mass_spring_get_position(data, i, pt);
        r_vertex_pressure[i] = dot_v3v3(pt, gradient_vector);
    }
}

__host__ __device__ float cloth_calc_volume(const ClothModifierData* clmd)
{
    /* Calculate the (closed) cloth volume. */
    const Cloth* cloth = clmd->clothObject;
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

__host__ __device__ static float cloth_calc_rest_volume(const ClothModifierData* clmd)
{
    /* Calculate the (closed) cloth volume. */
    const Cloth* cloth = clmd->clothObject;
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

__host__ __device__ static float cloth_calc_average_pressure(const ClothModifierData* clmd, const float* vertex_pressure)
{
	const Cloth* cloth = clmd->clothObject;
    const MVertTri* tri = cloth->tri;
    Implicit_Data* data = cloth->implicit;
    float weights[3] = { 1.0f, 1.0f, 1.0f };
    float total_force = 0;
    float total_area = 0;

    for (uint i = 0; i < cloth->primitive_num; i++) {
        const MVertTri* vt = &tri[i];

        if (cloth_get_pressure_weights(clmd, vt, weights)) {
	        const float area = SIM_tri_area(data, vt->tri[0], vt->tri[1], vt->tri[2]);

            total_force += (vertex_pressure[vt->tri[0]] + vertex_pressure[vt->tri[1]] +
                vertex_pressure[vt->tri[2]]) *
                area / 3.0f;
            total_area += area;
        }
    }

    return total_force / total_area;
}

__host__ int SIM_cloth_solver_init(const ClothModifierData* clmd)
{
    Cloth* cloth = clmd->clothObject;
    const ClothVertex* verts = cloth->verts;
    constexpr float ZERO[3] = { 0.0f, 0.0f, 0.0f };
    Implicit_Data* id;
    uint i;

    const uint nondiag = cloth_count_nondiag_blocks(cloth);
    cloth->implicit = id = SIM_mass_spring_solver_create(cloth->mvert_num, nondiag);

    for (i = 0; i < cloth->mvert_num; i++) {
        SIM_mass_spring_set_vertex_mass(id, i, verts[i].mass);
    }

    for (i = 0; i < cloth->mvert_num; i++) {
        SIM_mass_spring_set_motion_state(id, i, verts[i].x, ZERO);
    }

    return 1;
}

__host__ __device__ void SIM_cloth_solver_free(const ClothModifierData* clmd)
{
    Cloth* cloth = clmd->clothObject;

    if (cloth->implicit) {
        SIM_mass_spring_solver_free(cloth->implicit);
        cloth->implicit = nullptr;
    }
}

__host__ __device__ void SIM_cloth_solver_set_positions(const ClothModifierData* clmd)
{
    constexpr float I3[3][3] = { {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0} };
	const Cloth* cloth = clmd->clothObject;
	const ClothVertex* verts = cloth->verts;
	const uint mvert_num = cloth->mvert_num;
    Implicit_Data* id = cloth->implicit;

    for (uint i = 0; i < mvert_num; i++) 
    {
        SIM_mass_spring_set_rest_transform(id, i, I3);
        SIM_mass_spring_set_motion_state(id, i, verts[i].x, verts[i].v);
    }
}

__host__ __device__ void SIM_cloth_solver_set_volume(const ClothModifierData* clmd)
{
    Cloth* cloth = clmd->clothObject;

    cloth->initial_mesh_volume = cloth_calc_rest_volume(clmd);
}

/* Init constraint matrix
 * This is part of the modified CG method suggested by Baraff/Witkin in
 * "Large Steps in Cloth Simulation" (Siggraph 1998)
 */
__host__ __device__ void cloth_setup_constraints(const ClothModifierData* clmd)
{
	const Cloth* cloth = clmd->clothObject;
    Implicit_Data* data = cloth->implicit;
    ClothVertex* verts = cloth->verts;
	const uint mvert_num = cloth->mvert_num;

    constexpr float ZERO[3] = { 0.0f, 0.0f, 0.0f };

    SIM_mass_spring_clear_constraints(data);

    for (uint v = 0; v < mvert_num; ++v) 
    {
        if (verts[v].flags & CLOTH_VERT_FLAG_PINNED) 
        {
            /* pinned vertex constraints */
            SIM_mass_spring_add_constraint_ndof0(data, v, ZERO); /* velocity is defined externally */
        }

        verts[v].impulse_count = 0;
    }
}

__global__ void g_cloth_setup_constraints(const ClothModifierData* clmd)
{
    const Cloth* cloth = clmd->clothObject;
    Implicit_Data* data = cloth->implicit;
    ClothVertex* verts = cloth->verts;
    const int mvert_num = cloth->mvert_num;

    constexpr float ZERO[3] = { 0.0f, 0.0f, 0.0f };

    SIM_mass_spring_clear_constraints(data);

    for (int v = 0; v < mvert_num; v++)
    {
        if (verts[v].flags & CLOTH_VERT_FLAG_PINNED)
        {
            /* pinned vertex constraints */
            SIM_mass_spring_add_constraint_ndof0(data, v, ZERO); /* velocity is defined externally */
        }

        verts[v].impulse_count = 0;
    }
}

__global__ void g_cloth_calc_spring_force(const ClothModifierData* clmd, ClothSpring* s)
{
    const Cloth* cloth = clmd->clothObject;
    const ClothSimSettings* parms = clmd->sim_parms;
    Implicit_Data* data = cloth->implicit;
    const bool using_angular = parms->bending_model == CLOTH_BENDING_ANGULAR;
    const bool resist_compress = (parms->flags & CLOTH_SIMSETTINGS_FLAG_RESIST_SPRING_COMPRESS) && !using_angular;

    s->flags &= ~CLOTH_SPRING_FLAG_NEEDED;

    /* Рассчитайте усилие изгиба пружин. */
#ifdef CLOTH_FORCE_SPRING_BEND
    if ((s->type & CLOTH_SPRING_TYPE_BENDING) && using_angular)
    {

        s->flags |= CLOTH_SPRING_FLAG_NEEDED;

        const float scaling = parms->bending + s->ang_stiffness * fabsf(parms->max_bend - parms->bending);
        const float k = scaling * s->restlen * 0.1f; /* Умножаем на 0,1, просто чтобы масштабировать силы до более разумных значений. */

        SIM_mass_spring_force_spring_angular(data, s->ij, s->kl, s->pa, s->pb, s->la, s->lb, s->restang, k, parms->bending_damping);
    }
#endif

    /* Рассчитать усилие конструкционных + сдвиговых пружин. */
    if (s->type & (CLOTH_SPRING_TYPE_STRUCTURAL | CLOTH_SPRING_TYPE_SEWING | CLOTH_SPRING_TYPE_INTERNAL)) {
#ifdef CLOTH_FORCE_SPRING_STRUCTURAL

        s->flags |= CLOTH_SPRING_FLAG_NEEDED;

        float scaling_tension = parms->tension + s->lin_stiffness * fabsf(parms->max_tension - parms->tension);
        float k_tension = scaling_tension / (parms->avg_spring_len + FLT_EPSILON);

        if (s->type & CLOTH_SPRING_TYPE_SEWING)
        {
            /* TODO: проверить, наполовину проверено (не удалось увидеть ошибку)
             * поначалу расстояние между швейными пружинами обычно большое, поэтому ограничьте усилие, чтобы не получить
             * туннелирование через объекты столкновения. */
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
        else if (s->type & CLOTH_SPRING_TYPE_STRUCTURAL)
        {
            const float scaling_compression = parms->compression + s->lin_stiffness * fabsf(parms->max_compression - parms->compression);
            const float k_compression = scaling_compression / (parms->avg_spring_len + FLT_EPSILON);

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
        else
        {
            /* CLOTH_SPRING_TYPE_INTERNAL */
            BLI_assert(s->type & CLOTH_SPRING_TYPE_INTERNAL);

            scaling_tension = parms->internal_tension + s->lin_stiffness * fabsf(parms->max_internal_tension - parms->internal_tension);
            k_tension = scaling_tension / (parms->avg_spring_len + FLT_EPSILON);
            const float scaling_compression = parms->internal_compression + s->lin_stiffness * fabsf(parms->max_internal_compression - parms->internal_compression);
            const float k_compression = scaling_compression / (parms->avg_spring_len + FLT_EPSILON);

            float k_tension_damp = parms->tension_damp;
            float k_compression_damp = parms->compression_damp;

            if (k_tension == 0.0f)
            {
                /* Никакого демпфирования, поэтому он ведет себя так, как будто пружины натяжения вообще не было. */
                k_tension_damp = 0.0f;
            }

            if (k_compression == 0.0f)
            {
                /* Никакого демпфирования, поэтому он ведет себя так, как будто пружины сжатия вообще не было. */
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
    else if (s->type & CLOTH_SPRING_TYPE_SHEAR)
    {
#ifdef CLOTH_FORCE_SPRING_SHEAR

        s->flags |= CLOTH_SPRING_FLAG_NEEDED;

        const float scaling = parms->shear + s->lin_stiffness * fabsf(parms->max_shear - parms->shear);
        const float k = scaling / (parms->avg_spring_len + FLT_EPSILON);

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
    else if (s->type & CLOTH_SPRING_TYPE_BENDING) /* рассчитать усилие изгиба пружин */
    {
#ifdef CLOTH_FORCE_SPRING_BEND

        s->flags |= CLOTH_SPRING_FLAG_NEEDED;

        const float scaling = parms->bending + s->lin_stiffness * fabsf(parms->max_bend - parms->bending);
        const float kb = scaling / (20.0f * (parms->avg_spring_len + FLT_EPSILON));

        /* Исправление для T45084 для жесткости ткани должно быть cb пропорционально kb */
        const float cb = kb * parms->bending_damping;

        SIM_mass_spring_force_spring_bending(data, s->ij, s->kl, s->restlen, kb, cb);
#endif
    }
}


/// <summary>
/// Рассчитывает силы, действующие на сетку ткани.
/// </summary>
/// <param name="scene">Указатель на текущую сцену.</param>
/// <param name="clmd">Указатель на модификатор данных ткани.</param>
/// <param name="effectors">Список эффекторов, влияющих на ткань.</param>
/// <param name="time">Текущее время симуляции.</param>
__host__ __device__ void cloth_calc_force(const Scene* scene, const ClothModifierData* clmd, ListBase* effectors, const float time)
{
    // Сбор сил и производных: F, dFdX, dFdV
    const Cloth* cloth = clmd->clothObject;
    const ClothSimSettings* parms = clmd->sim_parms;
    Implicit_Data* data = cloth->implicit;
    uint i;
    const float drag = clmd->sim_parms->Cvi * 0.01f; // вязкость воздуха в процентах
    float gravity[3] = { 0.0f, 0.0f, 0.0f };
    const MVertTri* tri = cloth->tri;
    const uint mvert_num = cloth->mvert_num;
    ClothVertex* vert;

#ifdef CLOTH_FORCE_GRAVITY
    // глобальное ускорение (гравитация)
    if (scene->physics_settings.flag & PHYS_GLOBAL_GRAVITY)
    {
        // масштабирование силы гравитации
        mul_v3_v3fl(gravity, scene->physics_settings.gravity, 0.001f * clmd->sim_parms->effector_weights->global_gravity);
    }

    vert = cloth->verts;
    for (i = 0; i < cloth->mvert_num; i++, vert++)
    {
        SIM_mass_spring_force_gravity(data, i, vert->mass, gravity);

        // Вертикальные целевые пружины
        if ((!(vert->flags & CLOTH_VERT_FLAG_PINNED)) && (vert->goal > FLT_EPSILON))
        {
            float goal_x[3], goal_v[3];

            // разделение по time_scale для предотвращения умножения delta положений целевых вершин
            interp_v3_v3v3(goal_x, vert->xold, vert->xconst, time / clmd->sim_parms->time_scale);
            sub_v3_v3v3(goal_v, vert->xconst, vert->xold); // расстояние, пройденное за dt==1

            const float k = vert->goal * clmd->sim_parms->goalspring / (clmd->sim_parms->avg_spring_len + FLT_EPSILON);

            SIM_mass_spring_force_spring_goal(data, i, goal_x, goal_v, k, clmd->sim_parms->goalfrict * 0.01f);
        }
    }
#endif

#ifdef CLOTH_FORCE_DRAG
    SIM_mass_spring_force_drag(data, drag);
#endif

    // Обработка сил давления (убедитесь, что это никогда не вычисляется для волос).
    if ((parms->flags & CLOTH_SIMSETTINGS_FLAG_PRESSURE))
    {
        // Разница в давлении между внутренней и внешней частями сетки.
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

        // Проверка необходимости вычисления объема сетки.
        if (init_vol > 1E-6f) 
        {
	        const float vol = cloth_calc_volume(clmd);

            // Если объем такой же, не применяйте давление.
            volume_factor = init_vol / vol;
            pressure_difference = volume_factor - 1;

            // Вычисление искусственного максимального значения для давления ткани.
	        const float f = fabs(clmd->sim_parms->uniform_pressure_force) + 200.0f;

            // Зажим давления ткани до вычисленного максимального значения.
            CLAMP_MAX(pressure_difference, f);
        }

        pressure_difference += clmd->sim_parms->uniform_pressure_force;
        pressure_difference *= clmd->sim_parms->pressure_factor;

        // Вычисление градиента гидростатического давления, если включено.
        float fluid_density = clmd->sim_parms->fluid_density * 1000; // кг/л -> кг/м3
        float* hydrostatic_pressure = nullptr;

        if (fabs(fluid_density) > 1e-6f)
        {
            float hydrostatic_vector[3];
            copy_v3_v3(hydrostatic_vector, gravity);

            // Когда жидкость находится внутри объекта, учитывать ускорение
            // объекта в поле давления, так как гравитация неотличима
            // от ускорения изнутри.
            if (fluid_density > 0)
            {
                sub_v3_v3(hydrostatic_vector, cloth->average_acceleration);

                // Сохранение общей массы путем масштабирования плотности для соответствия изменению объема.
                fluid_density *= volume_factor;
            }

            mul_v3_fl(hydrostatic_vector, fluid_density);

            // Вычисление массива гидростатического давления для каждой вершины и вычитание среднего значения.
            hydrostatic_pressure = static_cast<float*>(MEM_lockfree_mallocN(sizeof(float) * mvert_num, "hydrostatic pressure gradient"));

            cloth_calc_pressure_gradient(clmd, hydrostatic_vector, hydrostatic_pressure);

            pressure_difference -= cloth_calc_average_pressure(clmd, hydrostatic_pressure);
        }

        // Применение давления.
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
            MEM_lockfree_freeN(hydrostatic_pressure);
        }
    }

    // Обработка внешних сил, таких как ветер.
    if (effectors)
    {
        bool has_wind = false, has_force = false;

        // Кэширование сил для каждой вершины, чтобы избежать избыточных вычислений.
        auto winvec = static_cast<float(*)[3]>(MEM_lockfree_callocN(sizeof(float[3]) * mvert_num * 2, "силы эффекторов"));
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
        MEM_lockfree_freeN(winvec);
    }

    for (const LinkNode* link = cloth->springs; link; link = link->next) 
    {
	    auto* spring = static_cast<ClothSpring*>(link->link);
        /* обращайтесь только с активными пружинами */
        if (!(spring->flags & CLOTH_SPRING_FLAG_DEACTIVATE)) 
        {
            cloth_calc_spring_force(clmd, spring);
        }
    }
}

/// <summary>
/// Возвращает состояние движения вершин.
/// </summary>
/// <param name="data">Указатель на данные имплицитной системы.</param>
/// <param name="cell_scale">Масштаб ячейки.</param>
/// <param name="cell_offset">Смещение ячейки.</param>
/// <param name="index">Индекс вершины.</param>
/// <param name="x">Массив для хранения координат вершины.</param>
/// <param name="v">Массив для хранения скорости вершины.</param>
 void cloth_get_grid_location(Implicit_Data* data, const float cell_scale, const float cell_offset[3], const int index, float x[3], float v[3])
{
    SIM_mass_spring_get_position(data, index, x);
    SIM_mass_spring_get_new_velocity(data, index, v);

    mul_v3_fl(x, cell_scale);
    add_v3_v3(x, cell_offset);
}

/// <summary>
/// Рассчитывает среднее ускорение.
/// </summary>
/// <param name="clmd">Указатель на структуру ClothModifierData.</param>
/// <param name="dt">Временной шаг.</param>
 __host__ __device__  void cloth_calc_average_acceleration(const ClothModifierData* clmd, const float dt)
{
    Cloth* cloth = clmd->clothObject;
    const Implicit_Data* data = cloth->implicit;
    const int mvert_num = cloth->mvert_num;
    float total[3] = { 0.0f, 0.0f, 0.0f };

    for (int i = 0; i < mvert_num; i++) 
    {
        float v[3], nv[3];

        SIM_mass_spring_get_velocity(data, i, v);
        SIM_mass_spring_get_new_velocity(data, i, nv);

        sub_v3_v3(nv, v);
        add_v3_v3(total, nv);
    }

    mul_v3_fl(total, 1.0f / dt / mvert_num);

    /* Сглаживание данных с использованием усреднения для предотвращения неустойчивости.
     * Это фактически является абстракцией скорости распространения волны в жидкости. */
    interp_v3_v3v3(cloth->average_acceleration, total, cloth->average_acceleration, powf(0.25f, dt));
}

///<summary>
/// Функция cloth_solve_collisions решает коллизии в ткани и обновляет позиции и скорости вершин.
/// Также включает обработку самоколлизий и коллизий с другими объектами.
///</summary>
/// <param name="depsgraph">Указатель на граф зависимостей, который содержит информацию о связях между объектами и модификаторами.</param>
/// <param name="ob">Указатель на объект, которому применяется модификатор ткани.</param>
/// <param name="clmd">Указатель на структуру данных модификатора ткани.</param>
/// <param name="step">Шаг времени симуляции.</param>
/// <param name="dt">Текущий временной интервал.</param>
__device__  void cloth_solve_collisions(const Depsgraph* depsgraph, Object* ob, ClothModifierData* clmd, const float step, const float dt)
{
	const Cloth* cloth = clmd->clothObject; // Получаем объект ткани
    Implicit_Data* id = cloth->implicit; // Получаем неявные данные ткани
    ClothVertex* verts = cloth->verts; // Получаем вершины ткани
    const int mvert_num = cloth->mvert_num; // Получаем количество вершин ткани
    const float time_multiplier = 1.0f / (clmd->sim_parms->dt * clmd->sim_parms->timescale); // Вычисляем множитель времени
    int i;

    // Если не включены коллизии или самоколлизии, выходим из функции
    if (!(clmd->coll_parms->flags & (CLOTH_COLLSETTINGS_FLAG_ENABLED | CLOTH_COLLSETTINGS_FLAG_SELF)))
    {
        return;
    }

    // Если нет BVH-дерева для ткани, выходим из функции
    if (!clmd->clothObject->bvhtree) {
        return;
    }

    // Решаем позиции массовых точек в ткани
    SIM_mass_spring_solve_positions(id, dt);

    // Обновление вершин до текущих позиций
    for (i = 0; i < mvert_num; i++) {
        SIM_mass_spring_get_new_position(id, i, verts[i].tx);

        // Вычисление текущей скорости вершины
        sub_v3_v3v3(verts[i].tv, verts[i].tx, verts[i].txold);
        zero_v3(verts[i].dcvel);
    }

    // Проверка коллизий с использованием BVH-дерева
    if (cloth_bvh_collision(depsgraph, ob, clmd, step / clmd->sim_parms->timescale, dt / clmd->sim_parms->timescale))
    {
        // Обновление скоростей вершин после обнаружения коллизий
        for (i = 0; i < mvert_num; i++)
        {
            // Пропускаем закрепленные вершины
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
/// Функция cloth_clear_result сбрасывает результаты решателя ткани.
/// </summary>
/// <param name="clmd">Указатель на структуру данных модификатора ткани.</param>
__host__ __device__ void cloth_clear_result(const ClothModifierData* clmd)
{
    ClothSolverResult* sres = clmd->solver_result; // Получение указателя на результат решателя

    // Сброс результатов решателя
    sres->status = 0;
    sres->max_error = sres->min_error = sres->avg_error = 0.0f;
    sres->max_iterations = sres->min_iterations = 0;
    sres->avg_iterations = 0.0f;
}

/// <summary>
/// Функция cloth_record_result записывает результаты решателя ткани.
/// </summary>
/// <param name="clmd">Указатель на структуру данных модификатора ткани.</param>
/// <param name="result">Указатель на структуру с результатами решателя.</param>
/// <param name="dt">Текущий временной интервал.</param>
__host__ __device__  void cloth_record_result(const ClothModifierData* clmd, const ImplicitSolverResult* result, const float dt)
{
    ClothSolverResult* sres = clmd->solver_result; // Получение указателя на результат решателя

    // Если результат уже инициализирован
    if (sres->status)
    {
        // Ошибка имеет смысл только для успешных итераций
        if (result->status == SIM_SOLVER_SUCCESS)
        {
            sres->min_error = min_ff(sres->min_error, result->error);
            sres->max_error = max_ff(sres->max_error, result->error);
            sres->avg_error += result->error * dt;
        }

        sres->min_iterations = min_ii(sres->min_iterations, result->iterations);
        sres->max_iterations = max_ii(sres->max_iterations, result->iterations);
        sres->avg_iterations += static_cast<float>(result->iterations) * dt;
    }
    else 
    {
        // Ошибка имеет смысл только для успешных итераций
        if (result->status == SIM_SOLVER_SUCCESS)
        {
            sres->min_error = sres->max_error = result->error;
            sres->avg_error += result->error * dt;
        }

        sres->min_iterations = sres->max_iterations = result->iterations;
        sres->avg_iterations += static_cast<float>(result->iterations) * dt;
    }

    sres->status |= result->status;
}

/// <summary>
/// Функция проводит симуляцию динамики ткани для объекта с модификатором Cloth.
/// </summary>
/// <param name="depsgraph">Указатель на граф зависимостей сцены.</param>
/// <param name="ob">Указатель на объект, для которого проводится симуляция.</param>
/// <param name="clmd">Указатель на структуру данных модификатора Cloth.</param>
/// <param name="effectors">Указатель на список влияющих объектов.</param>
/// <returns>Возвращает true, если симуляция выполнена успешно.</returns>
__device__ bool SIM_cloth_solve(const Depsgraph* depsgraph, Object* ob, ClothModifierData* clmd, ListBase* effectors)
{
	const Scene* scene = depsgraph->scene;

    // Инициализация переменных
    uint i;
    float step = 0.0f;
    const float tf = clmd->sim_parms->timescale;
    Cloth* cloth = clmd->clothObject;
    ClothVertex* verts = cloth->verts;
    const uint mvert_num = cloth->mvert_num;
    const float dt = clmd->sim_parms->dt * clmd->sim_parms->timescale;
    Implicit_Data* id = cloth->implicit;

    // Включить или отключить воздействие ускорения на гидростатическое давление жидкости внутри объекта
    const bool use_acceleration = (clmd->sim_parms->flags & CLOTH_SIMSETTINGS_FLAG_PRESSURE) && (clmd->sim_parms->fluid_density > 0);

    BKE_sim_debug_data_clear_category("collision");

    if (!clmd->solver_result)
    {
        clmd->solver_result = static_cast<ClothSolverResult*>(MEM_lockfree_callocN(sizeof(ClothSolverResult), "cloth solver result"));
    }
    cloth_clear_result(clmd);

    // Обновить вершины с заданными ограничениями (прикрепленные вершины)
    if (clmd->sim_parms->vgroup_mass > 0)
    {
        for (i = 0; i < mvert_num; i++)
        {
            if (verts[i].flags & CLOTH_VERT_FLAG_PINNED)
            {
                float v[3];
                sub_v3_v3v3(v, verts[i].xconst, verts[i].xold);
                mul_v3_fl(v, static_cast<float>(clmd->sim_parms->stepsPerFrame));
                mul_v3_fl(v, 1.0f / clmd->sim_parms->time_scale);
                SIM_mass_spring_set_velocity(id, i, v);
            }
        }
    }

    if (!use_acceleration)
    {
        zero_v3(cloth->average_acceleration);
    }

    // Цикл симуляции
    while (step < tf)
    {
        ImplicitSolverResult result;

        // Установка ограничений для прикрепленных вершин
        cloth_setup_constraints(clmd);

        // Обнуление сил
        SIM_mass_spring_clear_forces(id);

        // Расчет сил
        cloth_calc_force(scene, clmd, effectors, step);

        // Расчет новых скоростей и позиций
        SIM_mass_spring_solve_velocities(id, dt, &result); // result->status == SIM_SOLVER_SUCCESS
        cloth_record_result(clmd, &result, dt);

        // Расчет импульсов столкновений
        cloth_solve_collisions(depsgraph, ob, clmd, step, dt);

        if (use_acceleration)
        {
            cloth_calc_average_acceleration(clmd, dt);
        }

        // Решение уравнений для позиций и применение результата
        SIM_mass_spring_solve_positions(id, dt);
        SIM_mass_spring_apply_result(id);

        step += dt;
    }

    // Копирование результатов обратно в данные ткани
    for (i = 0; i < mvert_num; i++)
    {
        SIM_mass_spring_get_motion_state(id, i, verts[i].x, verts[i].v);
        copy_v3_v3(verts[i].txold, verts[i].x);
    }

    effectors_free(effectors);

    return true;
}

__host__ __device__ void SIM_mass_spring_get_position(Implicit_Data* data, const int index, float x[3])
{
    root_to_world_v3(data, index, x, data->X[index]);
}

__host__ __device__ float SIM_tri_tetra_volume_signed_6x(Implicit_Data* data, const int v1, const int v2, const int v3)
{
    /* The result will be 6x the volume */
    return volume_tri_tetrahedron_signed_v3_6x(data->X[v1], data->X[v2], data->X[v3]);
}

__host__ __device__ bool SIM_mass_spring_force_spring_angular(Implicit_Data* data,
    const int i, const int j,
    const int* i_a, const int* i_b,
    const int len_a, const int len_b,
    const float restang, const float stiffness, const float damping)
{
    float angle, dir_a[3], dir_b[3], vel_a[3], vel_b[3];
    float f_a[3], f_b[3], f_e[3];
    int x;

    spring_angle(data, i, j, i_a, i_b, len_a, len_b, dir_a, dir_b, &angle, vel_a, vel_b);

    /* spring force */
    float force = stiffness * (angle - restang);

    /* damping force */
    force += -damping * (dot_v3v3(vel_a, dir_a) + dot_v3v3(vel_b, dir_b));

    mul_v3_v3fl(f_a, dir_a, force / len_a);
    mul_v3_v3fl(f_b, dir_b, force / len_b);

    for (x = 0; x < len_a; x++)
    {
        add_v3_v3(data->F[i_a[x]], f_a);
    }

    for (x = 0; x < len_b; x++)
    {
        add_v3_v3(data->F[i_b[x]], f_b);
    }

    mul_v3_v3fl(f_a, dir_a, force * 0.5f);
    mul_v3_v3fl(f_b, dir_b, force * 0.5f);

    add_v3_v3v3(f_e, f_a, f_b);

    sub_v3_v3(data->F[i], f_e);
    sub_v3_v3(data->F[j], f_e);

    return true;
}

__host__ __device__ bool SIM_mass_spring_force_spring_linear(Implicit_Data* data,
    const int i, const int j,
    const float restlen,
    const float stiffness_tension,
    const float damping_tension,
    const float stiffness_compression,
    const float damping_compression,
    const bool resist_compress,
    const bool new_compress,
    const float clamp_force)
{
    float extent[3], length, dir[3], vel[3];
    float f[3], dfdx[3][3], dfdv[3][3];
    float damping;

    /* calculate elongation */
    spring_length(data, i, j, extent, dir, &length, vel);

    /* This code computes not only the force, but also its derivative.
     * Zero derivative effectively disables the spring for the implicit solver.
     * Thus length > restlen makes cloth unconstrained at the start of simulation. */
    if ((length >= restlen && length > 0) || resist_compress)
    {
        damping = damping_tension;

        float stretch_force = stiffness_tension * (length - restlen);
        if (clamp_force > 0.0f && stretch_force > clamp_force)
        {
            stretch_force = clamp_force;
        }
        mul_v3_v3fl(f, dir, stretch_force);

        dfdx_spring(dfdx, dir, length, restlen, stiffness_tension);
    }
    else if (new_compress) {
        /* This is based on the Choi and Ko bending model,
         * which works surprisingly well for compression. */
        const float kb = stiffness_compression;
        const float cb = kb; /* cb equal to kb seems to work, but a factor can be added if necessary */

        damping = damping_compression;

        mul_v3_v3fl(f, dir, fbstar(length, restlen, kb, cb));

        outerproduct(dfdx, dir, dir);
        mul_m3_fl(dfdx, fbstar_jacobi(length, restlen, kb, cb));
    }
    else {
        return false;
    }

    madd_v3_v3fl(f, dir, damping * dot_v3v3(vel, dir));
    dfdv_damp(dfdv, dir, damping);

    apply_spring(data, i, j, f, dfdx, dfdv);

    return true;
}

__host__ __device__ void BLI_addtail(ListBase* listbase, void* vlink)
{
    Link* link = (Link*)vlink;

    if (link == NULL) {
        return;
    }

    link->next = NULL;
    link->prev = static_cast<Link*>(listbase->last);

    if (listbase->last) {
        static_cast<Link*>(listbase->last)->next = link;
    }
    if (listbase->first == NULL) {
        listbase->first = link;
    }
    listbase->last = link;
}

__host__ __device__ void cloth_calc_spring_force(const ClothModifierData* clmd, ClothSpring* s)
{
    const Cloth* cloth = clmd->clothObject;
    const ClothSimSettings* parms = clmd->sim_parms;
    Implicit_Data* data = cloth->implicit;
    const bool using_angular = parms->bending_model == CLOTH_BENDING_ANGULAR;
    const bool resist_compress = (parms->flags & CLOTH_SIMSETTINGS_FLAG_RESIST_SPRING_COMPRESS) && !using_angular;

    s->flags &= ~CLOTH_SPRING_FLAG_NEEDED;

    /* Рассчитайте усилие изгиба пружин. */
#ifdef CLOTH_FORCE_SPRING_BEND
    if ((s->type & CLOTH_SPRING_TYPE_BENDING) && using_angular)
    {

        s->flags |= CLOTH_SPRING_FLAG_NEEDED;

        const float scaling = parms->bending + s->ang_stiffness * fabsf(parms->max_bend - parms->bending);
        const float k = scaling * s->restlen * 0.1f; /* Умножаем на 0,1, просто чтобы масштабировать силы до более разумных значений. */

        SIM_mass_spring_force_spring_angular(data, s->ij, s->kl, s->pa, s->pb, s->la, s->lb, s->restang, k, parms->bending_damping);
    }
#endif

    /* Рассчитать усилие конструкционных + сдвиговых пружин. */
    if (s->type & (CLOTH_SPRING_TYPE_STRUCTURAL | CLOTH_SPRING_TYPE_SEWING | CLOTH_SPRING_TYPE_INTERNAL)) {
#ifdef CLOTH_FORCE_SPRING_STRUCTURAL

        s->flags |= CLOTH_SPRING_FLAG_NEEDED;

        float scaling_tension = parms->tension + s->lin_stiffness * fabsf(parms->max_tension - parms->tension);
        float k_tension = scaling_tension / (parms->avg_spring_len + FLT_EPSILON);

        if (s->type & CLOTH_SPRING_TYPE_SEWING)
        {
            /* TODO: проверить, наполовину проверено (не удалось увидеть ошибку)
             * поначалу расстояние между швейными пружинами обычно большое, поэтому ограничьте усилие, чтобы не получить
             * туннелирование через объекты столкновения. */
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
        else if (s->type & CLOTH_SPRING_TYPE_STRUCTURAL)
        {
            const float scaling_compression = parms->compression + s->lin_stiffness * fabsf(parms->max_compression - parms->compression);
            const float k_compression = scaling_compression / (parms->avg_spring_len + FLT_EPSILON);

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
        else
        {
            /* CLOTH_SPRING_TYPE_INTERNAL */
            BLI_assert(s->type & CLOTH_SPRING_TYPE_INTERNAL);

            scaling_tension = parms->internal_tension + s->lin_stiffness * fabsf(parms->max_internal_tension - parms->internal_tension);
            k_tension = scaling_tension / (parms->avg_spring_len + FLT_EPSILON);
            const float scaling_compression = parms->internal_compression + s->lin_stiffness * fabsf(parms->max_internal_compression - parms->internal_compression);
            const float k_compression = scaling_compression / (parms->avg_spring_len + FLT_EPSILON);

            float k_tension_damp = parms->tension_damp;
            float k_compression_damp = parms->compression_damp;

            if (k_tension == 0.0f)
            {
                /* Никакого демпфирования, поэтому он ведет себя так, как будто пружины натяжения вообще не было. */
                k_tension_damp = 0.0f;
            }

            if (k_compression == 0.0f)
            {
                /* Никакого демпфирования, поэтому он ведет себя так, как будто пружины сжатия вообще не было. */
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
    else if (s->type & CLOTH_SPRING_TYPE_SHEAR)
    {
#ifdef CLOTH_FORCE_SPRING_SHEAR

        s->flags |= CLOTH_SPRING_FLAG_NEEDED;

        const float scaling = parms->shear + s->lin_stiffness * fabsf(parms->max_shear - parms->shear);
        const float k = scaling / (parms->avg_spring_len + FLT_EPSILON);

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
    else if (s->type & CLOTH_SPRING_TYPE_BENDING) /* рассчитать усилие изгиба пружин */
    {
#ifdef CLOTH_FORCE_SPRING_BEND

        s->flags |= CLOTH_SPRING_FLAG_NEEDED;

        const float scaling = parms->bending + s->lin_stiffness * fabsf(parms->max_bend - parms->bending);
        const float kb = scaling / (20.0f * (parms->avg_spring_len + FLT_EPSILON));

        /* Исправление для T45084 для жесткости ткани должно быть cb пропорционально kb */
        const float cb = kb * parms->bending_damping;

        SIM_mass_spring_force_spring_bending(data, s->ij, s->kl, s->restlen, kb, cb);
#endif
    }
}

__host__ __device__ void interp_v3_v3v3(float r[3], const float a[3], const float b[3], const float t)
{
    const float s = 1.0f - t;

    r[0] = s * a[0] + t * b[0];
    r[1] = s * a[1] + t * b[1];
    r[2] = s * a[2] + t * b[2];
}

__host__ __device__ int BLI_listbase_count(const ListBase* listbase)
{
    int count = 0;

    for (auto link = static_cast<Link*>(listbase->first); link; link = link->next) {
        count++;
    }

    return count;
}

__host__ __device__ void BLI_freelistN(ListBase* listbase)
{
    Link* link, * next;

    link = (Link*)listbase->first;
    while (link) {
        next = link->next;
        MEM_lockfree_freeN(link);
        link = next;
    }

    BLI_listbase_clear(listbase);
}

__host__ __device__ bool is_finite_v3(const float v[3])
{
    return (isfinite(v[0]) && isfinite(v[1]) && isfinite(v[2]));
}

__host__ __device__ bool is_finite_v4(const float v[4])
{
    return (isfinite(v[0]) && isfinite(v[1]) && isfinite(v[2]) && isfinite(v[3]));
}

__host__ __device__ uint BLI_ghashutil_strhash_p(const void* ptr)
{
    const char* p;
    uint h = 5381;

    for (p = (char*)ptr; *p != '\0'; p++) {
        h = (uint)((h << 5) + h) + (uint)*p;
    }

    return h;
}