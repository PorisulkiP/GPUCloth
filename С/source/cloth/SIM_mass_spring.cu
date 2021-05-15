#include "SIM_mass_spring.cuh"

/* Три фазы вычисления скорости:
* Первая фаза.
*   Вычисление динамики каждой частицы, при падении под силой тяжести
*   в вязкой среде(воздухе)
* Вторая фаза.
*   Минимизируем энергию, чтобы применить межчастичные ограничения.
* Третья фаза.
*   Корректировка скорости частиц с учётом второго этапа.
*/


static float I3[3][3] = { 
							{1.0, 0.0, 0.0}, 
							{0.0, 1.0, 0.0}, 
							{0.0, 0.0, 1.0} 
						};

/*
* Главная функция работы с тканью
*/
int SIM_cloth_solve(//Depsgraph* depsgraph, 
                    //Object* ob, 
                    float frame, 
                    ClothModifierData* clmd //, 
                    //ListBase* effectors
                    )
{

    unsigned int i = 0;
    float step = 0.0f, tf = clmd->sim_parms->timescale;
    Cloth* cloth = clmd->clothObject;
    ClothVertex* verts = cloth->verts;
    unsigned int mvert_num = cloth->mvert_num;
    float dt = clmd->sim_parms->dt * clmd->sim_parms->timescale;
    Implicit_Data* id = cloth->implicit;

    /* Hydrostatic pressure gradient of the fluid inside the object is affected by acceleration. */
    bool use_acceleration = (clmd->sim_parms->flags & CLOTH_SIMSETTINGS_FLAG_PRESSURE) &&
        (clmd->sim_parms->fluid_density > 0);

    BKE_sim_debug_data_clear_category("collision");

    if (!clmd->solver_result) {
        clmd->solver_result = (ClothSolverResult*)MEM_callocN(sizeof(ClothSolverResult),
            "cloth solver result");
    }
    cloth_clear_result(clmd);

    if (clmd->sim_parms->vgroup_mass > 0) { /* Do goal stuff. */
        for (i = 0; i < mvert_num; i++) {
            /* update velocities with constrained velocities from pinned verts */
            if (verts[i].flags & CLOTH_VERT_FLAG_PINNED) {
                float v[3];
                sub_v3_v3v3(v, verts[i].xconst, verts[i].xold);
                // mul_v3_fl(v, clmd->sim_parms->stepsPerFrame);
                /* divide by time_scale to prevent constrained velocities from being multiplied */
                mul_v3_fl(v, 1.0f / clmd->sim_parms->time_scale);
                SIM_mass_spring_set_velocity(id, i, v);
            }
        }
    }

    if (!use_acceleration) {
        zero_v3(cloth->average_acceleration);
    }

    while (step < tf) {
        ImplicitSolverResult result;

        /* setup vertex constraints for pinned vertices */
        cloth_setup_constraints(clmd);

        /* initialize forces to zero */
        SIM_mass_spring_clear_forces(id);

        /* calculate forces */
        cloth_calc_force(scene, clmd, frame, effectors, step);

        /* calculate new velocity and position */
        SIM_mass_spring_solve_velocities(id, dt, &result);
        cloth_record_result(clmd, &result, dt);

        /* Calculate collision impulses. */
        cloth_solve_collisions(depsgraph, ob, clmd, step, dt);

        if (is_hair) {
            cloth_continuum_step(clmd, dt);
        }

        if (use_acceleration) {
            cloth_calc_average_acceleration(clmd, dt);
        }

        SIM_mass_spring_solve_positions(id, dt);
        SIM_mass_spring_apply_result(id);

        /* move pinned verts to correct position */
        for (i = 0; i < mvert_num; i++) {
            if (clmd->sim_parms->vgroup_mass > 0) {
                if (verts[i].flags & CLOTH_VERT_FLAG_PINNED) {
                    float x[3];
                    /* divide by time_scale to prevent pinned vertices'
                     * delta locations from being multiplied */
                    interp_v3_v3v3(
                        x, verts[i].xold, verts[i].xconst, (step + dt) / clmd->sim_parms->time_scale);
                    SIM_mass_spring_set_position(id, i, x);
                }
            }

            SIM_mass_spring_get_motion_state(id, i, verts[i].txold, nullptr);
        }

        step += dt;
    }

    /* copy results back to cloth data */
    for (i = 0; i < mvert_num; i++) {
        SIM_mass_spring_get_motion_state(id, i, verts[i].x, verts[i].v);
        copy_v3_v3(verts[i].txold, verts[i].x);
    }

    return 1;
}