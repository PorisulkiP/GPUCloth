#pragma once

#ifndef __SIM_MASS_SPRING__
#define __SIM_MASS_SPRING__

#define PHYS_GLOBAL_GRAVITY 1

#include <cuda_runtime_api.h>

struct ClothModifierData;
struct Depsgraph;
struct Implicit_Data;
struct ListBase;
struct Object;

constexpr float I3[3][3] = { {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0} };

typedef enum eMassSpringSolverStatus {
    SIM_SOLVER_SUCCESS          = (1 << 0),
    SIM_SOLVER_NUMERICAL_ISSUE  = (1 << 1),
    SIM_SOLVER_NO_CONVERGENCE   = (1 << 2),
    SIM_SOLVER_INVALID_INPUT    = (1 << 3),
} eMassSpringSolverStatus;

__host__ __device__ Implicit_Data* SIM_mass_spring_solver_create(uint numverts, uint numsprings);
__host__ __device__ void SIM_mass_spring_solver_free(Implicit_Data* id);

__host__ int SIM_cloth_solver_init(const ClothModifierData* clmd);
__host__ __device__ void SIM_cloth_solver_free(const ClothModifierData* clmd);
__device__ bool SIM_cloth_solve(const Depsgraph* depsgraph, Object* ob, ClothModifierData* clmd, ListBase* effectors);
__host__ __device__ void SIM_cloth_solver_set_positions(const ClothModifierData* clmd);
__host__ __device__ void SIM_cloth_solver_set_volume(const ClothModifierData* clmd);


__host__ __device__ void cloth_calc_pressure_gradient(const ClothModifierData* clmd, const float gradient_vector[3], float* r_vertex_pressure);
__host__ __device__ float cloth_calc_volume(const ClothModifierData* clmd);

__host__ __device__ void SIM_mass_spring_get_position(Implicit_Data* data, int index, float x[3]);
__host__ __device__ float SIM_tri_tetra_volume_signed_6x(Implicit_Data* data, int v1, int v2, int v3);

#endif /* __SIM_MASS_SPRING__ */