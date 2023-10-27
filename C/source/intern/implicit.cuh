#pragma once

#ifndef __CLOTH_IMPLICIT_H__
#define __CLOTH_IMPLICIT_H__

#include "MEM_guardedalloc.cuh"
#include <cuda_runtime_api.h>

#include "collision.h"

#define IMPLICIT_SOLVER_BLENDER

#define CLOTH_ROOT_FRAME /* enable use of root frame coordinate transform */

#define CLOTH_FORCE_GRAVITY
#define CLOTH_FORCE_DRAG
#define CLOTH_FORCE_SPRING_STRUCTURAL
#define CLOTH_FORCE_SPRING_SHEAR
#define CLOTH_FORCE_SPRING_BEND
#define CLOTH_FORCE_SPRING_GOAL
#define CLOTH_FORCE_EFFECTORS

/* DEFINITIONS */
typedef float lfVector[3];
typedef struct fmatrix3x3 {
    float m[3][3];     /* 3x3 matrix */
    uint c, r; /* column and row number */
    // int pinned; /* is this vertex allowed to move? */
    float n1, n2, n3;    /* three normal vectors for collision constrains */
    uint vcount; /* vertex count */
    uint scount; /* spring count */
} fmatrix3x3;

///////////////////////////////////////////////////////////////////
/* simulator start */
///////////////////////////////////////////////////////////////////

typedef struct Implicit_Data {
    /* inputs */
    fmatrix3x3* bigI;        /* identity (constant) */
    fmatrix3x3* tfm;         /* local coordinate transform */
    fmatrix3x3* M;           /* masses */
    lfVector* F;             /* forces */
    fmatrix3x3* dFdV, * dFdX; /* force jacobians */
    int num_blocks;          /* number of off-diagonal blocks (springs) */

    /* motion state data */
    lfVector* X, * Xnew; /* positions */
    lfVector* V, * Vnew; /* velocities */

    /* internal solver data */
    lfVector* B;   /* B for A*dV = B */
    fmatrix3x3* A; /* A for A*dV = B */

    lfVector* dV;         /* velocity change (solution of A*dV = B) */
    lfVector* z;          /* target velocity in constrained directions */
    fmatrix3x3* S;        /* filtering matrix for constraints */
    fmatrix3x3* P, * Pinv; /* pre-conditioning matrix */
} Implicit_Data;


typedef struct ImplicitSolverResult {
  int status;

  int iterations;
  float error;
} ImplicitSolverResult;

__host__ __device__ void SIM_mass_spring_set_vertex_mass(const Implicit_Data *data, int index, float mass);
__host__ __device__ void SIM_mass_spring_set_rest_transform(const Implicit_Data *data, int index, const float tfm[3][3]);
__host__ __device__ void SIM_mass_spring_set_motion_state(Implicit_Data *data, int index, const float x[3],  const float v[3]);
__host__ __device__ void SIM_mass_spring_set_position(Implicit_Data* data, int index, const float x[3]);
__global__ void g_SIM_mass_spring_set_position(Implicit_Data* data, int index, const float x[3]);
__host__ __device__ void SIM_mass_spring_set_velocity(Implicit_Data *data, uint index, const float v[3]);

__host__ __device__  void SIM_mass_spring_get_motion_state(Implicit_Data* data, int index, float x[3], float v[3]);

__host__ __device__ void SIM_mass_spring_get_velocity(Implicit_Data* data, int index, float v[3]);

/* access to modified motion state during solver step */
__host__ __device__ void SIM_mass_spring_get_new_position(const Implicit_Data *data, int index, float x[3]);
__host__ __device__ void SIM_mass_spring_set_new_position(const Implicit_Data *data, int index, const float x[3]);
__host__ __device__ void SIM_mass_spring_get_new_velocity(const Implicit_Data *data, int index, float v[3]);
__host__ __device__ void SIM_mass_spring_set_new_velocity(const Implicit_Data *data, int index, const float v[3]);

__host__ __device__ int SIM_mass_spring_add_block(Implicit_Data* data, int v1, int v2);

__host__ __device__  void SIM_mass_spring_clear_constraints(Implicit_Data* data);
__global__ void g_SIM_mass_spring_clear_constraints(Implicit_Data *data);

__host__ __device__ void SIM_mass_spring_add_constraint_ndof0(Implicit_Data* data, int index, const float dV[3]);
__global__ void g_SIM_mass_spring_add_constraint_ndof0(Implicit_Data *data, int index, const float dV[3]);
__host__ __device__ void SIM_mass_spring_add_constraint_ndof1(Implicit_Data *data, int index, const float c1[3], const float c2[3], const float dV[3]);
__host__ __device__ void SIM_mass_spring_add_constraint_ndof2(Implicit_Data *data, int index, const float c1[3], const float dV[3]);

__host__ __device__ void SIM_mass_spring_solve_velocities(const Implicit_Data* data, float dt, ImplicitSolverResult* result);
__host__ __device__ void SIM_mass_spring_solve_positions(const Implicit_Data* data, float dt);

__host__ __device__ void SIM_mass_spring_apply_result(const Implicit_Data* data);
__global__ void g_SIM_mass_spring_apply_result(const Implicit_Data* data);

/* Clear the force vector at the beginning of the time step */
__host__ __device__ void SIM_mass_spring_clear_forces(Implicit_Data *data);
__global__ void g_SIM_mass_spring_clear_forces(Implicit_Data* data);

/* Fictitious forces introduced by moving coordinate systems */
__host__ __device__ void SIM_mass_spring_force_reference_frame(Implicit_Data *data, int index,  const float acceleration[3], const float omega[3], const float domega_dt[3], float mass);
/* Simple uniform gravity force */

__host__ __device__ void SIM_mass_spring_force_gravity(Implicit_Data *data, int index, float mass, const float g[3]);
__global__ void g_SIM_mass_spring_force_gravity(Implicit_Data* data, int index, float mass, const float g[3]);

/* Global drag force (velocity damping) */
__host__ __device__ void SIM_mass_spring_force_drag(Implicit_Data *data, float drag);

/* Custom external force */
__host__ __device__ void SIM_mass_spring_force_extern(Implicit_Data *data, int i, const float f[3], float dfdx[3][3], float dfdv[3][3]);
/* Wind force, acting on a face (only generates pressure from the normal component) */
__host__ __device__  void SIM_mass_spring_force_face_wind(Implicit_Data* data, int v1, int v2, int v3, const float(*winvec)[3]);
/* Arbitrary per-unit-area vector force field acting on a face. */
__host__ __device__  void SIM_mass_spring_force_face_extern(Implicit_Data *data, int v1, int v2, int v3, const float (*forcevec)[3]);
/* Wind force, acting on an edge */
void SIM_mass_spring_force_edge_wind(Implicit_Data *data,
                                     int v1,
                                     int v2,
                                     float radius1,
                                     float radius2,
                                     const float (*winvec)[3]);
/* Wind force, acting on a vertex */
__global__ void SIM_mass_spring_force_vertex_wind(Implicit_Data *data,
                                       int v,
                                       float radius,
                                       const float (*winvec)[3]);
/* Linear spring force between two points */
__host__ __device__  bool SIM_mass_spring_force_spring_linear(Implicit_Data *data,
                                         int i,
                                         int j,
                                         float restlen,
                                         float stiffness_tension,
                                         float damping_tension,
                                         float stiffness_compression,
                                         float damping_compression,
                                         bool resist_compress,
                                         bool new_compress,
                                         float clamp_force);
/* Angular spring force between two polygons */
__host__ __device__  bool SIM_mass_spring_force_spring_angular(Implicit_Data *data,
                                                     int i,
                                                     int j,
                                                     const int *i_a,
                                                     const int *i_b,
                                                     int len_a,
                                                     int len_b,
                                                     float restang,
                                                     float stiffness,
                                                     float damping);
/* Bending force, forming a triangle at the base of two structural springs */
__host__ __device__ bool SIM_mass_spring_force_spring_bending(Implicit_Data* data, int i, int j, float restlen, float kb, float cb);
__global__ void g_SIM_mass_spring_force_spring_bending(Implicit_Data *data, int i, int j, float restlen, float kb, float cb);
/* Angular bending force based on local target vectors */
bool SIM_mass_spring_force_spring_bending_hair(Implicit_Data *data,
                                               int i,
                                               int j,
                                               int k,
                                               const float target[3],
                                               float stiffness,
                                               float damping);
/* Global goal spring */
__host__ __device__ bool SIM_mass_spring_force_spring_goal(Implicit_Data *data, int i, const float goal_x[3], const float goal_v[3], float stiffness, float damping);
__global__ void g_SIM_mass_spring_force_spring_goal(Implicit_Data* data, int i, const float goal_x[3], const float goal_v[3], float stiffness, float damping);

__host__ __device__ float SIM_tri_area(Implicit_Data *data, int v1, int v2, int v3);

__host__ __device__ void SIM_mass_spring_force_pressure(Implicit_Data* data, int v1, int v2, int v3, float common_pressure, const float* vertex_pressure, const float weights[3]);

__host__ __device__ void root_to_world_v3(const Implicit_Data* data, int index, float r[3], const float v[3]);
/* ======== Hair Volumetric Forces ======== */

struct HairGrid;

#define MAX_HAIR_GRID_RES 256

HairGrid *SIM_hair_volume_create_vertex_grid(float cellsize,
                                             const float gmin[3],
                                             const float gmax[3]);
void SIM_hair_volume_free_vertex_grid(HairGrid *grid);
void SIM_hair_volume_grid_geometry(
	HairGrid *grid, float *cellsize, int res[3], float gmin[3], float gmax[3]);

void SIM_hair_volume_grid_clear(HairGrid *grid);
void SIM_hair_volume_add_vertex(HairGrid *grid, const float x[3], const float v[3]);
void SIM_hair_volume_add_segment(HairGrid *grid,
                                 const float x1[3],
                                 const float v1[3],
                                 const float x2[3],
                                 const float v2[3],
                                 const float x3[3],
                                 const float v3[3],
                                 const float x4[3],
                                 const float v4[3],
                                 const float dir1[3],
                                 const float dir2[3],
                                 const float dir3[3]);

void SIM_hair_volume_normalize_vertex_grid(HairGrid *grid);

bool SIM_hair_volume_solve_divergence(HairGrid *grid,
                                      float dt,
                                      float target_density,
                                      float target_strength);

void SIM_hair_volume_grid_interpolate(HairGrid *grid,
                                      const float x[3],
                                      float *density,
                                      float velocity[3],
                                      float velocity_smooth[3],
                                      float density_gradient[3],
                                      float velocity_gradient[3][3]);

/* Effect of fluid simulation grid on velocities.
 * fluid_factor controls blending between PIC (Particle-in-Cell)
 *     and FLIP (Fluid-Implicit-Particle) methods (0 = only PIC, 1 = only FLIP)
 */
void SIM_hair_volume_grid_velocity(HairGrid *grid, const float x[3], const float v[3], float fluid_factor, float r_v[3]);

/* XXX Warning: expressing grid effects on velocity as a force is not very stable,
 * due to discontinuities in interpolated values!
 * Better use hybrid approaches such as described in
 * "Detail Preserving Continuum Simulation of Straight Hair"
 * (McAdams, Selle 2009)
 */
void SIM_hair_volume_vertex_grid_forces(HairGrid *grid,
                                        const float x[3],
                                        const float v[3],
                                        float smoothfac,
                                        float pressurefac,
                                        float minpressure,
                                        float f[3],
                                        float dfdx[3][3],
                                        float dfdv[3][3]);

// Функция для копирования типа lfVector
inline float* copyLfVectorArrayToDevice(const float h_ptr[3])
{
    float* d_ptr;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_ptr), sizeof(float[3])))
    gpuErrchk(cudaMemcpy(d_ptr, h_ptr, sizeof(float[3]), cudaMemcpyHostToDevice))
    return d_ptr;
}

// Функция для копирования типа fmatrix3x3
inline fmatrix3x3* copyFmatrix3x3ArrayToDevice(const fmatrix3x3* h_ptr)
{
    fmatrix3x3* d_ptr;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_ptr), sizeof(fmatrix3x3)))
    gpuErrchk(cudaMemcpy(d_ptr, h_ptr, sizeof(fmatrix3x3), cudaMemcpyHostToDevice))
    return d_ptr;
}

// Функция для копирования массивов типа fmatrix3x3
inline Implicit_Data* copyImplicit_DataToDevice(Implicit_Data* data, const uint count)
{
    Implicit_Data* d_data;
    Implicit_Data h_data;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(Implicit_Data)))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.X), sizeof(lfVector) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.Xnew), sizeof(lfVector) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.F), sizeof(lfVector) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.V), sizeof(lfVector) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.Vnew), sizeof(lfVector) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.dV), sizeof(lfVector) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.z), sizeof(lfVector) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.B), sizeof(lfVector) * count))

    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.bigI), sizeof(fmatrix3x3) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.tfm), sizeof(fmatrix3x3) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.M), sizeof(fmatrix3x3) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.dFdV), sizeof(fmatrix3x3) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.dFdX), sizeof(fmatrix3x3) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.S), sizeof(fmatrix3x3) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.A), sizeof(fmatrix3x3) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.P), sizeof(fmatrix3x3) * count))
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&h_data.Pinv), sizeof(fmatrix3x3) * count))

    auto** tempX =     new float* [count];  // временный хостовый массив указателей
    auto** tempXnew =  new float* [count];  // временный хостовый массив указателей
    auto** tempF =     new float* [count];  // временный хостовый массив указателей
    auto** tempV =     new float* [count];  // временный хостовый массив указателей
    auto** tempVnew =  new float* [count];  // временный хостовый массив указателей
    auto** tempdV =    new float* [count];  // временный хостовый массив указателей
    auto** tempz =     new float* [count];  // временный хостовый массив указателей
    auto** tempB =     new float* [count];  // временный хостовый массив указателей

    auto** tempbigI = new fmatrix3x3 * [count];  // временный хостовый массив указателей
    auto** temptfm =  new fmatrix3x3 * [count];  // временный хостовый массив указателей
    auto** tempM =    new fmatrix3x3 * [count];  // временный хостовый массив указателей
    auto** tempdFdV = new fmatrix3x3 * [count];  // временный хостовый массив указателей
    auto** tempdFdX = new fmatrix3x3 * [count];  // временный хостовый массив указателей
    auto** tempPinv = new fmatrix3x3 * [count];  // временный хостовый массив указателей
    auto** tempP =    new fmatrix3x3 * [count];  // временный хостовый массив указателей
    auto** tempS =    new fmatrix3x3 * [count];  // временный хостовый массив указателей
    auto** tempA =    new fmatrix3x3 * [count];  // временный хостовый массив указателей
    for (uint i = 0; i < count; ++i)
    {
        tempX[i] = copyLfVectorArrayToDevice(data->X[i]);
        tempXnew[i] = copyLfVectorArrayToDevice(data->Xnew[i]);
        tempF[i] = copyLfVectorArrayToDevice(data->F[i]);
        tempV[i] = copyLfVectorArrayToDevice(data->V[i]);
        tempVnew[i] = copyLfVectorArrayToDevice(data->Vnew[i]);
        tempdV[i] = copyLfVectorArrayToDevice(data->dV[i]);
        tempz[i] = copyLfVectorArrayToDevice(data->z[i]);
        tempB[i] = copyLfVectorArrayToDevice(data->B[i]);

        tempbigI[i] = copyFmatrix3x3ArrayToDevice(&data->bigI[i]);
        temptfm[i] = copyFmatrix3x3ArrayToDevice(&data->tfm[i]);
        tempM[i] = copyFmatrix3x3ArrayToDevice(&data->M[i]);
        tempdFdV[i] = copyFmatrix3x3ArrayToDevice(&data->dFdV[i]);
        tempdFdX[i] = copyFmatrix3x3ArrayToDevice(&data->dFdX[i]);
        tempS[i] = copyFmatrix3x3ArrayToDevice(&data->S[i]);
        tempA[i] = copyFmatrix3x3ArrayToDevice(&data->A[i]);
        tempP[i] = copyFmatrix3x3ArrayToDevice(&data->P[i]);
        tempPinv[i] = copyFmatrix3x3ArrayToDevice(&data->Pinv[i]);
    }

    gpuErrchk(cudaMemcpy(h_data.X, tempX, sizeof(float) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.Xnew, tempXnew, sizeof(float) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.F, tempF, sizeof(float) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.V, tempV, sizeof(float) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.Vnew, tempVnew, sizeof(float) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.dV, tempdV, sizeof(float) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.z, tempz, sizeof(float) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.B, tempB, sizeof(float) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.bigI, tempbigI, sizeof(fmatrix3x3) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.tfm, temptfm, sizeof(fmatrix3x3) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.M, tempM, sizeof(fmatrix3x3) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.dFdV, tempdFdV, sizeof(fmatrix3x3) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.dFdX, tempdFdX, sizeof(fmatrix3x3) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.S, tempS, sizeof(fmatrix3x3) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.A, tempA, sizeof(fmatrix3x3) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.P, tempP, sizeof(fmatrix3x3) * count, cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(h_data.Pinv, tempPinv, sizeof(fmatrix3x3) * count, cudaMemcpyHostToDevice))

    gpuErrchk(cudaMemcpy(d_data, &h_data, sizeof(Implicit_Data), cudaMemcpyHostToDevice))

    delete[] tempX;
    delete[] tempXnew;
    delete[] tempF;
    delete[] tempV;
    delete[] tempVnew;
    delete[] tempz;
    delete[] tempB;
    delete[] tempbigI;
    delete[] temptfm;
    delete[] tempM;
    delete[] tempdFdV;
    delete[] tempdFdX;
    delete[] tempS;
    delete[] tempA;
    delete[] tempP;
    delete[] tempPinv;

    return d_data;
}


inline void freeImplicitDataOnDevice(Implicit_Data* d_data)
{
    cudaFree(d_data->X);
    cudaFree(d_data->Xnew);
    cudaFree(d_data->F);
    cudaFree(d_data->V);
    cudaFree(d_data->Vnew);
    cudaFree(d_data->dV);
    cudaFree(d_data->z);
    cudaFree(d_data->B);
    cudaFree(d_data->bigI);
    cudaFree(d_data->tfm);
    cudaFree(d_data->M);
    cudaFree(d_data->dFdV);
    cudaFree(d_data->dFdX);
    cudaFree(d_data->S);
    cudaFree(d_data->A);
    cudaFree(d_data->P);
    cudaFree(d_data->Pinv);
    cudaFree(d_data);
}



#endif // __CLOTH_IMPLICIT_H__