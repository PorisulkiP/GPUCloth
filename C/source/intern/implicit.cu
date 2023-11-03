#include "implicit.cuh"

#ifdef IMPLICIT_SOLVER_BLENDER

#include "MEM_guardedalloc.cuh"

#include "meshdata_types.cuh"
#include "object_force_types.cuh"
#include <cuda/atomic>

#include "cloth.h"

#include "SIM_mass_spring.cuh"

constexpr float I[3][3] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };
constexpr float ZERO[3][3] = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };

struct Cloth;

//////////////////////////////////////////
/* fast vector / matrix library, enhancements are welcome :) -dg */
/////////////////////////////////////////


///////////////////////////
/* float[3] vector */
///////////////////////////
/* simple vector code */
/* STATUS: verified */
__host__ __device__ void mul_fvector_S(float to[3], const float from[3], const float scalar)
{
    to[0] = from[0] * scalar;
    to[1] = from[1] * scalar;
    to[2] = from[2] * scalar;
}
/* simple v^T * v product ("outer product") */
/* STATUS: HAS TO BE verified (*should* work) */
__host__ __device__ void mul_fvectorT_fvector(float to[3][3], const float vectorA[3], const float vectorB[3])
{
    mul_fvector_S(to[0], vectorB, vectorA[0]);
    mul_fvector_S(to[1], vectorB, vectorA[1]);
    mul_fvector_S(to[2], vectorB, vectorA[2]);
}
/* simple v^T * v product with scalar ("outer product") */
/* STATUS: HAS TO BE verified (*should* work) */
__host__ __device__ void mul_fvectorT_fvectorS(float to[3][3], float vectorA[3], float vectorB[3], const float aS)
{
    mul_fvectorT_fvector(to, vectorA, vectorB);

    mul_fvector_S(to[0], to[0], aS);
    mul_fvector_S(to[1], to[1], aS);
    mul_fvector_S(to[2], to[2], aS);
}

/* create long vector */
__host__ __device__ lfVector* create_lfvector(const uint verts)
{
#ifdef __CUDA_ARCH__
    return static_cast<lfVector*>((lfVector*)malloc(verts * sizeof(lfVector)));
#else
    return static_cast<lfVector*>(MEM_lockfree_callocN(verts * sizeof(lfVector), "cloth_implicit_alloc_vector"));
#endif
}

/* delete long vector */
__host__ __device__ void del_lfvector(float(*fLongVector)[3])
{
    if (fLongVector) 
    {
        MEM_lockfree_freeN(fLongVector);
        fLongVector = nullptr;
    }
}
/* copy long vector */
__host__ __device__ void cp_lfvector(float(*to)[3], const float(*from)[3], const uint verts)
{
#ifdef __CUDA_ARCH__
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < verts) 
    {
        to[idx][0] = from[idx][0];
        to[idx][1] = from[idx][1];
        to[idx][2] = from[idx][2];
    }
#else
    memcpy(to, from, verts * sizeof(lfVector));
#endif
}

/* init long vector with float[3] */
__host__ __device__ void init_lfvector(float(*fLongVector)[3], const float vector[3], const uint verts)
{
#ifdef __CUDA_ARCH__
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < verts)
    {
        copy_v3_v3(fLongVector[i], vector);
    }
#else
    for (uint i = 0; i < verts; ++i)
    {
        copy_v3_v3(fLongVector[i], vector);
    }
#endif
}
/* zero long vector with float[3] */
__host__ __device__ void zero_lfvector(float(*to)[3], const uint verts)
{
#ifdef __CUDA_ARCH__
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < verts) 
    {
        to[idx][0] = 0.0f;
        to[idx][1] = 0.0f;
        to[idx][2] = 0.0f;
    }
#else
    memset(to, 0, verts * sizeof(lfVector));
#endif
}

__global__ void g_zero_lfvector(float(*to)[3], const uint verts)
{
#ifdef __CUDA_ARCH__
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < verts) 
    {
        to[idx][0] = 0.0f;
        to[idx][1] = 0.0f;
        to[idx][2] = 0.0f;
    }
#endif
}
/* Multiply long vector with scalar. */
__host__ __device__ void mul_lfvectorS(float(*to)[3], const float(*fLongVector)[3], const float scalar, const uint verts)
{
#ifdef __CUDA_ARCH__
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < verts)
    {
        mul_fvector_S(to[i], fLongVector[i], scalar);
    }
#else
    for (uint i = 0; i < verts; i++)
    {
        mul_fvector_S(to[i], fLongVector[i], scalar);
    }
#endif
}
/* Multiply long vector with scalar.
 * `A -= B * float` */
__host__ __device__ void submul_lfvectorS(float(*to)[3],  const float(*fLongVector)[3], const float scalar, const uint verts)
{
#ifdef __CUDA_ARCH__
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < verts)
    {
        VECSUBMUL(to[i], fLongVector[i], scalar);
    }
#else
    for (uint i = 0; i < verts; i++)
    {
        VECSUBMUL(to[i], fLongVector[i], scalar);
	}
#endif
}

/* dot product for big vector */
__host__ __device__ float dot_lfvector(const float(*fLongVectorA)[3], const float(*fLongVectorB)[3], const uint verts)
{
#ifdef __CUDA_ARCH__
    __shared__ float temp;

    if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
        temp = 0.0f;
    }
    __syncthreads();

    const uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < verts)
    {
        float local_dot = dot_v3v3(fLongVectorA[i], fLongVectorB[i]);
        atomicAdd(&temp, local_dot);
    }
    __syncthreads();
#else
    float temp = 0.f;
    for (uint64_t i = 0; i < static_cast<uint64_t>(verts); i++)
    {
        temp += dot_v3v3(fLongVectorA[i], fLongVectorB[i]);
    }
#endif

    return temp;
}

/* `A = B + C` -> for big vector. */
__host__ __device__ void add_lfvector_lfvector(float(*to)[3], const float(*fLongVectorA)[3], const float(*fLongVectorB)[3], const uint verts)
{
#ifdef __CUDA_ARCH__
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < verts)
    {
        add_v3_v3v3(to[i], fLongVectorA[i], fLongVectorB[i]);
    }
#else
    for (uint i = 0; i < verts; ++i) 
    {
        add_v3_v3v3(to[i], fLongVectorA[i], fLongVectorB[i]);
    }
#endif    
}
/* `A = B + C * float` -> for big vector. */
__host__ __device__ void add_lfvector_lfvectorS(float(*to)[3], const float(*fLongVectorA)[3], const float(*fLongVectorB)[3], const float bS, const uint verts)
{
#ifdef __CUDA_ARCH__
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < verts)
    {
        VECADDS(to[i], fLongVectorA[i], fLongVectorB[i], bS);
    }
#else
    for (uint i = 0; i < verts; ++i)
    {
        VECADDS(to[i], fLongVectorA[i], fLongVectorB[i], bS);
    }
#endif
}
/* `A = B * float + C * float` -> for big vector */
__host__ __device__ void add_lfvectorS_lfvectorS(float(*to)[3], const float(*fLongVectorA)[3], const float aS, const float(*fLongVectorB)[3], const float bS, const uint verts)
{
#ifdef __CUDA_ARCH__
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < verts)
    {
        VECADDS(to[i], fLongVectorA[i], fLongVectorB[i], bS);
    }
#else
    for (uint i = 0; i < verts; ++i)
    {
        VECADDS(to[i], fLongVectorA[i], fLongVectorB[i], bS);
    }
#endif
}
/* `A = B - C * float` -> for big vector. */
__host__ __device__ void sub_lfvector_lfvectorS(float(*to)[3], const float(*fLongVectorA)[3], const float(*fLongVectorB)[3], const float bS, const uint verts)
{
#ifdef __CUDA_ARCH__
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < verts)
    {
        VECSUBS(to[i], fLongVectorA[i], fLongVectorB[i], bS);
    }
#else
    for (uint i = 0; i < verts; ++i)
    {
        VECSUBS(to[i], fLongVectorA[i], fLongVectorB[i], bS);
    }
#endif
}
/* `A = B - C` -> for big vector. */
__host__ __device__ void sub_lfvector_lfvector(float(*to)[3], const float(*fLongVectorA)[3], const float(*fLongVectorB)[3], const uint verts)
{
#ifdef __CUDA_ARCH__
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < verts) 
    {
        sub_v3_v3v3(to[idx], fLongVectorA[idx], fLongVectorB[idx]);
    }
#else
    for (uint i = 0; i < verts; i++) 
    {
        sub_v3_v3v3(to[i], fLongVectorA[i], fLongVectorB[i]);
    }
#endif
}
///////////////////////////
// 3x3 matrix
///////////////////////////

/* copy 3x3 matrix */
__host__ __device__ void cp_fmatrix(float to[3][3], const float from[3][3])
{
    // memcpy(to, from, sizeof(float[3][3]));
    copy_v3_v3(to[0], from[0]);
    copy_v3_v3(to[1], from[1]);
    copy_v3_v3(to[2], from[2]);
}

/* copy 3x3 matrix */
__host__ __device__ void initdiag_fmatrixS(float to[3][3], const float aS)
{
    constexpr float ZERO[3][3] = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };
    cp_fmatrix(to, ZERO);

    to[0][0] = aS;
    to[1][1] = aS;
    to[2][2] = aS;
}

/* 3x3 matrix multiplied by a scalar */
/* STATUS: verified */
__host__ __device__ void mul_fmatrix_S(float matrix[3][3], const float scalar)
{
    mul_fvector_S(matrix[0], matrix[0], scalar);
    mul_fvector_S(matrix[1], matrix[1], scalar);
    mul_fvector_S(matrix[2], matrix[2], scalar);
}

/* a vector multiplied by a 3x3 matrix */
/* STATUS: verified */
__host__ __device__ void mul_fvector_fmatrix(float* to, const float* from, const float matrix[3][3])
{
    to[0] = matrix[0][0] * from[0] + matrix[1][0] * from[1] + matrix[2][0] * from[2];
    to[1] = matrix[0][1] * from[0] + matrix[1][1] * from[1] + matrix[2][1] * from[2];
    to[2] = matrix[0][2] * from[0] + matrix[1][2] * from[1] + matrix[2][2] * from[2];
}

/* 3x3 matrix multiplied by a vector */
/* STATUS: verified */
__host__ __device__ void mul_fmatrix_fvector(float* to, const float matrix[3][3], const float from[3])
{
    to[0] = dot_v3v3(matrix[0], from);
    to[1] = dot_v3v3(matrix[1], from);
    to[2] = dot_v3v3(matrix[2], from);
}
/* 3x3 matrix addition with 3x3 matrix */
__host__ __device__ void add_fmatrix_fmatrix(float to[3][3],
    const float matrixA[3][3],
    const float matrixB[3][3])
{
    add_v3_v3v3(to[0], matrixA[0], matrixB[0]);
    add_v3_v3v3(to[1], matrixA[1], matrixB[1]);
    add_v3_v3v3(to[2], matrixA[2], matrixB[2]);
}
/* `A -= B*x + (C * y)` (3x3 matrix sub-addition with 3x3 matrix). */
__host__ __device__ void subadd_fmatrixS_fmatrixS(
    float to[3][3], const float matrixA[3][3], const float aS, const float matrixB[3][3], const float bS)
{
    VECSUBADDSS(to[0], matrixA[0], aS, matrixB[0], bS);
    VECSUBADDSS(to[1], matrixA[1], aS, matrixB[1], bS);
    VECSUBADDSS(to[2], matrixA[2], aS, matrixB[2], bS);
}
/* `A = B - C` (3x3 matrix subtraction with 3x3 matrix). */
__host__ __device__ void sub_fmatrix_fmatrix(float to[3][3],
    const float matrixA[3][3],
    const float matrixB[3][3])
{
    sub_v3_v3v3(to[0], matrixA[0], matrixB[0]);
    sub_v3_v3v3(to[1], matrixA[1], matrixB[1]);
    sub_v3_v3v3(to[2], matrixA[2], matrixB[2]);
}
/////////////////////////////////////////////////////////////////
/* special functions */
/////////////////////////////////////////////////////////////////
/* 3x3 matrix multiplied+added by a vector */
/* STATUS: verified */
__host__ __device__ void muladd_fmatrix_fvector(float to[3], const float matrix[3][3], const float from[3])
{
    to[0] += dot_v3v3(matrix[0], from);
    to[1] += dot_v3v3(matrix[1], from);
    to[2] += dot_v3v3(matrix[2], from);
}

__host__ __device__ void muladd_fmatrixT_fvector(float to[3], const float matrix[3][3], const float from[3])
{
    to[0] += matrix[0][0] * from[0] + matrix[1][0] * from[1] + matrix[2][0] * from[2];
    to[1] += matrix[0][1] * from[0] + matrix[1][1] * from[1] + matrix[2][1] * from[2];
    to[2] += matrix[0][2] * from[0] + matrix[1][2] * from[1] + matrix[2][2] * from[2];
}

__host__ __device__ void outerproduct(float r[3][3], const float a[3], const float b[3])
{
#ifdef __CUDA_ARCH__
    const uint idx = threadIdx.x;
    if (idx < 3)
    {
        mul_v3_v3fl(r[idx], a, b[idx]);
    }
#else
    mul_v3_v3fl(r[0], a, b[0]);
    mul_v3_v3fl(r[1], a, b[1]);
    mul_v3_v3fl(r[2], a, b[2]);
#endif
}

__host__ __device__ void cross_m3_v3m3(float r[3][3], const float v[3], const float m[3][3])
{
    cross_v3_v3v3(r[0], v, m[0]);
    cross_v3_v3v3(r[1], v, m[1]);
    cross_v3_v3v3(r[2], v, m[2]);
}

__host__ __device__ void cross_v3_identity(float r[3][3], const float v[3])
{
    r[0][0] = 0.0f;
    r[1][0] = v[2];
    r[2][0] = -v[1];
    r[0][1] = -v[2];
    r[1][1] = 0.0f;
    r[2][1] = v[0];
    r[0][2] = v[1];
    r[1][2] = -v[0];
    r[2][2] = 0.0f;
}

__host__ __device__ void madd_m3_m3fl(float r[3][3], const float m[3][3], const float f)
{
    r[0][0] += m[0][0] * f;
    r[0][1] += m[0][1] * f;
    r[0][2] += m[0][2] * f;
    r[1][0] += m[1][0] * f;
    r[1][1] += m[1][1] * f;
    r[1][2] += m[1][2] * f;
    r[2][0] += m[2][0] * f;
    r[2][1] += m[2][1] * f;
    r[2][2] += m[2][2] * f;
}

/////////////////////////////////////////////////////////////////

///////////////////////////
/* SPARSE SYMMETRIC big matrix with 3x3 matrix entries */
///////////////////////////
/* printf a big matrix on console: for debug output */
#  if 0
static void print_bfmatrix(fmatrix3x3* m3)
{
    uint i = 0;

    for (i = 0; i < m3[0].vcount + m3[0].scount; i++) {
        print_fmatrix(m3[i].m);
    }
}
#  endif

__host__ __device__ void init_fmatrix(fmatrix3x3* matrix, const int r, const int c)
{
    matrix->r = r;
    matrix->c = c;
}

__global__ void g_init_fmatrix(fmatrix3x3* matrix, const int r, const int c)
{
    matrix->r = r;
    matrix->c = c;
}

/* create big matrix */
__host__ fmatrix3x3* create_bfmatrix(const uint verts, const uint springs)
{
    auto* temp = static_cast<fmatrix3x3*>(MEM_lockfree_callocN(sizeof(fmatrix3x3) * (verts + springs), "cloth_implicit_alloc_matrix"));
    temp[0].vcount = verts;
    temp[0].scount = springs;
    for (uint i = 0; i < verts; ++i)
    {
        init_fmatrix(temp + i, i, i);
    }
    return temp;
}

/* delete big matrix */
__host__ __device__ void del_bfmatrix(fmatrix3x3* matrix)
{
    if (matrix) 
    {
        MEM_lockfree_freeN(matrix);
    }
}

/* copy big matrix */
__host__ __device__ void cp_bfmatrix(fmatrix3x3* to, const fmatrix3x3* from)
{
    /* TODO: bounds checking. */
#ifdef __CUDA_ARCH__
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint verts = (from[0].vcount + from[0].scount);
    if (idx < verts) 
    {
        to[idx].m[0][0] = from[idx].m[0][0];
        to[idx].m[0][1] = from[idx].m[0][1];
        to[idx].m[0][2] = from[idx].m[0][2];
        to[idx].n1 = from[idx].n1;
        to[idx].n2 = from[idx].n2;
        to[idx].n3 = from[idx].n3;
        to[idx].c = from[idx].c;
        to[idx].r = from[idx].r;
        to[idx].vcount = from[idx].vcount;
        to[idx].scount = from[idx].scount;
    }
#else
    memcpy(to, from, sizeof(fmatrix3x3) * (from[0].vcount + from[0].scount));
#endif
}

/* init big matrix */
/* slow in parallel */
__host__ __device__ void init_bfmatrix(fmatrix3x3* matrix, const float m3[3][3])
{
#ifdef __CUDA_ARCH__
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint numverts = matrix[0].vcount + matrix[0].scount;

    if (i < numverts)
    {
        cp_fmatrix(matrix[i].m, m3);
    }
#else
    for (uint i = 0; i < matrix[0].vcount + matrix[0].scount; ++i)
    {
        cp_fmatrix(matrix[i].m, m3);
    }
#endif
}

__global__ void g_init_bfmatrix(fmatrix3x3* matrix, const float m3[3][3])
{
#ifdef __CUDA_ARCH__
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint numverts = matrix[0].vcount + matrix[0].scount;

    if (i < numverts)
    {
        cp_fmatrix(matrix[i].m, m3);
    }
#endif
}

/* init the diagonal of big matrix */
/* slow in parallel */
__host__ void initdiag_bfmatrix(fmatrix3x3* matrix, const float m3[3][3])
{
	constexpr float tmatrix[3][3] = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };

    for (uint i = 0; i < matrix[0].vcount; i++) 
    {
        cp_fmatrix(matrix[i].m, m3);
    }
    for (uint j = matrix[0].vcount; j < matrix[0].vcount + matrix[0].scount; j++) 
    {
        cp_fmatrix(matrix[j].m, tmatrix);
    }
}

/* SPARSE SYMMETRIC multiply big matrix with long vector. */
/* STATUS: verified */
__host__ __device__ void mul_bfmatrix_lfvector(float(*to)[3], const fmatrix3x3* from, const lfVector* fLongVector)
{
#ifdef __CUDA_ARCH__
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint vcount = from[0].vcount;
    const uint numverts = from[0].vcount + from[0].scount;
    lfVector* tmp = (lfVector*)malloc(numverts * sizeof(lfVector));

    zero_lfvector(to, vcount);

    if (i < numverts)
    {
        muladd_fmatrix_fvector(tmp[from[i].r], from[i].m, fLongVector[from[i].c]);
    }
    add_lfvector_lfvector(to, to, tmp, vcount);
    if (tmp)
    {
        free(tmp);
        tmp = nullptr;
	}
#else
    const uint vcount = from[0].vcount;
    lfVector* tmp = create_lfvector(vcount);

    zero_lfvector(to, vcount);

#  pragma omp parallel sections if (vcount > CLOTH_OPENMP_LIMIT)
    {
#  pragma omp section
        {
            for (uint i = from[0].vcount; i < from[0].vcount + from[0].scount; i++) {
                /* This is the lower triangle of the sparse matrix,
                 * therefore multiplication occurs with transposed submatrices. */
                muladd_fmatrixT_fvector(to[from[i].c], from[i].m, fLongVector[from[i].r]);
            }
        }
#  pragma omp section
        {
            for (uint i = 0; i < from[0].vcount + from[0].scount; i++) {
                muladd_fmatrix_fvector(tmp[from[i].r], from[i].m, fLongVector[from[i].c]);
            }
        }
    }
    add_lfvector_lfvector(to, to, tmp, from[0].vcount);

    del_lfvector(tmp);
#endif
}

/* SPARSE SYMMETRIC sub big matrix with big matrix. */
/* A -= B * float + C * float --> for big matrix */
/* VERIFIED */
__host__ __device__ void subadd_bfmatrixS_bfmatrixS(fmatrix3x3* to, const fmatrix3x3* from, const float aS, const fmatrix3x3* matrix, const float bS)
{
#ifdef __CUDA_ARCH__
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint numverts = matrix[0].vcount + matrix[0].scount;

    if (i < numverts)
    {
        subadd_fmatrixS_fmatrixS(to[i].m, from[i].m, aS, matrix[i].m, bS);
    }
#else
    /* process diagonal elements */
    for (uint i = 0; i < matrix[0].vcount + matrix[0].scount; i++)
    {
        subadd_fmatrixS_fmatrixS(to[i].m, from[i].m, aS, matrix[i].m, bS);
    }
#endif
}

__host__ Implicit_Data* SIM_mass_spring_solver_create(const uint numverts, const uint numsprings)
{
    constexpr float I[3][3] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };
	auto* id = static_cast<Implicit_Data*>(MEM_lockfree_callocN(sizeof(Implicit_Data), "implicit vecmat"));

    /* process diagonal elements */
    id->tfm = create_bfmatrix(numverts, 0);
    id->A = create_bfmatrix(numverts, numsprings);
    id->dFdV = create_bfmatrix(numverts, numsprings);
    id->dFdX = create_bfmatrix(numverts, numsprings);
    id->S = create_bfmatrix(numverts, 0);
    id->Pinv = create_bfmatrix(numverts, numsprings);
    id->P = create_bfmatrix(numverts, numsprings);
    id->bigI = create_bfmatrix(numverts, numsprings); /* TODO: 0 springs. */
    id->M = create_bfmatrix(numverts, numsprings);
    id->X = create_lfvector(numverts);
    id->Xnew = create_lfvector(numverts);
    id->V = create_lfvector(numverts);
    id->Vnew = create_lfvector(numverts);
    id->F = create_lfvector(numverts);
    id->B = create_lfvector(numverts);
    id->dV = create_lfvector(numverts);
    id->z = create_lfvector(numverts);

    initdiag_bfmatrix(id->bigI, I);

    return id;
}

__host__ __device__ void SIM_mass_spring_solver_free(Implicit_Data* id)
{
    del_bfmatrix(id->tfm);
    del_bfmatrix(id->A);
    del_bfmatrix(id->dFdV);
    del_bfmatrix(id->dFdX);
    del_bfmatrix(id->S);
    del_bfmatrix(id->P);
    del_bfmatrix(id->Pinv);
    del_bfmatrix(id->bigI);
    del_bfmatrix(id->M);

    del_lfvector(id->X);
    del_lfvector(id->Xnew);
    del_lfvector(id->V);
    del_lfvector(id->Vnew);
    del_lfvector(id->F);
    del_lfvector(id->B);
    del_lfvector(id->dV);
    del_lfvector(id->z);

    MEM_lockfree_freeN(id);
}

/* ==== Transformation from/to root reference frames ==== */

__host__ __device__ void world_to_root_v3(const Implicit_Data* data, const int index, float r[3], const float v[3])
{
    copy_v3_v3(r, v);
    mul_transposed_m3_v3(data->tfm[index].m, r);
}

__host__ __device__ void root_to_world_v3(const Implicit_Data* data, const int index, float r[3], const float v[3])
{
    mul_v3_m3v3(r, data->tfm[index].m, v);
}

__host__ __device__ void world_to_root_m3(const Implicit_Data* data,
                      const int index,
                      float r[3][3],
                      const float m[3][3])
{
    float trot[3][3];
    copy_m3_m3(trot, data->tfm[index].m);
    transpose_m3(trot);
    mul_m3_m3m3(r, trot, m);
}

__host__ __device__ void root_to_world_m3(const Implicit_Data* data,
                      const int index,
                      float r[3][3],
                      const float m[3][3])
{
    mul_m3_m3m3(r, data->tfm[index].m, m);
}

/* ================================ */

__host__ __device__ void filter(lfVector* V, const fmatrix3x3* S)
{
#ifdef __CUDA_ARCH__
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint numverts = S[0].vcount;

    if (i < numverts)
    {
        mul_m3_v3(S[i].m, V[S[i].r]);
    }
#else
    for (uint i = 0; i < S[0].vcount; ++i)
    {
        mul_m3_v3(S[i].m, V[S[i].r]);
    }
#endif
}

__host__ __device__ static int cg_filtered(lfVector* ldV,
    fmatrix3x3* lA,
    lfVector* lB,
    lfVector* z,
    fmatrix3x3* S,
    ImplicitSolverResult* result)
{
    /* Solves for unknown X in equation AX=B */
    uint conjgrad_loopcount = 0, conjgrad_looplimit = 100;
    constexpr float conjgrad_epsilon = 0.01f;

    uint numverts = lA[0].vcount;
    lfVector* fB = create_lfvector(numverts);
    lfVector* AdV = create_lfvector(numverts);
    lfVector* r = create_lfvector(numverts);
    lfVector* c = create_lfvector(numverts);
    lfVector* q = create_lfvector(numverts);
    lfVector* s = create_lfvector(numverts);
    float delta_old, alpha;

    cp_lfvector(ldV, z, numverts);

    /* d0 = filter(B)^T * P * filter(B) */
    cp_lfvector(fB, lB, numverts);
    filter(fB, S);
    const float bnorm2 = dot_lfvector(fB, fB, numverts);
    const float delta_target = conjgrad_epsilon * conjgrad_epsilon * bnorm2;

    /* r = filter(B - A * dV) */
    mul_bfmatrix_lfvector(AdV, lA, ldV);
    sub_lfvector_lfvector(r, lB, AdV, numverts);
    filter(r, S);

    /* c = filter(P^-1 * r) */
    cp_lfvector(c, r, numverts);
    filter(c, S);

    /* delta = r^T * c */
    float delta_new = dot_lfvector(r, c, numverts);

#  ifdef IMPLICIT_PRINT_SOLVER_INPUT_OUTPUT
    printf("==== A ====\n");
    print_bfmatrix(lA);
    printf("==== z ====\n");
    print_lvector(z, numverts);
    printf("==== B ====\n");
    print_lvector(lB, numverts);
    printf("==== S ====\n");
    print_bfmatrix(S);
#  endif

    while (delta_new > delta_target && conjgrad_loopcount < conjgrad_looplimit) {
        mul_bfmatrix_lfvector(q, lA, c);
        filter(q, S);

        alpha = delta_new / dot_lfvector(c, q, numverts);

        add_lfvector_lfvectorS(ldV, ldV, c, alpha, numverts);

        add_lfvector_lfvectorS(r, r, q, -alpha, numverts);

        /* s = P^-1 * r */
        cp_lfvector(s, r, numverts);
        delta_old = delta_new;
        delta_new = dot_lfvector(r, s, numverts);

        add_lfvector_lfvectorS(c, s, c, delta_new / delta_old, numverts);
        filter(c, S);

        conjgrad_loopcount++;
    }

#  ifdef IMPLICIT_PRINT_SOLVER_INPUT_OUTPUT
    printf("==== dV ====\n");
    print_lvector(ldV, numverts);
    printf("========\n");
#  endif
#ifdef __CUDA_ARCH__
    free(fB);
    free(AdV);
    free(r);
    free(c);
    free(q);
    free(s);
#else
    del_lfvector(fB);
    del_lfvector(AdV);
    del_lfvector(r);
    del_lfvector(c);
    del_lfvector(q);
    del_lfvector(s);
#endif

    // printf("W/O conjgrad_loopcount: %d\n", conjgrad_loopcount);

    result->status = conjgrad_loopcount < conjgrad_looplimit ? SIM_SOLVER_SUCCESS : SIM_SOLVER_NO_CONVERGENCE;
    result->iterations = conjgrad_loopcount;
    result->error = bnorm2 > 0.0f ? sqrtf(delta_new / bnorm2) : 0.0f;

    return conjgrad_loopcount < conjgrad_looplimit; /* true means we reached desired accuracy in given time - ie stable */
}

__host__ __device__ void SIM_mass_spring_solve_velocities(const Implicit_Data* data, const float dt, ImplicitSolverResult* result)
{
    const uint numverts = data->dFdV[0].vcount;
    auto dFdXmV = static_cast<lfVector*>(malloc(numverts * sizeof(lfVector)));
    zero_lfvector(data->dV, numverts);

    cp_bfmatrix(data->A, data->M);

    subadd_bfmatrixS_bfmatrixS(data->A, data->dFdV, dt, data->dFdX, (dt * dt));

    mul_bfmatrix_lfvector(dFdXmV, data->dFdX, data->V);

    add_lfvectorS_lfvectorS(data->B, data->F, dt, dFdXmV, (dt * dt), numverts);

    if (dFdXmV)
    {
        free(dFdXmV);
        dFdXmV = nullptr;
    }

    /* Conjugate gradient algorithm to solve Ax=b. */
    cg_filtered(data->dV, data->A, data->B, data->z, data->S, result);

    /* advance velocities */
    add_lfvector_lfvector(data->Vnew, data->V, data->dV, numverts);
}

__host__ __device__  void SIM_mass_spring_solve_positions(const Implicit_Data* data, const float dt)
{
    const uint numverts = data->M[0].vcount;

    /* advance positions */
    add_lfvector_lfvectorS(data->Xnew, data->X, data->Vnew, dt, numverts);
}

__global__ void g_SIM_mass_spring_apply_result(const Implicit_Data* data)
{
    const uint numverts = data->M[0].vcount;
    cp_lfvector(data->X, data->Xnew, numverts);
    cp_lfvector(data->V, data->Vnew, numverts);
}

__host__ __device__ void SIM_mass_spring_apply_result(const Implicit_Data* data)
{
	const uint numverts = data->M[0].vcount;
    cp_lfvector(data->X, data->Xnew, numverts);
    cp_lfvector(data->V, data->Vnew, numverts);
}

__host__ __device__ void SIM_mass_spring_set_vertex_mass(const Implicit_Data* data, const int index, const float mass)
{
    unit_m3(data->M[index].m);
    mul_m3_fl(data->M[index].m, mass);
}

__host__ __device__ void SIM_mass_spring_set_rest_transform(const Implicit_Data* data, const int index, const float tfm[3][3])
{
    copy_m3_m3(data->tfm[index].m, tfm);
}

__host__ __device__ void SIM_mass_spring_set_motion_state(Implicit_Data* data, const int index, const float x[3], const float v[3])
{
    world_to_root_v3(data, index, data->X[index], x);
    world_to_root_v3(data, index, data->V[index], v);
}

__global__ void g_SIM_mass_spring_set_position(Implicit_Data* data, const int index, const float x[3])
{
    world_to_root_v3(data, index, data->X[index], x);
}
__host__ __device__  void SIM_mass_spring_set_position(Implicit_Data* data, const int index, const float x[3])
{
    world_to_root_v3(data, index, data->X[index], x);
}

__host__ __device__ void SIM_mass_spring_set_velocity(Implicit_Data* data, const uint index, const float v[3])
{
    world_to_root_v3(data, index, data->V[index], v);
}

__host__ __device__ void SIM_mass_spring_get_motion_state(Implicit_Data* data,
                                      const int index,
                                      float x[3],
                                      float v[3])
{
    if (x) {
        root_to_world_v3(data, index, x, data->X[index]);
    }
    if (v) {
        root_to_world_v3(data, index, v, data->V[index]);
    }
}

__host__ __device__ void SIM_mass_spring_get_velocity(const Implicit_Data* data, const int index, float v[3])
{
    root_to_world_v3(data, index, v, data->V[index]);
}

__host__ __device__ void SIM_mass_spring_get_new_position(const Implicit_Data* data, const int index, float x[3])
{
    root_to_world_v3(data, index, x, data->Xnew[index]);
}


__host__ __device__ void SIM_mass_spring_set_new_position(const Implicit_Data* data, const int index, const float x[3])
{
    world_to_root_v3(data, index, data->Xnew[index], x);
}

__host__ __device__ void SIM_mass_spring_get_new_velocity(const Implicit_Data* data, const int index, float v[3])
{
    root_to_world_v3(data, index, v, data->Vnew[index]);
}

__host__ __device__ void SIM_mass_spring_set_new_velocity(const Implicit_Data* data, const int index, const float v[3])
{
    world_to_root_v3(data, index, data->Vnew[index], v);
}

/* -------------------------------- */

__host__ __device__ uint SIM_mass_spring_add_block(Implicit_Data* data, const int v1, const int v2)
{
    const uint s = data->M[0].vcount + data->num_blocks; /* index from array start */
#ifdef __CUDA_ARCH__
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < s)
    {
        atomicAdd(&data->num_blocks, 1);
        /* tfm and S don't have spring entries (diagonal blocks only) */
        init_fmatrix(data->bigI + s, v1, v2);
        init_fmatrix(data->M + s, v1, v2);
        init_fmatrix(data->dFdX + s, v1, v2);
        init_fmatrix(data->dFdV + s, v1, v2);
        init_fmatrix(data->A + s, v1, v2);
        init_fmatrix(data->P + s, v1, v2);
        init_fmatrix(data->Pinv + s, v1, v2);
    }
#else
    BLI_assert(s < data->M[0].vcount + data->M[0].scount);
    ++data->num_blocks;

    /* tfm and S don't have spring entries (diagonal blocks only) */
    init_fmatrix(data->bigI + s, v1, v2);
    init_fmatrix(data->M + s, v1, v2);
    init_fmatrix(data->dFdX + s, v1, v2);
    init_fmatrix(data->dFdV + s, v1, v2);
    init_fmatrix(data->A + s, v1, v2);
    init_fmatrix(data->P + s, v1, v2);
    init_fmatrix(data->Pinv + s, v1, v2);
#endif
    return s;
}

__host__ __device__ void SIM_mass_spring_clear_constraints(Implicit_Data* data)
{
#ifdef __CUDA_ARCH__
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint numverts = data->S[0].vcount;

    if (idx < numverts) 
    {
        unit_m3(data->S[idx].m);
        zero_v3(data->z[idx]);
	}
#else
    const uint numverts = data->S[0].vcount;
    for (uint i = 0; i < numverts; ++i)
    {
        unit_m3(data->S[i].m);
        zero_v3(data->z[i]);
}
#endif
}

__global__ void g_SIM_mass_spring_clear_constraints(Implicit_Data* data)
{
#ifdef __CUDA_ARCH__
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint numverts = data->S[0].vcount;

    if (idx < numverts)
    {
        unit_m3(data->S[idx].m);
        zero_v3(data->z[idx]);
    }
#endif
}

__host__ __device__ void SIM_mass_spring_add_constraint_ndof0(Implicit_Data* data, const int index, const float dV[3])
{
    zero_m3(data->S[index].m);

    world_to_root_v3(data, index, data->z[index], dV);
}

__global__ void g_SIM_mass_spring_add_constraint_ndof0(Implicit_Data* data, const int index, const float dV[3])
{
    zero_m3(data->S[index].m);

    world_to_root_v3(data, index, data->z[index], dV);
}

__host__ __device__ void SIM_mass_spring_add_constraint_ndof1(
    Implicit_Data* data, const int index, const float c1[3], const float c2[3], const float dV[3])
{
    float m[3][3], p[3], q[3], u[3], cmat[3][3];

    constexpr float I[3][3] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };
    world_to_root_v3(data, index, p, c1);
    mul_fvectorT_fvector(cmat, p, p);
    sub_m3_m3m3(m, I, cmat);

    world_to_root_v3(data, index, q, c2);
    mul_fvectorT_fvector(cmat, q, q);
    sub_m3_m3m3(m, m, cmat);

    /* XXX not sure but multiplication should work here */
    copy_m3_m3(data->S[index].m, m);
    //  mul_m3_m3m3(data->S[index].m, data->S[index].m, m);

    world_to_root_v3(data, index, u, dV);
    add_v3_v3(data->z[index], u);
}

__host__ __device__ void SIM_mass_spring_add_constraint_ndof2(Implicit_Data* data,
                                          const int index,
    const float c1[3],
    const float dV[3])
{
    float m[3][3], p[3], u[3], cmat[3][3];

    constexpr float I[3][3] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };
    world_to_root_v3(data, index, p, c1);
    mul_fvectorT_fvector(cmat, p, p);
    sub_m3_m3m3(m, I, cmat);

    copy_m3_m3(data->S[index].m, m);
    //  mul_m3_m3m3(data->S[index].m, data->S[index].m, m);

    world_to_root_v3(data, index, u, dV);
    add_v3_v3(data->z[index], u);
}

__host__ __device__ void SIM_mass_spring_clear_forces(Implicit_Data* data)
{
    constexpr float ZERO[3][3] = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };
    const uint numverts = data->M[0].vcount;
    zero_lfvector(data->F, numverts);
    init_bfmatrix(data->dFdX, ZERO);
    init_bfmatrix(data->dFdV, ZERO);

    data->num_blocks = 0;
}

__global__ void g_SIM_mass_spring_clear_forces(Implicit_Data* data)
{
    constexpr float ZERO[3][3] = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };
	const uint numverts = data->M[0].vcount;
    zero_lfvector(data->F, numverts);
    init_bfmatrix(data->dFdX, ZERO);
    init_bfmatrix(data->dFdV, ZERO);

    data->num_blocks = 0;
}

__host__ __device__ void SIM_mass_spring_force_reference_frame(Implicit_Data* data,
    int index,
    const float acceleration[3],
    const float omega[3],
    const float domega_dt[3],
    const float mass)
{
#  ifdef CLOTH_ROOT_FRAME
    float acc[3], w[3], dwdt[3];
    float f[3], dfdx[3][3], dfdv[3][3];
    float euler[3], coriolis[3], centrifugal[3], rotvel[3];
    float deuler[3][3], dcoriolis[3][3], dcentrifugal[3][3], drotvel[3][3];

    world_to_root_v3(data, index, acc, acceleration);
    world_to_root_v3(data, index, w, omega);
    world_to_root_v3(data, index, dwdt, domega_dt);

    cross_v3_v3v3(euler, dwdt, data->X[index]);
    cross_v3_v3v3(coriolis, w, data->V[index]);
    mul_v3_fl(coriolis, 2.0f);
    cross_v3_v3v3(rotvel, w, data->X[index]);
    cross_v3_v3v3(centrifugal, w, rotvel);

    sub_v3_v3v3(f, acc, euler);
    sub_v3_v3(f, coriolis);
    sub_v3_v3(f, centrifugal);

    mul_v3_fl(f, mass); /* F = m * a */

    cross_v3_identity(deuler, dwdt);
    cross_v3_identity(dcoriolis, w);
    mul_m3_fl(dcoriolis, 2.0f);
    cross_v3_identity(drotvel, w);
    cross_m3_v3m3(dcentrifugal, w, drotvel);

    add_m3_m3m3(dfdx, deuler, dcentrifugal);
    negate_m3(dfdx);
    mul_m3_fl(dfdx, mass);

    copy_m3_m3(dfdv, dcoriolis);
    negate_m3(dfdv);
    mul_m3_fl(dfdv, mass);

    add_v3_v3(data->F[index], f);
    add_m3_m3m3(data->dFdX[index].m, data->dFdX[index].m, dfdx);
    add_m3_m3m3(data->dFdV[index].m, data->dFdV[index].m, dfdv);
#  else
    (void)data;
    (void)index;
    (void)acceleration;
    (void)omega;
    (void)domega_dt;
#  endif
}

void SIM_mass_spring_force_gravity(Implicit_Data* data, const int index, const float mass, const float g[3])
{
    /* force = mass * acceleration (in this case: gravity) */
    float f[3];
    world_to_root_v3(data, index, f, g);
    mul_v3_fl(f, mass);

    add_v3_v3(data->F[index], f);
}

__device__ void d_SIM_mass_spring_force_gravity(Implicit_Data* data, const int index, const float mass, const float g[3])
{
    /* force = mass * acceleration (in this case: gravity) */
    float f[3];
    world_to_root_v3(data, index, f, g);
    mul_v3_fl(f, mass);

    add_v3_v3(data->F[index], f);
}

__host__ __device__ void SIM_mass_spring_force_drag(Implicit_Data* data, const float drag)
{
	const uint numverts = data->M[0].vcount;
    constexpr float I[3][3] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };
#ifdef __CUDA_ARCH__
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numverts)
    {
        float tmp[3][3];
        madd_v3_v3fl(data->F[i], data->V[i], -drag);
        copy_m3_m3(tmp, I);
        mul_m3_fl(tmp, -drag);
        add_m3_m3m3(data->dFdV[i].m, data->dFdV[i].m, tmp);
    }
#else
    for (uint i = 0; i < numverts; i++)
    {
        float tmp[3][3];

        /* NOTE: Uses root space velocity, no need to transform. */
        madd_v3_v3fl(data->F[i], data->V[i], -drag);

        copy_m3_m3(tmp, I);
        mul_m3_fl(tmp, -drag);
        add_m3_m3m3(data->dFdV[i].m, data->dFdV[i].m, tmp);
    }
#endif

}

__host__ __device__ void SIM_mass_spring_force_extern(Implicit_Data* data, const int i, const float f[3], float dfdx[3][3], float dfdv[3][3])
{
    float tf[3], tdfdx[3][3], tdfdv[3][3];
    world_to_root_v3(data, i, tf, f);
    world_to_root_m3(data, i, tdfdx, dfdx);
    world_to_root_m3(data, i, tdfdv, dfdv);

    add_v3_v3(data->F[i], tf);
    add_m3_m3m3(data->dFdX[i].m, data->dFdX[i].m, tdfdx);
    add_m3_m3m3(data->dFdV[i].m, data->dFdV[i].m, tdfdv);
}

__host__ __device__  float calc_nor_area_tri(float nor[3],
    const float v1[3],
    const float v2[3],
    const float v3[3])
{
    float n1[3], n2[3];

    sub_v3_v3v3(n1, v1, v2);
    sub_v3_v3v3(n2, v2, v3);

    cross_v3_v3v3(nor, n1, n2);
    return normalize_v3(nor) / 2.0f;
}

__host__ __device__  void SIM_mass_spring_force_face_wind(Implicit_Data* data, const int v1, const int v2, const int v3, const float(*winvec)[3])
{
    /* XXX does not support force jacobians yet,
     * since the effector system does not provide them either. */

    constexpr float effector_scale = 0.02f;
    const int vs[3] = { v1, v2, v3 };
    float win[3], nor[3];
    float force[3];

    /* calculate face normal and area */
    const float area = calc_nor_area_tri(nor, data->X[v1], data->X[v2], data->X[v3]);
    /* The force is calculated and split up evenly for each of the three face verts */
    const float factor = effector_scale * area / 3.0f;

    /* Calculate wind pressure at each vertex by projecting the wind field on the normal. */
    for (int i = 0; i < 3; i++) 
    {
        world_to_root_v3(data, vs[i], win, winvec[vs[i]]);

        force[i] = dot_v3v3(win, nor);
    }

    /* Compute per-vertex force values from local pressures.
     * From integrating the pressure over the triangle and deriving
     * equivalent vertex forces, it follows that:
     *
     * force[idx] = (sum(pressure) + pressure[idx]) * area / 12
     *
     * Effectively, 1/4 of the pressure acts just on its vertex,
     * while 3/4 is split evenly over all three.
     */
    mul_v3_fl(force, factor / 4.0f);

    const float base_force = force[0] + force[1] + force[2];

    /* add pressure to each of the face verts */
    madd_v3_v3fl(data->F[v1], nor, base_force + force[0]);
    madd_v3_v3fl(data->F[v2], nor, base_force + force[1]);
    madd_v3_v3fl(data->F[v3], nor, base_force + force[2]);
}

__host__ __device__  void SIM_mass_spring_force_face_extern(Implicit_Data* data, const int v1, const int v2, const int v3, const float(*forcevec)[3])
{
	constexpr float effector_scale = 0.02f;
    const int vs[3] = { v1, v2, v3 };
    float nor[3];
    float base_force[3];
    float force[3][3];

    /* calculate face normal and area */
	const float area = calc_nor_area_tri(nor, data->X[v1], data->X[v2], data->X[v3]);
    /* The force is calculated and split up evenly for each of the three face verts */
	const float factor = effector_scale * area / 3.0f;

    /* Compute common and per-vertex force vectors from the original inputs. */
    zero_v3(base_force);

    for (int i = 0; i < 3; i++) 
    {
        world_to_root_v3(data, vs[i], force[i], forcevec[vs[i]]);

        mul_v3_fl(force[i], factor / 4.0f);
        add_v3_v3(base_force, force[i]);
    }

    /* Apply the common and vertex components to all vertices. */
    for (int i = 0; i < 3; i++) 
    {
        add_v3_v3(force[i], base_force);
        add_v3_v3(data->F[vs[i]], force[i]);
    }
}

__host__ __device__ float SIM_tri_area(Implicit_Data* data, const int v1, const int v2, const int v3)
{
    float nor[3];

    return calc_nor_area_tri(nor, data->X[v1], data->X[v2], data->X[v3]);
}

__host__ __device__ void SIM_mass_spring_force_pressure(Implicit_Data* data,
                                    const int v1, const int v2, const int v3,
                                    const float common_pressure, const float* vertex_pressure, const float weights[3])
{
    float nor[3];
    float force[3];

    /* calculate face normal and area */
    const float area = calc_nor_area_tri(nor, data->X[v1], data->X[v2], data->X[v3]);
    /* The force is calculated and split up evenly for each of the three face verts */
    const float factor = area / 3.0f;
    float base_force = common_pressure * factor;

    /* Compute per-vertex force values from local pressures.
     * From integrating the pressure over the triangle and deriving
     * equivalent vertex forces, it follows that:
     *
     * force[idx] = (sum(pressure) + pressure[idx]) * area / 12
     *
     * Effectively, 1/4 of the pressure acts just on its vertex,
     * while 3/4 is split evenly over all three.
     */
    if (vertex_pressure) 
    {
        copy_v3_fl3(force, vertex_pressure[v1], vertex_pressure[v2], vertex_pressure[v3]);
        mul_v3_fl(force, factor / 4.0f);

        base_force += force[0] + force[1] + force[2];
    }
    else 
    {
        zero_v3(force);
    }

    /* add pressure to each of the face verts */
    madd_v3_v3fl(data->F[v1], nor, (base_force + force[0]) * weights[0]);
    madd_v3_v3fl(data->F[v2], nor, (base_force + force[1]) * weights[1]);
    madd_v3_v3fl(data->F[v3], nor, (base_force + force[2]) * weights[2]);
}

__host__ __device__ static void edge_wind_vertex(const float dir[3],
                             const float length,
                             const float radius,
    const float wind[3],
    float f[3],
    float UNUSED(dfdx[3][3]),
    float UNUSED(dfdv[3][3]))
{
	constexpr float density = 0.01f; /* XXX arbitrary value, corresponds to effect of air density */
	const float windlen = len_v3(wind);

    if (windlen == 0.0f) {
        zero_v3(f);
        return;
    }

    /* angle of wind direction to edge */
	const float cos_alpha = dot_v3v3(wind, dir) / windlen;
	const float sin_alpha = sqrtf(1.0f - cos_alpha * cos_alpha);
	const float cross_section = radius * (static_cast<float>(M_PI) * radius * sin_alpha + length * cos_alpha);

    mul_v3_v3fl(f, wind, density * cross_section);
}

__host__ __device__ void SIM_mass_spring_force_edge_wind(Implicit_Data* data, const int v1, const int v2, const float radius1, const float radius2, const float(*winvec)[3])
{
	float win[3], dir[3];
	float f[3], dfdx[3][3], dfdv[3][3];

	sub_v3_v3v3(dir, data->X[v1], data->X[v2]);
	const float length = normalize_v3(dir);

	world_to_root_v3(data, v1, win, winvec[v1]);
	edge_wind_vertex(dir, length, radius1, win, f, dfdx, dfdv);
	add_v3_v3(data->F[v1], f);

	world_to_root_v3(data, v2, win, winvec[v2]);
	edge_wind_vertex(dir, length, radius2, win, f, dfdx, dfdv);
	add_v3_v3(data->F[v2], f);
}

__global__ void SIM_mass_spring_force_vertex_wind(Implicit_Data* data, const int v, const float (*winvec)[3])
{
	constexpr float density = 0.01f; /* XXX arbitrary value, corresponds to effect of air density */

	float wind[3];
	float f[3];

	world_to_root_v3(data, v, wind, winvec[v]);
	mul_v3_v3fl(f, wind, density);
	add_v3_v3(data->F[v], f);
}

__host__ __device__ void dfdx_spring(float to[3][3], const float dir[3], const float length, const float L,
                                     const float k)
{
	constexpr float I[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
	/* dir is unit length direction, rest is spring's restlength, k is spring constant. */
	// return  ( (I-outerprod(dir, dir))*Min(1.0f, rest/length) - I) * -k;
	outerproduct(to, dir, dir);
	sub_m3_m3m3(to, I, to);

	mul_m3_fl(to, (L / length));
	sub_m3_m3m3(to, to, I);
	mul_m3_fl(to, k);
}

__host__ __device__ void dfdv_damp(float to[3][3], const float dir[3], const float damping)
{
	/* Derivative of force with regards to velocity. */
	outerproduct(to, dir, dir);
	mul_m3_fl(to, -damping);
}


__host__ __device__ float fb(const float length, const float L)
{
	const float x = length / L;
	const float xx = x * x;
	const float xxx = xx * x;
	const float xxxx = xxx * x;
	return (-11.541f * xxxx + 34.193f * xxx - 39.083f * xx + 23.116f * x - 9.713f);
}

__host__ __device__ float fbderiv(const float length, const float L)
{
	const float x = length / L;
	const float xx = x * x;
	const float xxx = xx * x;
	return (-46.164f * xxx + 102.579f * xx - 78.166f * x + 23.116f);
}

__host__ __device__ float fbstar(const float length, const float L, const float kb, const float cb)
{
	const float tempfb_fl = kb * fb(length, L);
	const float fbstar_fl = cb * (length - L);

	if (tempfb_fl < fbstar_fl)
	{
		return fbstar_fl;
	}

	return tempfb_fl;
}

/* Function to calculate bending spring force (taken from Choi & Co). */
__host__ __device__ float fbstar_jacobi(const float length, const float L, const float kb, const float cb)
{
	const float tempfb_fl = kb * fb(length, L);
	const float fbstar_fl = cb * (length - L);

	if (tempfb_fl < fbstar_fl)
	{
		return -cb;
	}

	return -kb * fbderiv(length, L);
}

/* calculate elongation */
__host__ __device__ bool spring_length(Implicit_Data* data,
                                       const int i,
                                       const int j,
                                       float r_extent[3],
                                       float r_dir[3],
                                       float* r_length,
                                       float r_vel[3])
{
	sub_v3_v3v3(r_extent, data->X[j], data->X[i]);
	sub_v3_v3v3(r_vel, data->V[j], data->V[i]);
	*r_length = len_v3(r_extent);

	if (*r_length > ALMOST_ZERO)
	{
		mul_v3_v3fl(r_dir, r_extent, 1.0f / (*r_length));
	}
	else
	{
		zero_v3(r_dir);
	}

	return true;
}

__global__ void g_spring_length(Implicit_Data* data,
                                const int i,
                                const int j,
                                float r_extent[3],
                                float r_dir[3],
                                float* r_length,
                                float r_vel[3])
{
	sub_v3_v3v3(r_extent, data->X[j], data->X[i]);
	sub_v3_v3v3(r_vel, data->V[j], data->V[i]);
	*r_length = len_v3(r_extent);

	if (*r_length > ALMOST_ZERO)
	{
		mul_v3_v3fl(r_dir, r_extent, 1.0f / (*r_length));
	}
	else
	{
		zero_v3(r_dir);
	}
}

__host__ __device__ void apply_spring(Implicit_Data* data, const int i, const int j, const float f[3],
                                      const float dfdx[3][3], const float dfdv[3][3])
{
	const uint block_ij = SIM_mass_spring_add_block(data, i, j);

	add_v3_v3(data->F[i], f);
	sub_v3_v3(data->F[j], f);

	add_m3_m3m3(data->dFdX[i].m, data->dFdX[i].m, dfdx);
	add_m3_m3m3(data->dFdX[j].m, data->dFdX[j].m, dfdx);
	sub_m3_m3m3(data->dFdX[block_ij].m, data->dFdX[block_ij].m, dfdx);

	add_m3_m3m3(data->dFdV[i].m, data->dFdV[i].m, dfdv);
	add_m3_m3m3(data->dFdV[j].m, data->dFdV[j].m, dfdv);
	sub_m3_m3m3(data->dFdV[block_ij].m, data->dFdV[block_ij].m, dfdv);
}

__host__ __device__ bool SIM_mass_spring_force_spring_bending(Implicit_Data* data, const int i, const int j,
                                                              const float restlen, const float kb, const float cb)
{
	/* Смотрите "Стабильная, но отзывчивая ткань" / "Stable but Responsive Cloth" (Choi, Ko, 2005).*/
	float extent[3], length, dir[3], vel[3];

	/* рассчитать относительное удлинение */
	spring_length(data, i, j, extent, dir, &length, vel);

	if (length < restlen)
	{
		float f[3], dfdx[3][3], dfdv[3][3];

		mul_v3_v3fl(f, dir, fbstar(length, restlen, kb, cb));

		outerproduct(dfdx, dir, dir);
		mul_m3_fl(dfdx, fbstar_jacobi(length, restlen, kb, cb));

		/* XXX демпфирование не поддерживается */
		zero_m3(dfdv);

		apply_spring(data, i, j, f, dfdx, dfdv);

		return true;
	}

	return false;
}

__host__ __device__ void poly_avg(const lfVector* data, const int* inds, const int len, float r_avg[3])
{
	const float fact = 1.0f / static_cast<float>(len);

	zero_v3(r_avg);

#ifdef __CUDA_ARCH__
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len)
    {
        madd_v3_v3fl(r_avg, data[inds[i]], fact);
    }
#else
	for (int i = 0; i < len; i++)
	{
		madd_v3_v3fl(r_avg, data[inds[i]], fact);
	}
#endif
}

__host__ __device__ void poly_norm(const lfVector* data, const int i, const int j, const int* inds, const int len,
                                   float r_dir[3])
{
	float mid[3];

	poly_avg(data, inds, len, mid);

	normal_tri_v3(r_dir, data[i], data[j], mid);
}

__host__ __device__ void edge_avg(const lfVector* data, const int i, const int j, float r_avg[3])
{
	r_avg[0] = (data[i][0] + data[j][0]) * 0.5f;
	r_avg[1] = (data[i][1] + data[j][1]) * 0.5f;
	r_avg[2] = (data[i][2] + data[j][2]) * 0.5f;
}

__host__ __device__ void edge_norm(const lfVector* data, const int i, const int j, float r_dir[3])
{
	sub_v3_v3v3(r_dir, data[i], data[j]);
	normalize_v3(r_dir);
}

__host__ __device__ float bend_angle(const float dir_a[3], const float dir_b[3], const float dir_e[3])
{
	float tmp[3];

	const float cos = dot_v3v3(dir_a, dir_b);

	cross_v3_v3v3(tmp, dir_a, dir_b);
	const float sin = dot_v3v3(tmp, dir_e);

	return atan2f(sin, cos);
}

__host__ __device__ void spring_angle(const Implicit_Data* data,
                                      const int i, const int j,
                                      const int* i_a, const int* i_b,
                                      const int len_a, const int len_b,
                                      float r_dir_a[3], float r_dir_b[3],
                                      float* r_angle, float r_vel_a[3],
                                      float r_vel_b[3])
{
	float dir_e[3], vel_e[3];

	poly_norm(data->X, j, i, i_a, len_a, r_dir_a);
	poly_norm(data->X, i, j, i_b, len_b, r_dir_b);

	edge_norm(data->X, i, j, dir_e);

	*r_angle = bend_angle(r_dir_a, r_dir_b, dir_e);

	poly_avg(data->V, i_a, len_a, r_vel_a);
	poly_avg(data->V, i_b, len_b, r_vel_b);

	edge_avg(data->V, i, j, vel_e);

	sub_v3_v3(r_vel_a, vel_e);
	sub_v3_v3(r_vel_b, vel_e);
}

///* Jacobian of a direction vector.
// * Basically the part of the differential orthogonal to the direction,
// * inversely proportional to the length of the edge.
// *
// * dD_ij/dx_i = -dD_ij/dx_j = (D_ij * D_ij^T - I) / len_ij
// */
//__host__ __device__ void spring_grad_dir(Implicit_Data* data, const int i, const int j, float edge[3], float dir[3], float grad_dir[3][3])
//{
//    constexpr float I[3][3] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };
//	sub_v3_v3v3(edge, data->X[j], data->X[i]);
//	const float length = normalize_v3_v3(dir, edge);
//
//    if (length > ALMOST_ZERO) 
//    {
//        outerproduct(grad_dir, dir, dir);
//        sub_m3_m3m3(grad_dir, I, grad_dir);
//        mul_m3_fl(grad_dir, 1.0f / length);
//    }
//    else 
//    {
//        zero_m3(grad_dir);
//    }
//}
//
//__host__ __device__ void spring_hairbend_forces(Implicit_Data* data,
//                            const int i,
//                            const int j,
//                            const int k,
//    const float goal[3],
//                            const float stiffness,
//                            const float damping,
//                            const int q,
//    const float dx[3],
//    const float dv[3],
//    float r_f[3])
//{
//    float edge_ij[3], dir_ij[3];
//    float edge_jk[3], dir_jk[3];
//    float vel_ij[3], vel_jk[3], vel_ortho[3];
//    float f_bend[3], f_damp[3];
//    float fk[3];
//    float dist[3];
//
//    zero_v3(fk);
//
//    sub_v3_v3v3(edge_ij, data->X[j], data->X[i]);
//    if (q == i) {
//        sub_v3_v3(edge_ij, dx);
//    }
//    if (q == j) {
//        add_v3_v3(edge_ij, dx);
//    }
//    normalize_v3_v3(dir_ij, edge_ij);
//
//    sub_v3_v3v3(edge_jk, data->X[k], data->X[j]);
//    if (q == j) {
//        sub_v3_v3(edge_jk, dx);
//    }
//    if (q == k) {
//        add_v3_v3(edge_jk, dx);
//    }
//    normalize_v3_v3(dir_jk, edge_jk);
//
//    sub_v3_v3v3(vel_ij, data->V[j], data->V[i]);
//    if (q == i) {
//        sub_v3_v3(vel_ij, dv);
//    }
//    if (q == j) {
//        add_v3_v3(vel_ij, dv);
//    }
//
//    sub_v3_v3v3(vel_jk, data->V[k], data->V[j]);
//    if (q == j) {
//        sub_v3_v3(vel_jk, dv);
//    }
//    if (q == k) {
//        add_v3_v3(vel_jk, dv);
//    }
//
//    /* bending force */
//    sub_v3_v3v3(dist, goal, edge_jk);
//    mul_v3_v3fl(f_bend, dist, stiffness);
//
//    add_v3_v3(fk, f_bend);
//
//    /* damping force */
//    madd_v3_v3v3fl(vel_ortho, vel_jk, dir_jk, -dot_v3v3(vel_jk, dir_jk));
//    mul_v3_v3fl(f_damp, vel_ortho, damping);
//
//    sub_v3_v3(fk, f_damp);
//
//    copy_v3_v3(r_f, fk);
//}
//
///* Finite Differences method for estimating the jacobian of the force */
//__host__ __device__ void spring_hairbend_estimate_dfdx(Implicit_Data* data,
//                                   const int i,
//                                   const int j,
//                                   const int k,
//    const float goal[3],
//                                   const float stiffness,
//                                   const float damping,
//                                   const int q,
//    float dfdx[3][3])
//{
//	constexpr float delta = 0.00001f; /* TODO: find a good heuristic for this. */
//    float dvec_null[3][3], dvec_pos[3][3], dvec_neg[3][3];
//    float f[3];
//    int a, b;
//
//    zero_m3(dvec_null);
//    unit_m3(dvec_pos);
//    mul_m3_fl(dvec_pos, delta * 0.5f);
//    copy_m3_m3(dvec_neg, dvec_pos);
//    negate_m3(dvec_neg);
//
//    /* XXX TODO: offset targets to account for position dependency. */
//
//    for (a = 0; a < 3; a++) {
//        spring_hairbend_forces(
//            data, i, j, k, goal, stiffness, damping, q, dvec_pos[a], dvec_null[a], f);
//        copy_v3_v3(dfdx[a], f);
//
//        spring_hairbend_forces(
//            data, i, j, k, goal, stiffness, damping, q, dvec_neg[a], dvec_null[a], f);
//        sub_v3_v3(dfdx[a], f);
//
//        for (b = 0; b < 3; b++) {
//            dfdx[a][b] /= delta;
//        }
//    }
//}
//
///* Finite Differences method for estimating the jacobian of the force */
//__host__ __device__ void spring_hairbend_estimate_dfdv(Implicit_Data* data,
//                                   const int i,
//                                   const int j,
//                                   const int k,
//    const float goal[3],
//                                   const float stiffness,
//                                   const float damping,
//                                   const int q,
//    float dfdv[3][3])
//{
//	constexpr float delta = 0.00001f; /* TODO: find a good heuristic for this. */
//    float dvec_null[3][3], dvec_pos[3][3], dvec_neg[3][3];
//    float f[3];
//    int a, b;
//
//    zero_m3(dvec_null);
//    unit_m3(dvec_pos);
//    mul_m3_fl(dvec_pos, delta * 0.5f);
//    copy_m3_m3(dvec_neg, dvec_pos);
//    negate_m3(dvec_neg);
//
//    /* XXX TODO: offset targets to account for position dependency. */
//
//    for (a = 0; a < 3; a++) {
//        spring_hairbend_forces(
//            data, i, j, k, goal, stiffness, damping, q, dvec_null[a], dvec_pos[a], f);
//        copy_v3_v3(dfdv[a], f);
//
//        spring_hairbend_forces(
//            data, i, j, k, goal, stiffness, damping, q, dvec_null[a], dvec_neg[a], f);
//        sub_v3_v3(dfdv[a], f);
//
//        for (b = 0; b < 3; b++) {
//            dfdv[a][b] /= delta;
//        }
//    }
//}

//__host__ __device__ bool SIM_mass_spring_force_spring_bending_hair(Implicit_Data* data,
//                                               const int i,
//                                               const int j,
//                                               const int k,
//    const float target[3],
//                                               const float stiffness,
//                                               const float damping)
//{
//    /* Angular springs roughly based on the bending model proposed by Baraff and Witkin in
//     * "Large Steps in Cloth Simulation". */
//
//    float goal[3];
//    float fj[3], fk[3];
//    float dfj_dxi[3][3], dfj_dxj[3][3], dfk_dxi[3][3], dfk_dxj[3][3], dfk_dxk[3][3];
//    float dfj_dvi[3][3], dfj_dvj[3][3], dfk_dvi[3][3], dfk_dvj[3][3], dfk_dvk[3][3];
//
//    constexpr float vecnull[3] = { 0.0f, 0.0f, 0.0f };
//
//    const int block_ij = SIM_mass_spring_add_block(data, i, j);
//    const int block_jk = SIM_mass_spring_add_block(data, j, k);
//    const int block_ik = SIM_mass_spring_add_block(data, i, k);
//
//    world_to_root_v3(data, j, goal, target);
//
//    spring_hairbend_forces(data, i, j, k, goal, stiffness, damping, k, vecnull, vecnull, fk);
//    negate_v3_v3(fj, fk); /* Counter-force. */
//
//    spring_hairbend_estimate_dfdx(data, i, j, k, goal, stiffness, damping, i, dfk_dxi);
//    spring_hairbend_estimate_dfdx(data, i, j, k, goal, stiffness, damping, j, dfk_dxj);
//    spring_hairbend_estimate_dfdx(data, i, j, k, goal, stiffness, damping, k, dfk_dxk);
//    copy_m3_m3(dfj_dxi, dfk_dxi);
//    negate_m3(dfj_dxi);
//    copy_m3_m3(dfj_dxj, dfk_dxj);
//    negate_m3(dfj_dxj);
//
//    spring_hairbend_estimate_dfdv(data, i, j, k, goal, stiffness, damping, i, dfk_dvi);
//    spring_hairbend_estimate_dfdv(data, i, j, k, goal, stiffness, damping, j, dfk_dvj);
//    spring_hairbend_estimate_dfdv(data, i, j, k, goal, stiffness, damping, k, dfk_dvk);
//    copy_m3_m3(dfj_dvi, dfk_dvi);
//    negate_m3(dfj_dvi);
//    copy_m3_m3(dfj_dvj, dfk_dvj);
//    negate_m3(dfj_dvj);
//
//    /* add forces and jacobians to the solver data */
//
//    add_v3_v3(data->F[j], fj);
//    add_v3_v3(data->F[k], fk);
//
//    add_m3_m3m3(data->dFdX[j].m, data->dFdX[j].m, dfj_dxj);
//    add_m3_m3m3(data->dFdX[k].m, data->dFdX[k].m, dfk_dxk);
//
//    add_m3_m3m3(data->dFdX[block_ij].m, data->dFdX[block_ij].m, dfj_dxi);
//    add_m3_m3m3(data->dFdX[block_jk].m, data->dFdX[block_jk].m, dfk_dxj);
//    add_m3_m3m3(data->dFdX[block_ik].m, data->dFdX[block_ik].m, dfk_dxi);
//
//    add_m3_m3m3(data->dFdV[j].m, data->dFdV[j].m, dfj_dvj);
//    add_m3_m3m3(data->dFdV[k].m, data->dFdV[k].m, dfk_dvk);
//
//    add_m3_m3m3(data->dFdV[block_ij].m, data->dFdV[block_ij].m, dfj_dvi);
//    add_m3_m3m3(data->dFdV[block_jk].m, data->dFdV[block_jk].m, dfk_dvj);
//    add_m3_m3m3(data->dFdV[block_ik].m, data->dFdV[block_ik].m, dfk_dvi);
//    return true;
//}

__host__ __device__ bool SIM_mass_spring_force_spring_goal(Implicit_Data* data,
                                                           const int i,
                                                           const float goal_x[3],
                                                           const float goal_v[3],
                                                           const float stiffness,
                                                           const float damping)
{
	float root_goal_x[3], root_goal_v[3], extent[3], dir[3], vel[3];
	float f[3], dfdx[3][3], dfdv[3][3];

	/* goal is in world space */
	world_to_root_v3(data, i, root_goal_x, goal_x);
	world_to_root_v3(data, i, root_goal_v, goal_v);

	sub_v3_v3v3(extent, root_goal_x, data->X[i]);
	sub_v3_v3v3(vel, root_goal_v, data->V[i]);
	const float length = normalize_v3_v3(dir, extent);

	if (length > ALMOST_ZERO)
	{
		mul_v3_v3fl(f, dir, stiffness * length);

		/* Ascher & Boxman, p.21: Damping only during elongation
		 * something wrong with it. */
		madd_v3_v3fl(f, dir, damping * dot_v3v3(vel, dir));

		dfdx_spring(dfdx, dir, length, 0.0f, stiffness);
		dfdv_damp(dfdv, dir, damping);

		add_v3_v3(data->F[i], f);
		add_m3_m3m3(data->dFdX[i].m, data->dFdX[i].m, dfdx);
		add_m3_m3m3(data->dFdV[i].m, data->dFdV[i].m, dfdv);

		return true;
	}

	return false;
}

#endif /* IMPLICIT_SOLVER_BLENDER */
