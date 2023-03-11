#include "MEM_guardedalloc.cuh"

#include "B_math.h"
#include "utildefines.h"

#include "effect.h"
#include "implicit.h"

float I[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

 int floor_int(float value)
{
  return value > 0.0f ? (int)value : ((int)value) - 1;
}

 float floor_mod(float value)
{
  return value - floorf(value);
}

 int hair_grid_size(const int res[3])
{
  return res[0] * res[1] * res[2];
}

struct HairGridVert {
  int samples;
  float velocity[3];
  float density;

  float velocity_smooth[3];
};

struct HairGrid {
  HairGridVert *verts;
  int res[3];
  float gmin[3], gmax[3];
  float cellsize, inv_cellsize;
};

#define HAIR_GRID_INDEX_AXIS(vec, res, gmin, scale, axis) \
  (min_ii(max_ii((int)((vec[axis] - gmin[axis]) * scale), 0), res[axis] - 2))

 int hair_grid_offset(const float vec[3],
                                const int res[3],
                                const float gmin[3],
                                float scale)
{
  int i, j, k;
  i = HAIR_GRID_INDEX_AXIS(vec, res, gmin, scale, 0);
  j = HAIR_GRID_INDEX_AXIS(vec, res, gmin, scale, 1);
  k = HAIR_GRID_INDEX_AXIS(vec, res, gmin, scale, 2);
  return i + (j + k * res[1]) * res[0];
}

 int hair_grid_interp_weights(
    const int res[3], const float gmin[3], float scale, const float vec[3], float uvw[3])
{
  int i, j, k, offset;

  i = HAIR_GRID_INDEX_AXIS(vec, res, gmin, scale, 0);
  j = HAIR_GRID_INDEX_AXIS(vec, res, gmin, scale, 1);
  k = HAIR_GRID_INDEX_AXIS(vec, res, gmin, scale, 2);
  offset = i + (j + k * res[1]) * res[0];

  uvw[0] = (vec[0] - gmin[0]) * scale - (float)i;
  uvw[1] = (vec[1] - gmin[1]) * scale - (float)j;
  uvw[2] = (vec[2] - gmin[2]) * scale - (float)k;

  return offset;
}

 void hair_grid_interpolate(const HairGridVert *grid,
                                      const int res[3],
                                      const float gmin[3],
                                      float scale,
                                      const float vec[3],
                                      float *density,
                                      float velocity[3],
                                      float vel_smooth[3],
                                      float density_gradient[3],
                                      float velocity_gradient[3][3])
{
  HairGridVert data[8];
  float uvw[3], muvw[3];
  int res2 = res[1] * res[0];
  int offset;

  offset = hair_grid_interp_weights(res, gmin, scale, vec, uvw);
  muvw[0] = 1.0f - uvw[0];
  muvw[1] = 1.0f - uvw[1];
  muvw[2] = 1.0f - uvw[2];

  data[0] = grid[offset];
  data[1] = grid[offset + 1];
  data[2] = grid[offset + res[0]];
  data[3] = grid[offset + res[0] + 1];
  data[4] = grid[offset + res2];
  data[5] = grid[offset + res2 + 1];
  data[6] = grid[offset + res2 + res[0]];
  data[7] = grid[offset + res2 + res[0] + 1];

  if (density) {
    *density = muvw[2] * (muvw[1] * (muvw[0] * data[0].density + uvw[0] * data[1].density) +
                          uvw[1] * (muvw[0] * data[2].density + uvw[0] * data[3].density)) +
               uvw[2] * (muvw[1] * (muvw[0] * data[4].density + uvw[0] * data[5].density) +
                         uvw[1] * (muvw[0] * data[6].density + uvw[0] * data[7].density));
  }

  if (velocity) {
    int k;
    for (k = 0; k < 3; k++) {
      velocity[k] = muvw[2] *
                        (muvw[1] * (muvw[0] * data[0].velocity[k] + uvw[0] * data[1].velocity[k]) +
                         uvw[1] * (muvw[0] * data[2].velocity[k] + uvw[0] * data[3].velocity[k])) +
                    uvw[2] *
                        (muvw[1] * (muvw[0] * data[4].velocity[k] + uvw[0] * data[5].velocity[k]) +
                         uvw[1] * (muvw[0] * data[6].velocity[k] + uvw[0] * data[7].velocity[k]));
    }
  }

  if (vel_smooth) {
    int k;
    for (k = 0; k < 3; k++) {
      vel_smooth[k] = muvw[2] * (muvw[1] * (muvw[0] * data[0].velocity_smooth[k] +
                                            uvw[0] * data[1].velocity_smooth[k]) +
                                 uvw[1] * (muvw[0] * data[2].velocity_smooth[k] +
                                           uvw[0] * data[3].velocity_smooth[k])) +
                      uvw[2] * (muvw[1] * (muvw[0] * data[4].velocity_smooth[k] +
                                           uvw[0] * data[5].velocity_smooth[k]) +
                                uvw[1] * (muvw[0] * data[6].velocity_smooth[k] +
                                          uvw[0] * data[7].velocity_smooth[k]));
    }
  }

  if (density_gradient) {
    density_gradient[0] = muvw[1] * muvw[2] * (data[0].density - data[1].density) +
                          uvw[1] * muvw[2] * (data[2].density - data[3].density) +
                          muvw[1] * uvw[2] * (data[4].density - data[5].density) +
                          uvw[1] * uvw[2] * (data[6].density - data[7].density);

    density_gradient[1] = muvw[2] * muvw[0] * (data[0].density - data[2].density) +
                          uvw[2] * muvw[0] * (data[4].density - data[6].density) +
                          muvw[2] * uvw[0] * (data[1].density - data[3].density) +
                          uvw[2] * uvw[0] * (data[5].density - data[7].density);

    density_gradient[2] = muvw[2] * muvw[0] * (data[0].density - data[4].density) +
                          uvw[2] * muvw[0] * (data[1].density - data[5].density) +
                          muvw[2] * uvw[0] * (data[2].density - data[6].density) +
                          uvw[2] * uvw[0] * (data[3].density - data[7].density);
  }

  if (velocity_gradient) {
    /* XXX TODO: */
    zero_m3(velocity_gradient);
  }
}

void SIM_hair_volume_vertex_grid_forces(HairGrid *grid,
                                        const float x[3],
                                        const float v[3],
                                        float smoothfac,
                                        float pressurefac,
                                        float minpressure,
                                        float f[3],
                                        float dfdx[3][3],
                                        float dfdv[3][3])
{
  float gdensity, gvelocity[3], ggrad[3], gvelgrad[3][3], gradlen;

  hair_grid_interpolate(grid->verts,
                        grid->res,
                        grid->gmin,
                        grid->inv_cellsize,
                        x,
                        &gdensity,
                        gvelocity,
                        nullptr,
                        ggrad,
                        gvelgrad);

  zero_v3(f);
  sub_v3_v3(gvelocity, v);
  mul_v3_v3fl(f, gvelocity, smoothfac);

  gradlen = normalize_v3(ggrad) - minpressure;
  if (gradlen > 0.0f) {
    mul_v3_fl(ggrad, gradlen);
    madd_v3_v3fl(f, ggrad, pressurefac);
  }

  zero_m3(dfdx);

  sub_m3_m3m3(dfdv, gvelgrad, I);
  mul_m3_fl(dfdv, smoothfac);
}

void SIM_hair_volume_grid_interpolate(HairGrid *grid,
                                      const float x[3],
                                      float *density,
                                      float velocity[3],
                                      float velocity_smooth[3],
                                      float density_gradient[3],
                                      float velocity_gradient[3][3])
{
  hair_grid_interpolate(grid->verts,
                        grid->res,
                        grid->gmin,
                        grid->inv_cellsize,
                        x,
                        density,
                        velocity,
                        velocity_smooth,
                        density_gradient,
                        velocity_gradient);
}

void SIM_hair_volume_grid_velocity(
    HairGrid *grid, const float x[3], const float v[3], float fluid_factor, float r_v[3])
{
  float gdensity, gvelocity[3], gvel_smooth[3], ggrad[3], gvelgrad[3][3];
  float v_pic[3], v_flip[3];

  hair_grid_interpolate(grid->verts,
                        grid->res,
                        grid->gmin,
                        grid->inv_cellsize,
                        x,
                        &gdensity,
                        gvelocity,
                        gvel_smooth,
                        ggrad,
                        gvelgrad);

  /* velocity according to PIC method (Particle-in-Cell) */
  copy_v3_v3(v_pic, gvel_smooth);

  /* velocity according to FLIP method (Fluid-Implicit-Particle) */
  sub_v3_v3v3(v_flip, gvel_smooth, gvelocity);
  add_v3_v3(v_flip, v);

  interp_v3_v3v3(r_v, v_pic, v_flip, fluid_factor);
}

void SIM_hair_volume_grid_clear(HairGrid *grid)
{
  const int size = hair_grid_size(grid->res);
  int i;
  for (i = 0; i < size; i++) {
    zero_v3(grid->verts[i].velocity);
    zero_v3(grid->verts[i].velocity_smooth);
    grid->verts[i].density = 0.0f;
    grid->verts[i].samples = 0;
  }
}

 bool hair_grid_point_valid(const float vec[3], const float gmin[3], const float gmax[3])
{
  return !(vec[0] < gmin[0] || vec[1] < gmin[1] || vec[2] < gmin[2] || vec[0] > gmax[0] ||
           vec[1] > gmax[1] || vec[2] > gmax[2]);
}

 float dist_tent_v3f3(const float a[3], float x, float y, float z)
{
  float w = (1.0f - fabsf(a[0] - x)) * (1.0f - fabsf(a[1] - y)) * (1.0f - fabsf(a[2] - z));
  return w;
}

 float weights_sum(const float weights[8])
{
  float totweight = 0.0f;
  int i;
  for (i = 0; i < 8; i++) {
    totweight += weights[i];
  }
  return totweight;
}

/* returns the grid array offset as well to avoid redundant calculation */
 int hair_grid_weights(
    const int res[3], const float gmin[3], float scale, const float vec[3], float weights[8])
{
  int i, j, k, offset;
  float uvw[3];

  i = HAIR_GRID_INDEX_AXIS(vec, res, gmin, scale, 0);
  j = HAIR_GRID_INDEX_AXIS(vec, res, gmin, scale, 1);
  k = HAIR_GRID_INDEX_AXIS(vec, res, gmin, scale, 2);
  offset = i + (j + k * res[1]) * res[0];

  uvw[0] = (vec[0] - gmin[0]) * scale;
  uvw[1] = (vec[1] - gmin[1]) * scale;
  uvw[2] = (vec[2] - gmin[2]) * scale;

  weights[0] = dist_tent_v3f3(uvw, (float)i, (float)j, (float)k);
  weights[1] = dist_tent_v3f3(uvw, (float)(i + 1), (float)j, (float)k);
  weights[2] = dist_tent_v3f3(uvw, (float)i, (float)(j + 1), (float)k);
  weights[3] = dist_tent_v3f3(uvw, (float)(i + 1), (float)(j + 1), (float)k);
  weights[4] = dist_tent_v3f3(uvw, (float)i, (float)j, (float)(k + 1));
  weights[5] = dist_tent_v3f3(uvw, (float)(i + 1), (float)j, (float)(k + 1));
  weights[6] = dist_tent_v3f3(uvw, (float)i, (float)(j + 1), (float)(k + 1));
  weights[7] = dist_tent_v3f3(uvw, (float)(i + 1), (float)(j + 1), (float)(k + 1));

  // BLI_assert(fabsf(weights_sum(weights) - 1.0f) < 0.0001f);

  return offset;
}

 void grid_to_world(HairGrid *grid, float vecw[3], const float vec[3])
{
  copy_v3_v3(vecw, vec);
  mul_v3_fl(vecw, grid->cellsize);
  add_v3_v3(vecw, grid->gmin);
}

void SIM_hair_volume_add_vertex(HairGrid *grid, const float x[3], const float v[3])
{
  const int res[3] = {grid->res[0], grid->res[1], grid->res[2]};
  float weights[8];
  int di, dj, dk;
  int offset;

  if (!hair_grid_point_valid(x, grid->gmin, grid->gmax)) {
    return;
  }

  offset = hair_grid_weights(res, grid->gmin, grid->inv_cellsize, x, weights);

  for (di = 0; di < 2; di++) {
    for (dj = 0; dj < 2; dj++) {
      for (dk = 0; dk < 2; dk++) {
        int voffset = offset + di + (dj + dk * res[1]) * res[0];
        int iw = di + dj * 2 + dk * 4;

        grid->verts[voffset].density += weights[iw];
        madd_v3_v3fl(grid->verts[voffset].velocity, v, weights[iw]);
      }
    }
  }
}

 void hair_volume_eval_grid_vertex_sample(HairGridVert *vert,
                                                    const float loc[3],
                                                    float radius,
                                                    float dist_scale,
                                                    const float x[3],
                                                    const float v[3])
{
  float dist, weight;

  dist = len_v3v3(x, loc);

  weight = (radius - dist) * dist_scale;

  if (weight > 0.0f) {
    madd_v3_v3fl(vert->velocity, v, weight);
    vert->density += weight;
    vert->samples += 1;
  }
}

void SIM_hair_volume_add_segment(HairGrid *grid,
                                 const float UNUSED(x1[3]),
                                 const float UNUSED(v1[3]),
                                 const float x2[3],
                                 const float v2[3],
                                 const float x3[3],
                                 const float v3[3],
                                 const float UNUSED(x4[3]),
                                 const float UNUSED(v4[3]),
                                 const float UNUSED(dir1[3]),
                                 const float UNUSED(dir2[3]),
                                 const float UNUSED(dir3[3]))
{
  /* XXX simplified test implementation using a series of discrete sample along the segment,
   * instead of finding the closest point for all affected grid vertices. */

  const float radius = 1.5f;
  const float dist_scale = grid->inv_cellsize;

  const int res[3] = {grid->res[0], grid->res[1], grid->res[2]};
  const int stride[3] = {1, res[0], res[0] * res[1]};
  const int num_samples = 10;

  int s;

  for (s = 0; s < num_samples; s++) {
    float x[3], v[3];
    int i, j, k;

    float f = (float)s / (float)(num_samples - 1);
    interp_v3_v3v3(x, x2, x3, f);
    interp_v3_v3v3(v, v2, v3, f);

    int imin = max_ii(floor_int(x[0]) - 2, 0);
    int imax = min_ii(floor_int(x[0]) + 2, res[0] - 1);
    int jmin = max_ii(floor_int(x[1]) - 2, 0);
    int jmax = min_ii(floor_int(x[1]) + 2, res[1] - 1);
    int kmin = max_ii(floor_int(x[2]) - 2, 0);
    int kmax = min_ii(floor_int(x[2]) + 2, res[2] - 1);

    for (k = kmin; k <= kmax; k++) {
      for (j = jmin; j <= jmax; j++) {
        for (i = imin; i <= imax; i++) {
          float loc[3] = {(float)i, (float)j, (float)k};
          HairGridVert *vert = grid->verts + i * stride[0] + j * stride[1] + k * stride[2];

          hair_volume_eval_grid_vertex_sample(vert, loc, radius, dist_scale, x, v);
        }
      }
    }
  }
}

void SIM_hair_volume_normalize_vertex_grid(HairGrid *grid)
{
  int i, size = hair_grid_size(grid->res);
  /* divide velocity with density */
  for (i = 0; i < size; i++) {
    float density = grid->verts[i].density;
    if (density > 0.0f) {
      mul_v3_fl(grid->verts[i].velocity, 1.0f / density);
    }
  }
}

/* Cells with density below this are considered empty. */
const float density_threshold = 0.001f;

/* Contribution of target density pressure to the laplacian in the pressure poisson equation.
 * This is based on the model found in
 * "Two-way Coupled SPH and Particle Level Set Fluid Simulation" (Losasso et al., 2008)
 */
 float hair_volume_density_divergence(float density,
                                                float target_density,
                                                float strength)
{
  if (density > density_threshold && density > target_density) {
    return strength * logf(target_density / density);
  }

  return 0.0f;
}

bool SIM_hair_volume_solve_divergence(HairGrid *grid,
                                      float /*dt*/,
                                      float target_density,
                                      float target_strength)
{
  //const float flowfac = grid->cellsize;
  const float inv_flowfac = 1.0f / grid->cellsize;

  // const int num_cells = hair_grid_size(grid->res);
  const int res[3] = {grid->res[0], grid->res[1], grid->res[2]};
  const int resA[3] = {grid->res[0] + 2, grid->res[1] + 2, grid->res[2] + 2};

  const int stride0 = 1;
  const int stride1 = grid->res[0];
  const int stride2 = grid->res[1] * grid->res[0];
  const int strideA0 = 1;
  const int strideA1 = grid->res[0] + 2;
  const int strideA2 = (grid->res[1] + 2) * (grid->res[0] + 2);

  const int num_cells = res[0] * res[1] * res[2];
  //const int num_cellsA = (res[0] + 2) * (res[1] + 2) * (res[2] + 2);

  HairGridVert *vert_start = grid->verts - (stride0 + stride1 + stride2);
  HairGridVert *vert;
  int i, j, k;

#define MARGIN_i0 (i < 1)
#define MARGIN_j0 (j < 1)
#define MARGIN_k0 (k < 1)
#define MARGIN_i1 (i >= resA[0] - 1)
#define MARGIN_j1 (j >= resA[1] - 1)
#define MARGIN_k1 (k >= resA[2] - 1)

#define NEIGHBOR_MARGIN_i0 (i < 2)
#define NEIGHBOR_MARGIN_j0 (j < 2)
#define NEIGHBOR_MARGIN_k0 (k < 2)
#define NEIGHBOR_MARGIN_i1 (i >= resA[0] - 2)
#define NEIGHBOR_MARGIN_j1 (j >= resA[1] - 2)
#define NEIGHBOR_MARGIN_k1 (k >= resA[2] - 2)

  BLI_assert(num_cells >= 1);

  /* Calculate divergence */
  //lVector B(num_cellsA);
  //for (k = 0; k < resA[2]; k++) {
  //  for (j = 0; j < resA[1]; j++) {
  //    for (i = 0; i < resA[0]; i++) {
  //      int u = i * strideA0 + j * strideA1 + k * strideA2;
  //      bool is_margin = MARGIN_i0 || MARGIN_i1 || MARGIN_j0 || MARGIN_j1 || MARGIN_k0 ||
  //                       MARGIN_k1;

  //      if (is_margin) {
  //        //B[u] = 0.0f;
  //        continue;
  //      }

  //      vert = vert_start + i * stride0 + j * stride1 + k * stride2;

  //      const float *v0 = vert->velocity;
  //      float dx = 0.0f, dy = 0.0f, dz = 0.0f;
  //      if (!NEIGHBOR_MARGIN_i0) {
  //        dx += v0[0] - (vert - stride0)->velocity[0];
  //      }
  //      if (!NEIGHBOR_MARGIN_i1) {
  //        dx += (vert + stride0)->velocity[0] - v0[0];
  //      }
  //      if (!NEIGHBOR_MARGIN_j0) {
  //        dy += v0[1] - (vert - stride1)->velocity[1];
  //      }
  //      if (!NEIGHBOR_MARGIN_j1) {
  //        dy += (vert + stride1)->velocity[1] - v0[1];
  //      }
  //      if (!NEIGHBOR_MARGIN_k0) {
  //        dz += v0[2] - (vert - stride2)->velocity[2];
  //      }
  //      if (!NEIGHBOR_MARGIN_k1) {
  //        dz += (vert + stride2)->velocity[2] - v0[2];
  //      }

  //      float divergence = -0.5f * flowfac * (dx + dy + dz);

  //      /* adjustment term for target density */
  //      float target = hair_volume_density_divergence(
  //          vert->density, target_density, target_strength);

  //      /* B vector contains the finite difference approximation of the velocity divergence.
  //       * NOTE: according to the discretized Navier-Stokes equation the rhs vector
  //       * and resulting pressure gradient should be multiplied by the (inverse) density;
  //       * however, this is already included in the weighting of hair velocities on the grid!
  //       */
  //      B[u] = divergence - target;
  //    }
  //  }
  //}

  //lMatrix A(num_cellsA, num_cellsA);

  for (k = 0; k < resA[2]; k++) {
      for (j = 0; j < resA[1]; j++) {
          for (i = 0; i < resA[0]; i++) {
              int u = i * strideA0 + j * strideA1 + k * strideA2;
              bool is_margin = MARGIN_i0 || MARGIN_i1 || MARGIN_j0 || MARGIN_j1 || MARGIN_k0 ||
                  MARGIN_k1;

              vert = vert_start + i * stride0 + j * stride1 + k * stride2;
              if (!is_margin && vert->density > density_threshold) {
                  int neighbors_lo = 0;
                  int neighbors_hi = 0;
                  //int non_solid_neighbors = 0;
                  int neighbor_lo_index[3];
                  int neighbor_hi_index[3];
                  //int n;

                  /* check for upper bounds in advance
                   * to get the correct number of neighbors,
                   * needed for the diagonal element
                   */
                  if (!NEIGHBOR_MARGIN_k0 && (vert - stride2)->density > density_threshold) {
                      neighbor_lo_index[neighbors_lo++] = u - strideA2;
                  }
                  if (!NEIGHBOR_MARGIN_j0 && (vert - stride1)->density > density_threshold) {
                      neighbor_lo_index[neighbors_lo++] = u - strideA1;
                  }
                  if (!NEIGHBOR_MARGIN_i0 && (vert - stride0)->density > density_threshold) {
                      neighbor_lo_index[neighbors_lo++] = u - strideA0;
                  }
                  if (!NEIGHBOR_MARGIN_i1 && (vert + stride0)->density > density_threshold) {
                      neighbor_hi_index[neighbors_hi++] = u + strideA0;
                  }
                  if (!NEIGHBOR_MARGIN_j1 && (vert + stride1)->density > density_threshold) {
                      neighbor_hi_index[neighbors_hi++] = u + strideA1;
                  }
                  if (!NEIGHBOR_MARGIN_k1 && (vert + stride2)->density > density_threshold) {
                      neighbor_hi_index[neighbors_hi++] = u + strideA2;
                  }

                  // int liquid_neighbors = neighbors_lo + neighbors_hi;
                  //non_solid_neighbors = 6;

                  //for (n = 0; n < neighbors_lo; n++) {
                  //  A.insert(neighbor_lo_index[n], u) = -1.0f;
                  //}
                  //A.insert(u, u) = (float)non_solid_neighbors;
                  //for (n = 0; n < neighbors_hi; n++) {
                  //  A.insert(neighbor_hi_index[n], u) = -1.0f;
                  //}
              }
              else { //A.insert(u, u) = 1.0f; }
              }
          }
      }
  }

  //ConjugateGradient cg;
  //cg.setMaxIterations(100);
  //cg.setTolerance(0.01f);

  //cg.compute(A);

  //Eigen::VectorXf p = cg.solve(B);

  //if (cg.info() == Eigen::Success) {
  //  /* Calculate velocity = grad(p) */
  //  for (k = 0; k < resA[2]; k++) {
  //    for (j = 0; j < resA[1]; j++) {
  //      for (i = 0; i < resA[0]; i++) {
  //        int u = i * strideA0 + j * strideA1 + k * strideA2;
  //        bool is_margin = MARGIN_i0 || MARGIN_i1 || MARGIN_j0 || MARGIN_j1 || MARGIN_k0 ||
  //                         MARGIN_k1;
  //        if (is_margin) {
  //          continue;
  //        }

  //        vert = vert_start + i * stride0 + j * stride1 + k * stride2;
  //        if (vert->density > density_threshold) {
  //          float p_left = p[u - strideA0];
  //          float p_right = p[u + strideA0];
  //          float p_down = p[u - strideA1];
  //          float p_up = p[u + strideA1];
  //          float p_bottom = p[u - strideA2];
  //          float p_top = p[u + strideA2];

  //          /* finite difference estimate of pressure gradient */
  //          float dvel[3];
  //          dvel[0] = p_right - p_left;
  //          dvel[1] = p_up - p_down;
  //          dvel[2] = p_top - p_bottom;
  //          mul_v3_fl(dvel, -0.5f * inv_flowfac);

  //          /* pressure gradient describes velocity delta */
  //          add_v3_v3v3(vert->velocity_smooth, vert->velocity, dvel);
  //        }
  //        else {
  //          zero_v3(vert->velocity_smooth);
  //        }
  //      }
  //    }
  //  }
  //  return true;
  //}

  /* Clear result in case of error */
  for (i = 0, vert = grid->verts; i < num_cells; i++, vert++) 
  {
    zero_v3(vert->velocity_smooth);
  }

  return false;
}

HairGrid *SIM_hair_volume_create_vertex_grid(float cellsize,
                                             const float gmin[3],
                                             const float gmax[3])
{
  float extent[3], gmin_margin[3], gmax_margin[3], scale;
  int resmin[3], resmax[3], res[3], i, size;
  HairGrid *grid;

  /* sanity check */
  if (cellsize <= 0.0f) 
  {
    cellsize = 1.0f;
  }
  scale = 1.0f / cellsize;

  sub_v3_v3v3(extent, gmax, gmin);
  for (i = 0; i < 3; ++i) 
  {
    resmin[i] = floor_int(gmin[i] * scale);
    resmax[i] = floor_int(gmax[i] * scale) + 1;

    /* add margin of 1 cell */
    resmin[i] -= 1;
    resmax[i] += 1;

    res[i] = resmax[i] - resmin[i] + 1;
    /* sanity check: avoid null-sized grid */
    if (res[i] < 4) {
      res[i] = 4;
      resmax[i] = resmin[i] + 4;
    }
    /* sanity check: avoid too large grid size */
    if (res[i] > MAX_HAIR_GRID_RES) {
      res[i] = MAX_HAIR_GRID_RES;
      resmax[i] = resmin[i] + MAX_HAIR_GRID_RES;
    }

    gmin_margin[i] = (float)resmin[i] * cellsize;
    gmax_margin[i] = (float)resmax[i] * cellsize;
  }
  size = hair_grid_size(res);

  grid = MEM_cnew<HairGrid>("hair grid");
  grid->res[0] = res[0];
  grid->res[1] = res[1];
  grid->res[2] = res[2];
  copy_v3_v3(grid->gmin, gmin_margin);
  copy_v3_v3(grid->gmax, gmax_margin);
  grid->cellsize = cellsize;
  grid->inv_cellsize = scale;
  grid->verts = (HairGridVert *)MEM_callocN(sizeof(HairGridVert) * size, "hair voxel data");

  return grid;
}

void SIM_hair_volume_free_vertex_grid(HairGrid *grid)
{
  if (grid) {
    if (grid->verts) {
      MEM_freeN(grid->verts);
    }
    MEM_freeN(grid);
  }
}

void SIM_hair_volume_grid_geometry(
    HairGrid *grid, float *cellsize, int res[3], float gmin[3], float gmax[3])
{
  if (cellsize) {
    *cellsize = grid->cellsize;
  }
  if (res) {
    copy_v3_v3_int(res, grid->res);
  }
  if (gmin) {
    copy_v3_v3(gmin, grid->gmin);
  }
  if (gmax) {
    copy_v3_v3(gmax, grid->gmax);
  }
}
