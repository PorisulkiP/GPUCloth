#include "MEM_guardedalloc.cuh"

#include "cloth_types.cuh"
#include "mesh_types.h"
#include "meshdata_types.cuh"
#include "object_types.cuh"

#include "edgehash.h"
#include "linklist.cuh"
#include "B_math.h"
#include "rand.h"
#include "utildefines.h"
#include "mesh_runtime.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_query.h"

#include "bvhutils.h"
#include "cloth.h"
#include "effect.h"
#include "pointcache.h"
#include "task.hh"
#include "modifier.h"
#include "memarena.h"
#include "customdata.h"

#include "SIM_mass_spring.cuh"
#include "BKE_lib_id.h"

/** Compared against total loops. */
#define MESH_FACE_TESSELLATE_THREADED_LIMIT 4096

#define USE_CLIP_SWEEP

typedef struct BendSpringRef {
  int index;
  int polys;
  ClothSpring *spring;
} BendSpringRef;

/******************************************************************************
 *
 * External interface called by modifier.c clothModifier functions.
 *
 ******************************************************************************/

static BVHTree *bvhtree_build_from_cloth(ClothModifierData *clmd, float epsilon)
{
  if (!clmd) {
	return NULL;
  }

  Cloth *cloth = clmd->clothObject;

  if (!cloth) {
	return NULL;
  }

  ClothVertex *verts = cloth->verts;
  const MVertTri *vt = cloth->tri;

  /* in the moment, return zero if no faces there */
  if (!cloth->primitive_num) {
	return NULL;
  }

  /* create quadtree with k=26 */
  BVHTree *bvhtree = BLI_bvhtree_new(cloth->primitive_num, epsilon, 4, 26);

  /* fill tree */
  if (clmd->hairdata == NULL) {
	for (int i = 0; i < cloth->primitive_num; i++, vt++) {
	  float co[3][3];

	  copy_v3_v3(co[0], verts[vt->tri[0]].xold);
	  copy_v3_v3(co[1], verts[vt->tri[1]].xold);
	  copy_v3_v3(co[2], verts[vt->tri[2]].xold);

	  BLI_bvhtree_insert(bvhtree, i, co[0], 3);
	}
  }
  else {
	MEdge *edges = cloth->edges;

	for (int i = 0; i < cloth->primitive_num; i++) {
	  float co[2][3];

	  copy_v3_v3(co[0], verts[edges[i].v1].xold);
	  copy_v3_v3(co[1], verts[edges[i].v2].xold);

	  BLI_bvhtree_insert(bvhtree, i, co[0], 2);
	}
  }

  /* balance tree */
  BLI_bvhtree_balance(bvhtree);

  return bvhtree;
}

void bvhtree_update_from_cloth(ClothModifierData *clmd, bool moving, bool self)
{
  uint i = 0;
  Cloth *cloth = clmd->clothObject;
  BVHTree *bvhtree;
  ClothVertex *verts = cloth->verts;
  const MVertTri *vt;

  BLI_assert(!(clmd->hairdata != NULL && self));

  if (self) {
	bvhtree = cloth->bvhselftree;
  }
  else {
	bvhtree = cloth->bvhtree;
  }

  if (!bvhtree) {
	return;
  }

  vt = cloth->tri;

  /* update vertex position in bvh tree */
  if (clmd->hairdata == NULL) {
	if (verts && vt) {
	  for (i = 0; i < cloth->primitive_num; i++, vt++) {
		float co[3][3], co_moving[3][3];
		bool ret;

		/* copy new locations into array */
		if (moving) {
		  copy_v3_v3(co[0], verts[vt->tri[0]].txold);
		  copy_v3_v3(co[1], verts[vt->tri[1]].txold);
		  copy_v3_v3(co[2], verts[vt->tri[2]].txold);

		  /* update moving positions */
		  copy_v3_v3(co_moving[0], verts[vt->tri[0]].tx);
		  copy_v3_v3(co_moving[1], verts[vt->tri[1]].tx);
		  copy_v3_v3(co_moving[2], verts[vt->tri[2]].tx);

		  ret = BLI_bvhtree_update_node(bvhtree, i, co[0], co_moving[0], 3);
		}
		else {
		  copy_v3_v3(co[0], verts[vt->tri[0]].tx);
		  copy_v3_v3(co[1], verts[vt->tri[1]].tx);
		  copy_v3_v3(co[2], verts[vt->tri[2]].tx);

		  ret = BLI_bvhtree_update_node(bvhtree, i, co[0], NULL, 3);
		}

		/* check if tree is already full */
		if (ret == false) {
		  break;
		}
	  }

	  BLI_bvhtree_update_tree(bvhtree);
	}
  }
  else {
	if (verts) {
	  MEdge *edges = cloth->edges;

	  for (i = 0; i < cloth->primitive_num; i++) {
		float co[2][3];

		copy_v3_v3(co[0], verts[edges[i].v1].tx);
		copy_v3_v3(co[1], verts[edges[i].v2].tx);

		if (!BLI_bvhtree_update_node(bvhtree, i, co[0], NULL, 2)) {
		  break;
		}
	  }

	  BLI_bvhtree_update_tree(bvhtree);
	}
  }
}

// Симуляция одного кадра
int do_step_cloth(Depsgraph *depsgraph, Object *ob, ClothModifierData *clmd, Mesh *result, int framenr)
{
  Cloth *cloth = clmd->clothObject;
  ClothVertex* verts = cloth->verts;
  MVert *mvert = result->mvert;

  /* Принудительно верните все закрепленные вершины в их ограниченное местоположение. */
  for (uint i = 0; i < clmd->clothObject->mvert_num; ++i, ++verts)
  {
	// save the previous position.
	// сохранение предыдущую позицию.
	copy_v3_v3(verts->xold, verts->xconst);
	copy_v3_v3(verts->txold, verts->x);

	// Get the current position.
	// Получение текущей позиции.
	copy_v3_v3(verts->xconst, mvert[i].co);
	mul_m4_v3(ob->obmat, verts->xconst);
  }
  // Получение эффектов, которые могут влиять на ткань
  // пока nullptr
  ListBase* effectors = effectors_create(depsgraph, ob, NULL, clmd->sim_parms->effector_weights, false);

  if (clmd->sim_parms->flags & CLOTH_SIMSETTINGS_FLAG_DYNAMIC_BASEMESH) 
  {
	// Копирование данных вершин из ClothModifierData* clmd в Mesh *result
	cloth_update_verts(ob, clmd, result);
  }

  if ((clmd->sim_parms->flags & CLOTH_SIMSETTINGS_FLAG_DYNAMIC_BASEMESH) ||
	  (clmd->sim_parms->vgroup_shrink > 0) || (clmd->sim_parms->shrink_min != 0.0f)) 
  {
	cloth_update_spring_lengths(clmd, result);
  }

  cloth_update_springs(clmd);

  // Засечение времени симуляции
  // TIMEIT_START(cloth_step)

  /* call the solver. */
  if (SIM_cloth_solve(depsgraph, ob, framenr, clmd, effectors))
  {
	  return 1;
  }
  else
  {
	  return 0;
  }

  // TIMEIT_END(cloth_step)
  // printf ( "%f\n", ( float ) tval() );
}

/************************************************
 * clothModifier_do - main simulation function
 ************************************************/
Cloth* clothModifier_do(ClothModifierData *clmd, Depsgraph *depsgraph, Object *ob, Mesh *mesh)
{
	PointCache *cache;
	PTCacheID pid;
	float timescale = 1;
	int framenr = 0, startframe = 0, endframe = 0, cache_result = 0;

	framenr = depsgraph->ctime;
	cache = clmd->point_cache;
	// Если финальный кадр больше, чем конечный в сцене, то симуляция идёт до конца сцены
	if (framenr > endframe)  framenr = endframe;

	BKE_ptcache_id_from_cloth(&pid, ob, clmd);
	BKE_ptcache_id_time(&pid, depsgraph->scene, framenr, &startframe, &endframe, &timescale);
	clmd->sim_parms->timescale = timescale * clmd->sim_parms->time_scale;

	// Если в интерфейсе нажато "сбросить" или кол-во точек в объекте симуляции и ткани не равны, то всё очищается
	if (clmd->sim_parms->reset || (clmd->clothObject && mesh->totvert != clmd->clothObject->mvert_num)) 
	{
		clmd->sim_parms->reset = 0;
		cache->flag |= PTCACHE_OUTDATED;
		BKE_ptcache_id_reset(depsgraph->scene, &pid, PTCACHE_RESET_OUTDATED);
		BKE_ptcache_validate(cache, 0);
		cache->last_exact = 0;
		cache->flag &= ~PTCACHE_REDO_NEEDED;
	}

	// Симуляция происходит только в течение определенного периода
	if (framenr < startframe) 
	{
		BKE_ptcache_invalidate(cache);
		return nullptr;
	}

	// Если кадр начала и текущй равны
	if (framenr == startframe) 
	{
		BKE_ptcache_id_reset(depsgraph->scene, &pid, PTCACHE_RESET_OUTDATED);
		BKE_ptcache_validate(cache, framenr);
		cache->flag &= ~PTCACHE_REDO_NEEDED;
		clmd->clothObject->last_frame = framenr;
		return nullptr;
	}

  // try to read from cache
  // Попытка прочитать из кеша, пока не будет этой функции
	bool can_simulate = false;// (framenr == clmd->clothObject->last_frame + 1) && !(cache->flag & PTCACHE_BAKED);

  //cache_result = BKE_ptcache_read(&pid, (float)framenr + depsgraph->scene->r.subframe, can_simulate);

  //if (cache_result == PTCACHE_READ_EXACT || cache_result == PTCACHE_READ_INTERPOLATED ||
  //    (!can_simulate && cache_result == PTCACHE_READ_OLD)) {
  //  SIM_cloth_solver_set_positions(clmd);
  //  BKE_ptcache_validate(cache, framenr);

  //  if (cache_result == PTCACHE_READ_INTERPOLATED && cache->flag & PTCACHE_REDO_NEEDED) 
  //  {
  //    BKE_ptcache_write(&pid, framenr);
  //  }

  //  clmd->clothObject->last_frame = framenr;

  //  return;
  //}
  //if (cache_result == PTCACHE_READ_OLD) {
  //  SIM_cloth_solver_set_positions(clmd);
  //}
  //else if (
  //    /* 2.4x disabled lib, but this can be used in some cases, testing further - campbell */
  //    /*ob->id.lib ||*/ (cache->flag & PTCACHE_BAKED)) {
  //  /* if baked and nothing in cache, do nothing */
  //  BKE_ptcache_invalidate(cache);
  //  return;
  //}

  /* if on second frame, write cache for first frame */
  //if (cache->simframe == startframe && (cache->flag & PTCACHE_OUTDATED || cache->last_exact == 0)) 
  //{
  //  BKE_ptcache_write(&pid, startframe);
  //}

	clmd->sim_parms->timescale = 1;//*= framenr - cache->simframe;

	//BKE_ptcache_validate(cache, framenr);

	/* do simulation */
	if (do_step_cloth(depsgraph, ob, clmd, mesh, framenr)) 
	{
		// Если симуляция прошла успешно, то проверяем кеш
		//BKE_ptcache_write(&pid, framenr);
	}
	else {
		// В противном случае всё обнуляем
		BKE_ptcache_invalidate(cache);
	}
	clmd->clothObject->last_frame = framenr;
	return clmd->clothObject;
}

/* frees all */
void cloth_free_modifier(ClothModifierData *clmd)
{
  Cloth *cloth = NULL;

  if (!clmd) { return; }

  cloth = clmd->clothObject;

  if (cloth) {
	SIM_cloth_solver_free(clmd);

	/* Free the verts. */
	if (cloth->verts != NULL) {
	  MEM_freeN(cloth->verts);
	}

	cloth->verts = NULL;
	cloth->mvert_num = 0;

	/* Free the springs. */
	if (cloth->springs != NULL) {
	  LinkNode *search = cloth->springs;
	  while (search) {
		ClothSpring *spring = (ClothSpring*)search->link;

		MEM_SAFE_FREE(spring->pa);
		MEM_SAFE_FREE(spring->pb);

		MEM_freeN(spring);
		search = search->next;
	  }
	  BLI_linklist_free(cloth->springs, NULL);

	  cloth->springs = NULL;
	}

	cloth->springs = NULL;
	cloth->numsprings = 0;

	/* free BVH collision tree */
	if (cloth->bvhtree) {
	  BLI_bvhtree_free(cloth->bvhtree);
	}

	if (cloth->bvhselftree) {
	  BLI_bvhtree_free(cloth->bvhselftree);
	}

	/* we save our faces for collision objects */
	if (cloth->tri) {
	  MEM_freeN(cloth->tri);
	}

	if (cloth->edgeset) {
	  BLI_edgeset_free(cloth->edgeset);
	}

	if (cloth->sew_edge_graph) {
	  BLI_edgeset_free(cloth->sew_edge_graph);
	  cloth->sew_edge_graph = NULL;
	}
	MEM_freeN(cloth);
	clmd->clothObject = NULL;
  }
}

void cloth_free_modifier_extern(ClothModifierData* clmd)
{
	Cloth* cloth = NULL;
	if (!clmd) { return; }

	cloth = clmd->clothObject;

	if (cloth) {
		SIM_cloth_solver_free(clmd);

		/* Free the verts. */
		MEM_SAFE_FREE(cloth->verts);
		cloth->mvert_num = 0;

		/* Free the springs. */
		if (cloth->springs != NULL) {
			LinkNode* search = cloth->springs;
			while (search) {
				ClothSpring* spring = (ClothSpring*)search->link;

				MEM_SAFE_FREE(spring->pa);
				MEM_SAFE_FREE(spring->pb);

				MEM_freeN(spring);
				search = search->next;
			}
			BLI_linklist_free(cloth->springs, NULL);

			cloth->springs = NULL;
		}

		cloth->springs = NULL;
		cloth->numsprings = 0;

		/* free BVH collision tree */
		if (cloth->bvhtree) {
			BLI_bvhtree_free(cloth->bvhtree);
		}

		if (cloth->bvhselftree) {
			BLI_bvhtree_free(cloth->bvhselftree);
		}

		/* we save our faces for collision objects */
		if (cloth->tri) {
			MEM_freeN(cloth->tri);
		}

		if (cloth->edgeset) {
			BLI_edgeset_free(cloth->edgeset);
		}

		if (cloth->sew_edge_graph) {
			BLI_edgeset_free(cloth->sew_edge_graph);
			cloth->sew_edge_graph = NULL;
		}
		MEM_freeN(cloth);
		clmd->clothObject = NULL;
	}
}

int cloth_uses_vgroup(ClothModifierData *clmd)
{
  return (((clmd->coll_parms->flags & CLOTH_COLLSETTINGS_FLAG_SELF) &&
		   (clmd->coll_parms->vgroup_selfcol > 0)) ||
		  ((clmd->coll_parms->flags & CLOTH_COLLSETTINGS_FLAG_ENABLED) &&
		   (clmd->coll_parms->vgroup_objcol > 0)) ||
		  (clmd->sim_parms->vgroup_pressure > 0) || (clmd->sim_parms->vgroup_struct > 0) ||
		  (clmd->sim_parms->vgroup_bend > 0) || (clmd->sim_parms->vgroup_shrink > 0) ||
		  (clmd->sim_parms->vgroup_intern > 0) || (clmd->sim_parms->vgroup_mass > 0));
}

static float cloth_shrink_factor(ClothModifierData *clmd, ClothVertex *verts, int i1, int i2)
{
  /* Linear interpolation between min and max shrink factor based on weight. */
  float base = 1.0f - clmd->sim_parms->shrink_min;
  float shrink_factor_delta = clmd->sim_parms->shrink_min - clmd->sim_parms->shrink_max;

  float k1 = base + shrink_factor_delta * verts[i1].shrink_factor;
  float k2 = base + shrink_factor_delta * verts[i2].shrink_factor;

  /* Use geometrical mean to average two factors since it behaves better
   * for diagonals when a rectangle transforms into a trapezoid. */
  return sqrtf(k1 * k2);
}
/* -------------------------------------------------------------------- */
/** \name Spring Network Building Implementation
 * \{ */

 void spring_verts_ordered_set(ClothSpring *spring, int v0, int v1)
{
  if (v0 < v1) 
  {
	spring->ij = v0;
	spring->kl = v1;
  }
  else 
  {
	spring->ij = v1;
	spring->kl = v0;
  }
}

static void cloth_free_edgelist(LinkNodePair *edgelist, uint mvert_num)
{
  if (edgelist) 
  {
	for (uint i = 0; i < mvert_num; i++) 
	{
	  BLI_linklist_free(edgelist[i].list, NULL);
	}

	MEM_freeN(edgelist);
  }
}

static void cloth_free_errorsprings(Cloth *cloth,
									LinkNodePair *edgelist,
									BendSpringRef *spring_ref)
{
  if (cloth->springs != NULL) {
	LinkNode *search = cloth->springs;
	while (search) {
	  ClothSpring *spring = (ClothSpring*)search->link;

	  MEM_SAFE_FREE(spring->pa);
	  MEM_SAFE_FREE(spring->pb);

	  MEM_freeN(spring);
	  search = search->next;
	}
	BLI_linklist_free(cloth->springs, NULL);

	cloth->springs = NULL;
  }

  cloth_free_edgelist(edgelist, cloth->mvert_num);

  MEM_SAFE_FREE(spring_ref);

  if (cloth->edgeset) {
	BLI_edgeset_free(cloth->edgeset);
	cloth->edgeset = NULL;
  }
}

 void cloth_bend_poly_dir(
	ClothVertex *verts, int i, int j, const int *inds, int len, float r_dir[3])
{
  float cent[3] = {0};
  float fact = 1.0f / len;

  for (int x = 0; x < len; x++) {
	madd_v3_v3fl(cent, verts[inds[x]].xrest, fact);
  }

  normal_tri_v3(r_dir, verts[i].xrest, verts[j].xrest, cent);
}

static float cloth_spring_angle(
	ClothVertex *verts, int i, int j, int *i_a, int *i_b, int len_a, int len_b)
{
  float dir_a[3], dir_b[3];
  float tmp[3], vec_e[3];
  float sin, cos;

  /* Poly vectors. */
  cloth_bend_poly_dir(verts, j, i, i_a, len_a, dir_a);
  cloth_bend_poly_dir(verts, i, j, i_b, len_b, dir_b);

  /* Edge vector. */
  sub_v3_v3v3(vec_e, verts[i].xrest, verts[j].xrest);
  normalize_v3(vec_e);

  /* Compute angle. */
  cos = dot_v3v3(dir_a, dir_b);

  cross_v3_v3v3(tmp, dir_a, dir_b);
  sin = dot_v3v3(tmp, vec_e);

  return atan2f(sin, cos);
}

static void cloth_hair_update_bending_targets(ClothModifierData *clmd)
{
  Cloth *cloth = clmd->clothObject;
  LinkNode *search = NULL;
  float hair_frame[3][3], dir_old[3], dir_new[3];
  int prev_mn; /* to find hair chains */

  if (!clmd->hairdata) {
	return;
  }

  /* XXX Note: we need to propagate frames from the root up,
   * but structural hair springs are stored in reverse order.
   * The bending springs however are then inserted in the same
   * order as vertices again ...
   * This messy situation can be resolved when solver data is
   * generated directly from a dedicated hair system.
   */

  prev_mn = -1;
  for (search = cloth->springs; search; search = search->next) {
	ClothSpring *spring = (ClothSpring*)search->link;
	ClothHairData *hair_ij, *hair_kl;
	bool is_root = spring->kl != prev_mn;

	if (spring->type != CLOTH_SPRING_TYPE_BENDING_HAIR) {
	  continue;
	}

	hair_ij = &clmd->hairdata[spring->ij];
	hair_kl = &clmd->hairdata[spring->kl];
	if (is_root) {
	  /* initial hair frame from root orientation */
	  copy_m3_m3(hair_frame, hair_ij->rot);
	  /* surface normal is the initial direction,
	   * parallel transport then keeps it aligned to the hair direction
	   */
	  copy_v3_v3(dir_new, hair_frame[2]);
	}

	copy_v3_v3(dir_old, dir_new);
	sub_v3_v3v3(dir_new, cloth->verts[spring->mn].x, cloth->verts[spring->kl].x);
	normalize_v3(dir_new);

	/* get local targets for kl/mn vertices by putting rest targets into the current frame,
	 * then multiply with the rest length to get the actual goals
	 */

	mul_v3_m3v3(spring->target, hair_frame, hair_kl->rest_target);
	mul_v3_fl(spring->target, spring->restlen);

	/* move frame to next hair segment */
	cloth_parallel_transport_hair_frame(hair_frame, dir_old, dir_new);

	prev_mn = spring->mn;
  }
}

static void cloth_hair_update_bending_rest_targets(ClothModifierData *clmd)
{
  Cloth *cloth = clmd->clothObject;
  LinkNode *search = NULL;
  float hair_frame[3][3], dir_old[3], dir_new[3];
  int prev_mn; /* to find hair roots */

  if (!clmd->hairdata) {
	return;
  }

  /* XXX Note: we need to propagate frames from the root up,
   * but structural hair springs are stored in reverse order.
   * The bending springs however are then inserted in the same
   * order as vertices again ...
   * This messy situation can be resolved when solver data is
   * generated directly from a dedicated hair system.
   */

  prev_mn = -1;
  for (search = cloth->springs; search; search = search->next) {
	ClothSpring *spring = (ClothSpring*)search->link;
	ClothHairData *hair_ij, *hair_kl;
	bool is_root = spring->kl != prev_mn;

	if (spring->type != CLOTH_SPRING_TYPE_BENDING_HAIR) {
	  continue;
	}

	hair_ij = &clmd->hairdata[spring->ij];
	hair_kl = &clmd->hairdata[spring->kl];
	if (is_root) {
	  /* initial hair frame from root orientation */
	  copy_m3_m3(hair_frame, hair_ij->rot);
	  /* surface normal is the initial direction,
	   * parallel transport then keeps it aligned to the hair direction
	   */
	  copy_v3_v3(dir_new, hair_frame[2]);
	}

	copy_v3_v3(dir_old, dir_new);
	sub_v3_v3v3(dir_new, cloth->verts[spring->mn].xrest, cloth->verts[spring->kl].xrest);
	normalize_v3(dir_new);

	/* dir expressed in the hair frame defines the rest target direction */
	copy_v3_v3(hair_kl->rest_target, dir_new);
	mul_transposed_m3_v3(hair_frame, hair_kl->rest_target);

	/* move frame to next hair segment */
	cloth_parallel_transport_hair_frame(hair_frame, dir_old, dir_new);

	prev_mn = spring->mn;
  }
}

/* update stiffness if vertex group values are changing from frame to frame */
static void cloth_update_springs(ClothModifierData *clmd)
{
  Cloth *cloth = clmd->clothObject;
  LinkNode *search = NULL;

  search = cloth->springs;
  while (search) {
	ClothSpring *spring = (ClothSpring*)search->link;

	spring->lin_stiffness = 0.0f;

	if (clmd->sim_parms->bending_model == CLOTH_BENDING_ANGULAR) {
	  if (spring->type & CLOTH_SPRING_TYPE_BENDING) {
		spring->ang_stiffness = (cloth->verts[spring->kl].bend_stiff +
								 cloth->verts[spring->ij].bend_stiff) /
								2.0f;
	  }
	}

	if (spring->type & CLOTH_SPRING_TYPE_STRUCTURAL) {
	  spring->lin_stiffness = (cloth->verts[spring->kl].struct_stiff +
							   cloth->verts[spring->ij].struct_stiff) /
							  2.0f;
	}
	else if (spring->type & CLOTH_SPRING_TYPE_SHEAR) {
	  spring->lin_stiffness = (cloth->verts[spring->kl].shear_stiff +
							   cloth->verts[spring->ij].shear_stiff) /
							  2.0f;
	}
	else if (spring->type == CLOTH_SPRING_TYPE_BENDING) {
	  spring->lin_stiffness = (cloth->verts[spring->kl].bend_stiff +
							   cloth->verts[spring->ij].bend_stiff) /
							  2.0f;
	}
	else if (spring->type & CLOTH_SPRING_TYPE_INTERNAL) {
	  spring->lin_stiffness = (cloth->verts[spring->kl].internal_stiff +
							   cloth->verts[spring->ij].internal_stiff) /
							  2.0f;
	}
	else if (spring->type == CLOTH_SPRING_TYPE_BENDING_HAIR) {
	  ClothVertex *v1 = &cloth->verts[spring->ij];
	  ClothVertex *v2 = &cloth->verts[spring->kl];
	  if (clmd->hairdata) {
		/* copy extra hair data to generic cloth vertices */
		v1->bend_stiff = clmd->hairdata[spring->ij].bending_stiffness;
		v2->bend_stiff = clmd->hairdata[spring->kl].bending_stiffness;
	  }
	  spring->lin_stiffness = (v1->bend_stiff + v2->bend_stiff) / 2.0f;
	}
	else if (spring->type == CLOTH_SPRING_TYPE_GOAL) {
	  /* Warning: Appending NEW goal springs does not work
	   * because implicit solver would need reset! */

	  /* Activate / Deactivate existing springs */
	  if ((!(cloth->verts[spring->ij].flags & CLOTH_VERT_FLAG_PINNED)) &&
		  (cloth->verts[spring->ij].goal > ALMOST_ZERO)) {
		spring->flags &= ~CLOTH_SPRING_FLAG_DEACTIVATE;
	  }
	  else {
		spring->flags |= CLOTH_SPRING_FLAG_DEACTIVATE;
	  }
	}

	search = search->next;
  }

  cloth_hair_update_bending_targets(clmd);
}

// Update rest verts, for dynamically deformable cloth
// Обновите остальные вершины для динамически деформируемой ткани
static void cloth_update_verts(Object *ob, ClothModifierData *clmd, Mesh *mesh)
{
  MVert *mvert = mesh->mvert;
  ClothVertex *verts = clmd->clothObject->verts;

  // vertex count is already ensured to match
  // количество вершин уже обеспечено для соответствия
  for (uint i = 0; i < mesh->totvert; ++i, ++verts)
  {
	copy_v3_v3(verts->xrest, mvert[i].co);
	mul_m4_v3(ob->obmat, verts->xrest);
  }
}

/* Write rest vert locations to a copy of the mesh. */
static Mesh *cloth_make_rest_mesh(ClothModifierData *clmd, Mesh *mesh)
{
  Mesh* new_mesh = NULL;// BKE_mesh_copy_for_eval(mesh, false);
  ClothVertex *verts = clmd->clothObject->verts;
  MVert *mvert = new_mesh->mvert;

  /* vertex count is already ensured to match */
  for (unsigned i = 0; i < mesh->totvert; i++, verts++) {
	copy_v3_v3(mvert[i].co, verts->xrest);
  }

  return new_mesh;
}

/* Update spring rest length, for dynamically deformable cloth */
static void cloth_update_spring_lengths(ClothModifierData *clmd, Mesh *mesh)
{
  Cloth *cloth = clmd->clothObject;
  LinkNode *search = cloth->springs;
  uint struct_springs = 0;
  uint i = 0;
  uint mvert_num = (uint)mesh->totvert;
  float shrink_factor;

  clmd->sim_parms->avg_spring_len = 0.0f;

  for (i = 0; i < mvert_num; i++) {
	cloth->verts[i].avg_spring_len = 0.0f;
  }

  while (search) {
	ClothSpring *spring = (ClothSpring*)search->link;

	if (spring->type != CLOTH_SPRING_TYPE_SEWING) {
	  if (spring->type & (CLOTH_SPRING_TYPE_STRUCTURAL | CLOTH_SPRING_TYPE_SHEAR |
						  CLOTH_SPRING_TYPE_BENDING | CLOTH_SPRING_TYPE_INTERNAL)) {
		shrink_factor = cloth_shrink_factor(clmd, cloth->verts, spring->ij, spring->kl);
	  }
	  else {
		shrink_factor = 1.0f;
	  }

	  spring->restlen = len_v3v3(cloth->verts[spring->kl].xrest, cloth->verts[spring->ij].xrest) *
						shrink_factor;

	  if (spring->type & CLOTH_SPRING_TYPE_BENDING) {
		spring->restang = cloth_spring_angle(
			cloth->verts, spring->ij, spring->kl, spring->pa, spring->pb, spring->la, spring->lb);
	  }
	}

	if (spring->type & CLOTH_SPRING_TYPE_STRUCTURAL) {
	  clmd->sim_parms->avg_spring_len += spring->restlen;
	  cloth->verts[spring->ij].avg_spring_len += spring->restlen;
	  cloth->verts[spring->kl].avg_spring_len += spring->restlen;
	  struct_springs++;
	}

	search = search->next;
  }

  if (struct_springs > 0) {
	clmd->sim_parms->avg_spring_len /= struct_springs;
  }

  for (i = 0; i < mvert_num; i++) {
	if (cloth->verts[i].spring_count > 0) {
	  cloth->verts[i].avg_spring_len = cloth->verts[i].avg_spring_len * 0.49f /
									   ((float)cloth->verts[i].spring_count);
	}
  }
}

 void cross_identity_v3(float r[3][3], const float v[3])
{
  zero_m3(r);
  r[0][1] = v[2];
  r[0][2] = -v[1];
  r[1][0] = -v[2];
  r[1][2] = v[0];
  r[2][0] = v[1];
  r[2][1] = -v[0];
}

 void madd_m3_m3fl(float r[3][3], const float m[3][3], float f)
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

void cloth_parallel_transport_hair_frame(float mat[3][3],
										 const float dir_old[3],
										 const float dir_new[3])
{
  float rot[3][3];

  /* rotation between segments */
  rotation_between_vecs_to_mat3(rot, dir_old, dir_new);

  /* rotate the frame */
  mul_m3_m3m3(mat, rot, mat);
}

/* Add a shear and a bend spring between two verts within a poly. */
static bool cloth_add_shear_bend_spring(ClothModifierData *clmd,
										LinkNodePair *edgelist,
										const MLoop *mloop,
										const MPoly *mpoly,
										int i,
										int j,
										int k)
{
  Cloth *cloth = clmd->clothObject;
  ClothSpring *spring;
  const MLoop *tmp_loop;
  float shrink_factor;
  int x, y;

  /* Combined shear/bend properties. */
  spring = (ClothSpring *)MEM_callocN(sizeof(ClothSpring), "cloth spring");

  if (!spring) {
	return false;
  }

  spring_verts_ordered_set(
	  spring, mloop[mpoly[i].loopstart + j].v, mloop[mpoly[i].loopstart + k].v);

  shrink_factor = cloth_shrink_factor(clmd, cloth->verts, spring->ij, spring->kl);
  spring->restlen = len_v3v3(cloth->verts[spring->kl].xrest, cloth->verts[spring->ij].xrest) *
					shrink_factor;
  spring->type |= CLOTH_SPRING_TYPE_SHEAR;
  spring->lin_stiffness = (cloth->verts[spring->kl].shear_stiff +
						   cloth->verts[spring->ij].shear_stiff) /
						  2.0f;

  if (edgelist) {
	BLI_linklist_append(&edgelist[spring->ij], spring);
	BLI_linklist_append(&edgelist[spring->kl], spring);
  }

  /* Bending specific properties. */
  if (clmd->sim_parms->bending_model == CLOTH_BENDING_ANGULAR) {
	spring->type |= CLOTH_SPRING_TYPE_BENDING;

	spring->la = k - j + 1;
	spring->lb = mpoly[i].totloop - k + j + 1;

	spring->pa = (int*)MEM_mallocN(sizeof(*spring->pa) * spring->la, "spring poly");
	if (!spring->pa) {
	  return false;
	}

	spring->pb = (int*)MEM_mallocN(sizeof(*spring->pb) * spring->lb, "spring poly");
	if (!spring->pb) {
	  return false;
	}

	tmp_loop = mloop + mpoly[i].loopstart;

	for (x = 0; x < spring->la; x++) {
	  spring->pa[x] = tmp_loop[j + x].v;
	}

	for (x = 0; x <= j; x++) {
	  spring->pb[x] = tmp_loop[x].v;
	}

	for (y = k; y < mpoly[i].totloop; x++, y++) {
	  spring->pb[x] = tmp_loop[y].v;
	}

	spring->mn = -1;

	spring->restang = cloth_spring_angle(
		cloth->verts, spring->ij, spring->kl, spring->pa, spring->pb, spring->la, spring->lb);

	spring->ang_stiffness = (cloth->verts[spring->ij].bend_stiff +
							 cloth->verts[spring->kl].bend_stiff) /
							2.0f;
  }

  BLI_linklist_prepend(&cloth->springs, spring);

  return true;
}

 bool cloth_bend_set_poly_vert_array(int **poly, int len, const MLoop *mloop)
{
  int *p = (int*)MEM_mallocN(sizeof(int) * len, "spring poly");

  if (!p) {
	return false;
  }

  for (int i = 0; i < len; i++, mloop++) {
	p[i] = mloop->v;
  }

  *poly = p;

  return true;
}

static bool find_internal_spring_target_vertex(BVHTreeFromMesh *treedata,
											   uint v_idx,
											   RNG *rng,
											   float max_length,
											   float max_diversion,
											   bool check_normal,
											   uint *r_tar_v_idx)
{
  float co[3], no[3], new_co[3];
  float radius;

  copy_v3_v3(co, treedata->vert[v_idx].co);
  normal_short_to_float_v3(no, treedata->vert[v_idx].no);
  negate_v3(no);

  float vec_len = sin(max_diversion);
  float offset[3];

  offset[0] = 0.5f - BLI_rng_get_float(rng);
  offset[1] = 0.5f - BLI_rng_get_float(rng);
  offset[2] = 0.5f - BLI_rng_get_float(rng);

  normalize_v3(offset);
  mul_v3_fl(offset, vec_len);
  add_v3_v3(no, offset);
  normalize_v3(no);

  /* Nudge the start point so we do not hit it with the ray. */
  copy_v3_v3(new_co, no);
  mul_v3_fl(new_co, FLT_EPSILON);
  add_v3_v3(new_co, co);

  radius = 0.0f;
  if (max_length == 0.0f) {
	max_length = FLT_MAX;
  }

  BVHTreeRayHit rayhit = {0};
  rayhit.index = -1;
  rayhit.dist = max_length;

  BLI_bvhtree_ray_cast(
	  treedata->tree, new_co, no, radius, &rayhit, treedata->raycast_callback, treedata);

  uint vert_idx = -1;
  const MLoop *mloop = treedata->loop;
  const MLoopTri *lt = NULL;

  if (rayhit.index != -1 && rayhit.dist <= max_length) {
	if (check_normal && dot_v3v3(rayhit.no, no) < 0.0f) {
	  /* We hit a point that points in the same direction as our starting point. */
	  return false;
	}

	float min_len = FLT_MAX;
	lt = &treedata->looptri[rayhit.index];

	for (int i = 0; i < 3; i++) {
	  uint tmp_vert_idx = mloop[lt->tri[i]].v;
	  if (tmp_vert_idx == v_idx) {
		/* We managed to hit ourselves. */
		return false;
	  }

	  float len = len_v3v3(co, rayhit.co);
	  if (len < min_len) {
		min_len = len;
		vert_idx = tmp_vert_idx;
	  }
	}

	*r_tar_v_idx = vert_idx;
	return true;
  }

  return false;
}

static ModifierData* modifier_allocate_and_init(int type)
{
	BKE_modifier_init();
	const ModifierTypeInfo* mti = BKE_modifier_get_info((ModifierType)type);
	auto md = static_cast<ModifierData*>(MEM_callocN(mti->structSize, mti->structName));

	/* NOTE: this name must be made unique later. */
	strncpy(md->name, mti->name, sizeof(md->name));

	md->type = type;
	md->mode = eModifierMode_Realtime | eModifierMode_Render;
	md->flag = eModifierFlag_OverrideLibrary_Local;
	md->ui_expand_flag = 1; /* Only open the main panel at the beginning, not the sub-panels. */

	if (mti->flags & eModifierTypeFlag_EnableInEditmode) 
	{
		md->mode |= eModifierMode_Editmode;
	}

	if (mti->initData) 
	{
		mti->initData(md);
	}

	return md;
}

bool cloth_build_springs(ClothModifierData *clmd, Mesh *mesh)
{
	Cloth *cloth = clmd->clothObject;
	ClothSpring *spring = nullptr, *tspring = nullptr, *tspring2 = nullptr;
	uint struct_springs = 0, shear_springs = 0, bend_springs = 0, struct_springs_real = 0;
	uint mvert_num = (uint)mesh->totvert;
	uint numedges = (uint)mesh->totedge;
	uint numpolys = (uint)mesh->totpoly;
	float shrink_factor;
	const MEdge *medge = mesh->medge;
	const MPoly *mpoly = mesh->mpoly;
	const MLoop *mloop = mesh->mloop;
	int index2 = 0; /* our second vertex index */
	LinkNodePair *edgelist = nullptr;
	EdgeSet *edgeset = nullptr;
	LinkNode *search = nullptr, *search2 = nullptr;
	BendSpringRef *spring_ref = nullptr;

	printf("\ntotvert = %d", mesh->totvert);
	printf("\ntotedge = %d", mesh->totedge);
	printf("\ntotpoly = %d\n", mesh->totpoly);

	/* error handling */
	if (numedges == 0) 
	{
		printf("Error handling\n");
		return false;
	}

	/* NOTE: handling ownership of springs and edgeset is quite sloppy
	* currently they are never initialized but assert just to be sure */
	BLI_assert(cloth->springs == nullptr);
	BLI_assert(cloth->edgeset == nullptr);

	printf("Assert is ok\n");

	cloth->springs = nullptr;
	cloth->edgeset = nullptr;

	if (clmd->sim_parms->bending_model == CLOTH_BENDING_ANGULAR) 
	{
		spring_ref = (BendSpringRef*)MEM_callocN(sizeof(*spring_ref) * numedges, "temp bend spring reference");

		if (!spring_ref) 
		{
			return false;
		}
	}
	else 
	{
		edgelist = (LinkNodePair*)MEM_callocN(sizeof(*edgelist) * mvert_num, "cloth_edgelist_alloc");

		if (!edgelist) 
		{
			return false;
		}
	}

	printf("\MEM_callocN is DONE");

	bool use_internal_springs = (clmd->sim_parms->flags & CLOTH_SIMSETTINGS_FLAG_INTERNAL_SPRINGS);

	if (use_internal_springs && numpolys > 0) 
	{
		BVHTreeFromMesh treedata = {NULL};
		uint tar_v_idx = 0;
		Mesh *tmp_mesh = nullptr;
		RNG *rng = nullptr;

		/* If using the rest shape key, it's necessary to make a copy of the mesh. */
		if (clmd->sim_parms->shapekey_rest && !(clmd->sim_parms->flags & CLOTH_SIMSETTINGS_FLAG_DYNAMIC_BASEMESH)) 
		{
			tmp_mesh = cloth_make_rest_mesh(clmd, mesh);
			//BKE_mesh_calc_normals(tmp_mesh);
		}

		auto existing_vert_pairs = BLI_edgeset_new("cloth_sewing_edges_graph");
		BKE_bvhtree_from_mesh_get(&treedata, tmp_mesh ? tmp_mesh : mesh, BVHTREE_FROM_LOOPTRI, 2);
		rng = BLI_rng_new_srandom(0);

		for (int i = 0; i < mvert_num; i++) 
		{
			if (find_internal_spring_target_vertex(&treedata, i, rng,
					clmd->sim_parms->internal_spring_max_length, clmd->sim_parms->internal_spring_max_diversion,
					(clmd->sim_parms->flags & CLOTH_SIMSETTINGS_FLAG_INTERNAL_SPRINGS_NORMAL), &tar_v_idx)) 
			{
				if (BLI_edgeset_haskey(existing_vert_pairs, i, tar_v_idx)) 
				{
					/* We have already created a spring between these verts! */
					continue;
				}

				BLI_edgeset_insert(existing_vert_pairs, i, tar_v_idx);

				spring = (ClothSpring *)MEM_callocN(sizeof(ClothSpring), "cloth spring");

				if (spring) 
				{
					spring_verts_ordered_set(spring, i, tar_v_idx);

					shrink_factor = cloth_shrink_factor(clmd, cloth->verts, spring->ij, spring->kl);
					spring->restlen = len_v3v3(cloth->verts[spring->kl].xrest, cloth->verts[spring->ij].xrest) * shrink_factor;
					spring->lin_stiffness = (cloth->verts[spring->kl].internal_stiff + cloth->verts[spring->ij].internal_stiff) / 2.0f;
					spring->type = CLOTH_SPRING_TYPE_INTERNAL;
					spring->flags = 0;

					BLI_linklist_prepend(&cloth->springs, spring);

					if (spring_ref) 
					{
						spring_ref[i].spring = spring;
					}
				}
				else 
				{
					cloth_free_errorsprings(cloth, edgelist, spring_ref);
					BLI_edgeset_free(existing_vert_pairs);
					free_bvhtree_from_mesh(&treedata);
					if (tmp_mesh) 
					{
						BKE_id_free(NULL, &tmp_mesh->id);
					}
					return false;
				}
			}
		}
		BLI_edgeset_free(existing_vert_pairs);
		free_bvhtree_from_mesh(&treedata);
		if (tmp_mesh) 
		{
			BKE_id_free(NULL, &tmp_mesh->id);
		}
		BLI_rng_free(rng);
	}

	clmd->sim_parms->avg_spring_len = 0.0f;
	for (int i = 0; i < mvert_num; i++) 
	{
		cloth->verts[i].avg_spring_len = 0.0f;
	}

	if (clmd->sim_parms->flags & CLOTH_SIMSETTINGS_FLAG_SEW) 
	{
		/* cloth->sew_edge_graph should not exist before this */
		BLI_assert(cloth->sew_edge_graph == NULL);
		cloth->sew_edge_graph = BLI_edgeset_new("cloth_sewing_edges_graph");
	}

	/* Structural springs. */
	for (int i = 0; i < numedges; i++) 
	{
		spring = (ClothSpring *)MEM_callocN(sizeof(ClothSpring), "cloth spring");

		if (spring) 
		{
			spring_verts_ordered_set(spring, medge[i].v1, medge[i].v2);
			if (clmd->sim_parms->flags & CLOTH_SIMSETTINGS_FLAG_SEW && medge[i].flag & ME_LOOSEEDGE) 
			{
				/* handle sewing (loose edges will be pulled together) */
				spring->restlen = 0.0f;
				spring->lin_stiffness = 1.0f;
				spring->type = CLOTH_SPRING_TYPE_SEWING;

				BLI_edgeset_insert(cloth->sew_edge_graph, medge[i].v1, medge[i].v2);
			}
			else 
			{
				shrink_factor = cloth_shrink_factor(clmd, cloth->verts, spring->ij, spring->kl);
				spring->restlen = len_v3v3(cloth->verts[spring->kl].xrest,
											cloth->verts[spring->ij].xrest) *
									shrink_factor;
				spring->lin_stiffness = (cloth->verts[spring->kl].struct_stiff +
											cloth->verts[spring->ij].struct_stiff) /
										2.0f;
				spring->type = CLOTH_SPRING_TYPE_STRUCTURAL;

				clmd->sim_parms->avg_spring_len += spring->restlen;
				cloth->verts[spring->ij].avg_spring_len += spring->restlen;
				cloth->verts[spring->kl].avg_spring_len += spring->restlen;
				cloth->verts[spring->ij].spring_count++;
				cloth->verts[spring->kl].spring_count++;
				struct_springs_real++;
			}

			spring->flags = 0;
			struct_springs++;

			BLI_linklist_prepend(&cloth->springs, spring);

			if (spring_ref) 
			{
				spring_ref[i].spring = spring;
			}
		}
		else 
		{
			cloth_free_errorsprings(cloth, edgelist, spring_ref);
			return false;
		}
	}

	if (struct_springs_real > 0) 
	{
		clmd->sim_parms->avg_spring_len /= struct_springs_real;
	}

	for (int i = 0; i < mvert_num; i++) 
	{
		if (cloth->verts[i].spring_count > 0) 
		{
			cloth->verts[i].avg_spring_len = cloth->verts[i].avg_spring_len * 0.49f / ((float)cloth->verts[i].spring_count);
		}
	}

	edgeset = BLI_edgeset_new_ex(__func__, numedges);
	cloth->edgeset = edgeset;

	if (numpolys) 
	{
		for (int i = 0; i < numpolys; i++) 
		{
			/* Shear springs. */
			/* Triangle faces already have shear springs due to structural geometry. */
			if (mpoly[i].totloop > 3) 
			{
				for (int j = 1; j < mpoly[i].totloop - 1; j++) 
				{
					if (j > 1) 
					{
						if (cloth_add_shear_bend_spring(clmd, edgelist, mloop, mpoly, i, 0, j)) 
						{
							shear_springs++;

							if (clmd->sim_parms->bending_model == CLOTH_BENDING_ANGULAR) 
							{
							bend_springs++;
							}
						}
						else 
						{
							cloth_free_errorsprings(cloth, edgelist, spring_ref);
							return false;
						}
					}

				for (int k = j + 2; k < mpoly[i].totloop; k++) 
					{
					if (cloth_add_shear_bend_spring(clmd, edgelist, mloop, mpoly, i, j, k)) 
					{
						shear_springs++;

						if (clmd->sim_parms->bending_model == CLOTH_BENDING_ANGULAR) 
						{
							bend_springs++;
						}
					}
					else 
					{
						cloth_free_errorsprings(cloth, edgelist, spring_ref);
						return false;
					}
				}
			}
		}

		/* Angular bending springs along struct springs. */
		if (clmd->sim_parms->bending_model == CLOTH_BENDING_ANGULAR) 
		{
			const MLoop *ml = mloop + mpoly[i].loopstart;

			for (int j = 0; j < mpoly[i].totloop; j++, ml++) 
			{
				BendSpringRef *curr_ref = &spring_ref[ml->e];
				curr_ref->polys++;

				/* First poly found for this edge, store poly index. */
				if (curr_ref->polys == 1) 
				{
					curr_ref->index = i;
				}
				/* Second poly found for this edge, add bending data. */
				else if (curr_ref->polys == 2) 
				{
					spring = curr_ref->spring;

					spring->type |= CLOTH_SPRING_TYPE_BENDING;

					spring->la = mpoly[curr_ref->index].totloop;
					spring->lb = mpoly[i].totloop;

					if (!cloth_bend_set_poly_vert_array(&spring->pa, spring->la, &mloop[mpoly[curr_ref->index].loopstart]) ||
						!cloth_bend_set_poly_vert_array(&spring->pb, spring->lb, &mloop[mpoly[i].loopstart])) 
					{
						cloth_free_errorsprings(cloth, edgelist, spring_ref);
						return false;
					}

					spring->mn = ml->e;

					spring->restang = cloth_spring_angle(cloth->verts,
															spring->ij, spring->kl,
															spring->pa, spring->pb,
															spring->la, spring->lb);

					spring->ang_stiffness = (cloth->verts[spring->ij].bend_stiff + cloth->verts[spring->kl].bend_stiff) / 2.0f;

					bend_springs++;
				}
				/* Third poly found for this edge, remove bending data. */
				else if (curr_ref->polys == 3) 
				{
					spring = curr_ref->spring;

					spring->type &= ~CLOTH_SPRING_TYPE_BENDING;
					MEM_freeN(spring->pa);
					MEM_freeN(spring->pb);
					spring->pa = NULL;
					spring->pb = NULL;

					bend_springs--;
				}
			}
		}
	}

	/* Linear bending springs. */
	if (clmd->sim_parms->bending_model == CLOTH_BENDING_LINEAR) 
	{
		search2 = cloth->springs;

		for (int i = struct_springs; i < struct_springs + shear_springs; i++) 
		{
			if (!search2) 
			{
				break;
			}

			tspring2 = (ClothSpring*)search2->link;
			search = edgelist[tspring2->kl].list;

			while (search) 
			{
				tspring = (ClothSpring*)search->link;
				index2 = ((tspring->ij == tspring2->kl) ? (tspring->kl) : (tspring->ij));

				/* Check for existing spring. */
				/* Check also if startpoint is equal to endpoint. */
				if ((index2 != tspring2->ij) && !BLI_edgeset_haskey(edgeset, tspring2->ij, index2)) 
				{
					spring = (ClothSpring *)MEM_callocN(sizeof(ClothSpring), "cloth spring");

					if (!spring) 
					{
						cloth_free_errorsprings(cloth, edgelist, spring_ref);
						return false;
					}

					spring_verts_ordered_set(spring, tspring2->ij, index2);
					shrink_factor = cloth_shrink_factor(clmd, cloth->verts, spring->ij, spring->kl);
					spring->restlen = len_v3v3(cloth->verts[spring->kl].xrest,
												cloth->verts[spring->ij].xrest) *
										shrink_factor;
					spring->type = CLOTH_SPRING_TYPE_BENDING;
					spring->lin_stiffness = (cloth->verts[spring->kl].bend_stiff +
												cloth->verts[spring->ij].bend_stiff) /
											2.0f;
					BLI_edgeset_insert(edgeset, spring->ij, spring->kl);
					bend_springs++;

					BLI_linklist_prepend(&cloth->springs, spring);
				}

				search = search->next;
			}

			search2 = search2->next;
			}
		}
	}
	else if (struct_springs > 2) 
	{
		/* bending springs for hair strands
			* The current algorithm only goes through the edges in order of the mesh edges list
			* and makes springs between the outer vert of edges sharing a vertice. This works just
			* fine for hair, but not for user generated string meshes. This could/should be later
			* extended to work with non-ordered edges so that it can be used for general "rope
			* dynamics" without the need for the vertices or edges to be ordered through the length
			* of the strands. -jahka */
		search = cloth->springs;
		search2 = search->next;
		while (search && search2) 
		{
			tspring = (ClothSpring*)search->link;
			tspring2 = (ClothSpring*)search2->link;

			if (tspring->ij == tspring2->kl) 
			{
				spring = (ClothSpring*)MEM_callocN(sizeof(ClothSpring), "cloth spring");

				if (!spring) 
				{
					cloth_free_errorsprings(cloth, edgelist, spring_ref);
					return false;
				}

				spring->ij = tspring2->ij;
				spring->kl = tspring->kl;
				spring->restlen = len_v3v3(cloth->verts[spring->kl].xrest,
					cloth->verts[spring->ij].xrest);
				spring->type = CLOTH_SPRING_TYPE_BENDING;
				spring->lin_stiffness = (cloth->verts[spring->kl].bend_stiff + cloth->verts[spring->ij].bend_stiff) / 2.0f;
				bend_springs++;

				BLI_linklist_prepend(&cloth->springs, spring);
			}

			search = search->next;
			search2 = search2->next;
		}

		cloth_hair_update_bending_rest_targets(clmd);
	}

	/* note: the edges may already exist so run reinsert */

	/* insert other near springs in edgeset AFTER bending springs are calculated (for selfcolls) */
	for (int i = 0; i < numedges; i++) 
	{ /* struct springs */
		BLI_edgeset_add(edgeset, medge[i].v1, medge[i].v2);
	}

	for (int i = 0; i < numpolys; i++) 
	{ /* edge springs */
		if (mpoly[i].totloop == 4) 
		{
			BLI_edgeset_add(edgeset, mloop[mpoly[i].loopstart + 0].v, mloop[mpoly[i].loopstart + 2].v);
			BLI_edgeset_add(edgeset, mloop[mpoly[i].loopstart + 1].v, mloop[mpoly[i].loopstart + 3].v);
		}
	}

	MEM_SAFE_FREE(spring_ref);

	cloth->numsprings = struct_springs + shear_springs + bend_springs;

	cloth_free_edgelist(edgelist, mvert_num);
	return true;
}

bool cloth_from_object(ClothModifierData* clmd, Mesh* mesh)
{
	int i = 0;
	MVert* mvert = NULL;
	ClothVertex* verts = NULL;
	const float(*shapekey_rest)[3] = NULL;
	const float tnull[3] = { 0, 0, 0 };
	clmd->clothObject = nullptr;

	/* If we have a clothObject, free it. */
	//if (clmd->clothObject != NULL) 
	//{
	//	cloth_free_modifier(clmd);
	//}

	/* Allocate a new cloth object. */
	clmd->clothObject = (Cloth*)MEM_callocN(sizeof(Cloth), "cloth");
	if (clmd->clothObject) 
	{
		//clmd->clothObject->old_solver_type = 255;
		clmd->clothObject->edgeset = NULL;
	}
	else 
	{
		printf("Out of memory on allocating clmd->clothObject");
		//BKE_modifier_set_error(ob, &(clmd->modifier), "Out of memory on allocating clmd->clothObject");
		return false;
	}

	/* mesh input objects need Mesh */
	if (!mesh) 
	{
		return false;
	}

	cloth_from_mesh(clmd, mesh);

	/* create springs */
	clmd->clothObject->springs = NULL;
	clmd->clothObject->numsprings = -1;

	clmd->clothObject->sew_edge_graph = NULL;

	//if (clmd->sim_parms->shapekey_rest && !(clmd->sim_parms->flags & CLOTH_SIMSETTINGS_FLAG_DYNAMIC_BASEMESH)) 
	//{
	//	shapekey_rest = CustomData_get_layer(&mesh->vdata, CD_CLOTH_ORCO);
	//}

	mvert = mesh->mvert;

	verts = clmd->clothObject->verts;

	/* set initial values */
	for (i = 0; i < mesh->totvert; i++, verts++) 
	{
		/*if (first) 
		{
			copy_v3_v3(verts->x, mvert[i].co);

			mul_m4_v3(ob->obmat, verts->x);

			if (shapekey_rest) 
			{
				copy_v3_v3(verts->xrest, shapekey_rest[i]);
				mul_m4_v3(ob->obmat, verts->xrest);
			}
			else 
			{
				copy_v3_v3(verts->xrest, verts->x);
			}
		}*/

		/* no GUI interface yet */
		verts->mass = clmd->sim_parms->mass;
		verts->impulse_count = 0;

		if (clmd->sim_parms->vgroup_mass > 0) 
		{
			verts->goal = clmd->sim_parms->defgoal;
		}
		else 
		{
			verts->goal = 0.0f;
		}

		verts->shrink_factor = 0.0f;

		verts->flags = 0;
		copy_v3_v3(verts->xold, verts->x);
		copy_v3_v3(verts->xconst, verts->x);
		copy_v3_v3(verts->txold, verts->x);
		copy_v3_v3(verts->tx, verts->x);
		mul_v3_fl(verts->v, 0.0f);

		verts->impulse_count = 0;
		copy_v3_v3(verts->impulse, tnull);
	}

	/* apply / set vertex groups */
	/* has to be happen before springs are build! */
	//cloth_apply_vgroup(clmd, mesh);

	if (!cloth_build_springs(clmd, mesh)) 
	{
		cloth_free_modifier(clmd);
		printf("Cannot build springs");
		//BKE_modifier_set_error(ob, &(clmd->modifier), "Cannot build springs");
		return false;
	}

	/* init our solver */
	SIM_cloth_solver_init(clmd);

	//if (!first) 
	//{
	SIM_cloth_solver_set_positions(clmd);
	//}

	clmd->clothObject->bvhtree = bvhtree_build_from_cloth(clmd, clmd->coll_parms->epsilon);
	clmd->clothObject->bvhselftree = bvhtree_build_from_cloth(clmd, clmd->coll_parms->selfepsilon);

	return true;
}

/**
 * Ensure the array is large enough
 *
 * \note This function must always be thread-protected by caller.
 * It should only be used by internal code.
 */
static void mesh_ensure_looptri_data(Mesh* mesh)
{
	/* This is a ported copy of `DM_ensure_looptri_data(dm)`. */
	const uint totpoly = mesh->totpoly;
	const int looptris_len = poly_to_tri_count(totpoly, mesh->totloop);

	BLI_assert(mesh->runtime->looptris.array_wip == nullptr);

	swap(mesh->runtime->looptris.array, mesh->runtime->looptris.array_wip);

	if ((looptris_len > mesh->runtime->looptris.len_alloc) ||
		(looptris_len < mesh->runtime->looptris.len_alloc * 2) || (totpoly == 0)) 
	{
		MEM_SAFE_FREE(mesh->runtime->looptris.array_wip);
		mesh->runtime->looptris.len_alloc = 0;
		mesh->runtime->looptris.len = 0;
	}

	if (totpoly) 
	{
		if (mesh->runtime->looptris.array_wip == nullptr) 
		{
			mesh->runtime->looptris.array_wip = static_cast<MLoopTri*>(MEM_malloc_arrayN(looptris_len, sizeof(*mesh->runtime->looptris.array_wip), __func__));
			mesh->runtime->looptris.len_alloc = looptris_len;
		}

		mesh->runtime->looptris.len = looptris_len;
	}
}

bool is_quad_flip_v3_first_third_fast_with_normal(const float v1[3],
	const float v2[3], const float v3[3], const float v4[3],
	const float normal[3])
{
	float dir_v3v1[3], tangent[3];
	sub_v3_v3v3(dir_v3v1, v3, v1);
	cross_v3_v3v3(tangent, dir_v3v1, normal);
	const float dot = dot_v3v3(v1, tangent);
	return (dot_v3v3(v4, tangent) >= dot) || (dot_v3v3(v2, tangent) <= dot);
}

enum {
	KDNODE_FLAG_REMOVED = (1 << 0),
};

#ifdef USE_KDTREE
#  define KDNODE_UNSET ((uint32_t)-1)

static bool kdtree2d_isect_tri_recursive(const struct KDTree2D* tree,
	const uint32_t tri_index[3],
	const float* tri_coords[3],
	const float tri_center[2],
	const struct KDRange2D bounds[2],
	const KDTreeNode2D* node)
{
	const float* co = tree->coords[node->index];

	/* bounds then triangle intersect */
	if ((node->flag & KDNODE_FLAG_REMOVED) == 0) {
		/* bounding box test first */
		if ((co[0] >= bounds[0].min) && (co[0] <= bounds[0].max) && (co[1] >= bounds[1].min) &&
			(co[1] <= bounds[1].max)) {
			if ((span_tri_v2_sign(tri_coords[0], tri_coords[1], co) != CONCAVE) &&
				(span_tri_v2_sign(tri_coords[1], tri_coords[2], co) != CONCAVE) &&
				(span_tri_v2_sign(tri_coords[2], tri_coords[0], co) != CONCAVE)) {
				if (!ELEM(node->index, tri_index[0], tri_index[1], tri_index[2])) {
					return true;
				}
			}
		}
	}

#  define KDTREE2D_ISECT_TRI_RECURSE_NEG \
    (((node->neg != KDNODE_UNSET) && (co[node->axis] >= bounds[node->axis].min)) && \
     kdtree2d_isect_tri_recursive(tree, tri_index, tri_coords, tri_center, bounds, &tree->nodes[node->neg]))
#  define KDTREE2D_ISECT_TRI_RECURSE_POS \
    (((node->pos != KDNODE_UNSET) && (co[node->axis] <= bounds[node->axis].max)) && \
     kdtree2d_isect_tri_recursive(tree, tri_index, tri_coords, tri_center, bounds, &tree->nodes[node->pos]))

	if (tri_center[node->axis] > co[node->axis]) {
		if (KDTREE2D_ISECT_TRI_RECURSE_POS) {
			return true;
		}
		if (KDTREE2D_ISECT_TRI_RECURSE_NEG) {
			return true;
		}
	}
	else {
		if (KDTREE2D_ISECT_TRI_RECURSE_NEG) {
			return true;
		}
		if (KDTREE2D_ISECT_TRI_RECURSE_POS) {
			return true;
		}
	}

#  undef KDTREE2D_ISECT_TRI_RECURSE_NEG
#  undef KDTREE2D_ISECT_TRI_RECURSE_POS

	BLI_assert(node->index != KDNODE_UNSET);

	return false;
}


static void kdtree2d_node_remove(struct KDTree2D* tree, uint32_t index)
{
	uint32_t node_index = tree->nodes_map[index];
	KDTreeNode2D* node;

	if (node_index == KDNODE_UNSET) 
	{
		return;
	}

	tree->nodes_map[index] = KDNODE_UNSET;

	node = &tree->nodes[node_index];
	tree->node_num -= 1;

	BLI_assert((node->flag & KDNODE_FLAG_REMOVED) == 0);
	node->flag |= KDNODE_FLAG_REMOVED;

	while ((node->neg == KDNODE_UNSET) && (node->pos == KDNODE_UNSET) &&
		(node->parent != KDNODE_UNSET)) 
	{
		KDTreeNode2D* node_parent = &tree->nodes[node->parent];

		BLI_assert((uint32_t)(node - tree->nodes) == node_index);
		if (node_parent->neg == node_index) 
		{
			node_parent->neg = KDNODE_UNSET;
		}
		else 
		{
			BLI_assert(node_parent->pos == node_index);
			node_parent->pos = KDNODE_UNSET;
		}

		if (node_parent->flag & KDNODE_FLAG_REMOVED) 
		{
			node_index = node->parent;
			node = node_parent;
		}
		else 
		{
			break;
		}
	}
}

static bool kdtree2d_isect_tri(struct KDTree2D* tree, const uint32_t ind[3])
{
	const float* vs[3];
	uint32_t i;
	KDRange2D bounds[2] = 
	{
		{FLT_MAX, -FLT_MAX},
		{FLT_MAX, -FLT_MAX},
	};
	float tri_center[2] = { 0.0f, 0.0f };

	for (i = 0; i < 3; i++) {
		vs[i] = tree->coords[ind[i]];

		add_v2_v2(tri_center, vs[i]);

		CLAMP_MAX(bounds[0].min, vs[i][0]);
		CLAMP_MIN(bounds[0].max, vs[i][0]);
		CLAMP_MAX(bounds[1].min, vs[i][1]);
		CLAMP_MIN(bounds[1].max, vs[i][1]);
	}

	mul_v2_fl(tri_center, 1.0f / 3.0f);

	return kdtree2d_isect_tri_recursive(tree, ind, vs, tri_center, bounds, &tree->nodes[tree->root]);
}

#endif /* USE_KDTREE */

static bool pf_ear_tip_check(PolyFill* pf, PolyIndex* pi_ear_tip, const eSign sign_accept)
{
#ifndef USE_KDTREE
	/* localize */
	const float(*coords)[2] = pf->coords;
	PolyIndex* pi_curr;

	const float* v1, * v2, * v3;
#endif

#if defined(USE_CONVEX_SKIP) && !defined(USE_KDTREE)
	uint32_t coords_num_concave_checked = 0;
#endif

#ifdef USE_CONVEX_SKIP

#  ifdef USE_CONVEX_SKIP_TEST
	/* check if counting is wrong */
	{
		uint32_t coords_num_concave_test = 0;
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

	if (UNLIKELY(pi_ear_tip->sign != sign_accept)) {
		return false;
	}

#ifdef USE_KDTREE
	{
		const uint32_t ind[3] = { pi_ear_tip->index, pi_ear_tip->next->index, pi_ear_tip->prev->index };

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
			if ((span_tri_v2_sign(v3, v1, v) != CONCAVE) && (span_tri_v2_sign(v1, v2, v) != CONCAVE) &&
				(span_tri_v2_sign(v2, v3, v) != CONCAVE)) {
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


/**
 * \return CONCAVE, TANGENTIAL or CONVEX
 */
static void pf_coord_sign_calc(PolyFill* pf, PolyIndex* pi)
{
	/* localize */
	const float(*coords)[2] = pf->coords;

	pi->sign = span_tri_v2_sign(coords[pi->prev->index], coords[pi->index], coords[pi->next->index]);
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
	const uint32_t coords_num = pf->coords_num;
	PolyIndex* pi_ear;

	uint32_t i;

	/* Use two passes when looking for an ear.
	 *
	 * - The first pass only picks *good* (concave) choices.
	 *   For polygons which aren't degenerate this works well
	 *   since it avoids creating any zero area faces.
	 *
	 * - The second pass is only met if no concave choices are possible,
	 *   so the cost of a second pass is only incurred for degenerate polygons.
	 *   In this case accept zero area faces as better alternatives aren't available.
	 *
	 * See: #103913 for reference.
	 *
	 * NOTE: these passes draw a distinction between zero area faces and concave
	 * which is susceptible minor differences in float precision
	 * (since #TANGENTIAL compares with 0.0f).
	 *
	 * While it's possible to compute an error threshold and run a pass that picks
	 * ears which are more likely not to appear as zero area from a users perspective,
	 * this API prioritizes performance (for real-time updates).
	 * Higher quality tessellation can always be achieved using #BLI_polyfill_beautify.
	 */
	for (eSign sign_accept = CONVEX; sign_accept >= TANGENTIAL; sign_accept--) {
#ifdef USE_CLIP_EVEN
		pi_ear = pi_ear_init;
#else
		pi_ear = pf->indices;
#endif
		i = coords_num;
		while (i--) {
			if (pf_ear_tip_check(pf, pi_ear, sign_accept)) {
				return pi_ear;
			}
#ifdef USE_CLIP_SWEEP
			pi_ear = reverse ? pi_ear->prev : pi_ear->next;
#else
			pi_ear = pi_ear->next;
#endif
		}
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

	for (i = 0; i < coords_num; i++) {
		PolyIndex* pi = &indices[i];
		pf_coord_sign_calc(pf, pi);
#ifdef USE_CONVEX_SKIP
		if (pi->sign != CONVEX) {
			pf->coords_num_concave += 1;
		}
#endif
	}
}

eSign signum_enum(float a)
{
	if (a > 0.0f) {
		return CONVEX;
	}
	if (UNLIKELY(a == 0.0f)) {
		return TANGENTIAL;
	}
	return CONCAVE;
}

float area_tri_signed_v2_alt_2x(const float v1[2], const float v2[2], const float v3[2])
{
	float d2[2], d3[2];
	sub_v2_v2v2(d2, v2, v1);
	sub_v2_v2v2(d3, v3, v1);
	return (d2[0] * d3[1]) - (d3[0] * d2[1]);
}

static eSign span_tri_v2_sign(const float v1[2], const float v2[2], const float v3[2])
{
	return signum_enum(area_tri_signed_v2_alt_2x(v3, v2, v1));
}

static uint* pf_tri_add(PolyFill* pf)
{
	return pf->tris[pf->tris_num++];
}

static void pf_coord_remove(PolyFill* pf, PolyIndex* pi)
{
#ifdef USE_KDTREE
	/* avoid double lookups, since convex coords are ignored when testing intersections */
	if (pf->kdtree.node_num) 
	{
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
	/* localize */
	PolyIndex* pi_ear;

#ifdef USE_CLIP_EVEN
	PolyIndex* pi_ear_init = pf->indices;
#endif
#ifdef USE_CLIP_SWEEP
	bool reverse = false;
#endif

	while (pf->coords_num > 3) {
		PolyIndex* pi_prev, * pi_next;
		eSign sign_orig_prev, sign_orig_next;

		pi_ear = pf_ear_tip_find(pf
#ifdef USE_CLIP_EVEN
			, pi_ear_init
#endif
#ifdef USE_CLIP_SWEEP
			, reverse
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
#else
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

static void kdtree2d_new(struct KDTree2D* tree, uint32_t tot, const float(*coords)[2])
{
	/* set by caller */
	// tree->nodes = nodes;
	tree->coords = coords;
	tree->root = KDNODE_UNSET;
	tree->node_num = tot;
}

/**
 * no need for kdtree2d_insert, since we know the coords array.
 */
static void kdtree2d_init(struct KDTree2D* tree,
	const uint32_t coords_num,
	const PolyIndex* indices)
{
	KDTreeNode2D* node;
	uint32_t i;

	for (i = 0, node = tree->nodes; i < coords_num; i++) 
	{
		if (indices[i].sign != CONVEX) 
		{
			node->neg = node->pos = KDNODE_UNSET;
			node->index = indices[i].index;
			node->axis = 0;
			node->flag = 0;
			node++;
		}
	}

	BLI_assert(tree->node_num == (uint32_t)(node - tree->nodes));
}


static uint32_t kdtree2d_balance_recursive(KDTreeNode2D* nodes,
	uint32_t node_num,
	axis_t axis,
	const float(*coords)[2],
	const uint32_t ofs)
{
	KDTreeNode2D* node;
	uint32_t neg, pos, median, i, j;

	if (node_num <= 0) {
		return KDNODE_UNSET;
	}
	if (node_num == 1) {
		return 0 + ofs;
	}

	/* Quick-sort style sorting around median. */
	neg = 0;
	pos = node_num - 1;
	median = node_num / 2;

	while (pos > neg) {
		const float co = coords[nodes[pos].index][axis];
		i = neg - 1;
		j = pos;

		while (1) {
			while (coords[nodes[++i].index][axis] < co) { /* pass */ }
			while (coords[nodes[--j].index][axis] > co && j > neg) { /* pass */ }

			if (i >= j) 
			{
				break;
			}
			swap(nodes[i], nodes[j]);
		}

		swap(nodes[i], nodes[pos]);
		if (i >= median) {
			pos = i - 1;
		}
		if (i <= median) {
			neg = i + 1;
		}
	}

	/* Set node and sort sub-nodes. */
	node = &nodes[median];
	node->axis = axis;
	axis = !axis;
	node->neg = kdtree2d_balance_recursive(nodes, median, axis, coords, ofs);
	node->pos = kdtree2d_balance_recursive(&nodes[median + 1], (node_num - (median + 1)), axis, coords, (median + 1) + ofs);

	return median + ofs;
}


static void kdtree2d_balance(struct KDTree2D* tree)
{
	tree->root = kdtree2d_balance_recursive(tree->nodes, tree->node_num, 0, tree->coords, 0);
}

static void kdtree2d_init_mapping(struct KDTree2D* tree)
{
	uint32_t i;
	KDTreeNode2D* node;

	for (i = 0, node = tree->nodes; i < tree->node_num; i++, node++) 
	{
		if (node->neg != KDNODE_UNSET) 
		{
			tree->nodes[node->neg].parent = i;
		}
		if (node->pos != KDNODE_UNSET) 
		{
			tree->nodes[node->pos].parent = i;
		}

		/* build map */
		BLI_assert(tree->nodes_map[node->index] == KDNODE_UNSET);
		tree->nodes_map[node->index] = i;
	}

	tree->nodes[tree->root].parent = KDNODE_UNSET;
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

void BLI_polyfill_calc_arena(const float(*coords)[2],
	const uint coords_num,
	const int coords_sign,
	uint(*r_tris)[3],

	struct MemArena* arena)
{
	PolyFill pf;
	PolyIndex* indices = (PolyIndex*)BLI_memarena_alloc(arena, sizeof(*indices) * coords_num);

#ifdef DEBUG_TIME
	TIMEIT_START(polyfill2d);
#endif

	polyfill_prepare(&pf,
		coords,
		coords_num,
		coords_sign,
		r_tris,
		/* cache */
		indices);

#ifdef USE_KDTREE
	if (pf.coords_num_concave) {
		pf.kdtree.nodes = (KDTreeNode2D*)BLI_memarena_alloc(arena, sizeof(*pf.kdtree.nodes) * pf.coords_num_concave);
		pf.kdtree.nodes_map = (uint32_t*)memset(BLI_memarena_alloc(arena, sizeof(*pf.kdtree.nodes_map) * coords_num), 0xff, sizeof(*pf.kdtree.nodes_map) * coords_num);
	}
	else 
	{
		pf.kdtree.node_num = 0;
	}
#endif

	polyfill_calc(&pf);

	/* indices are no longer needed,
	 * caller can clear arena */

#ifdef DEBUG_TIME
	TIMEIT_END(polyfill2d);
#endif
}
void mesh_calc_tessellation_for_face_impl(const MLoop* mloop,
	const MPoly* mpoly,
	const float(*positions)[3],
	uint poly_index,
	MLoopTri* mlt,
	MemArena** pf_arena_p,
	const bool face_normal,
	const float normal_precalc[3])
{
	const uint mp_loopstart = uint(mpoly[poly_index].loopstart);
	const uint mp_totloop = uint(mpoly[poly_index].totloop);

#define ML_TO_MLT(i1, i2, i3) \
  { \
    ARRAY_SET_ITEMS(mlt->tri, mp_loopstart + i1, mp_loopstart + i2, mp_loopstart + i3); \
    mlt->poly = poly_index; \
  } \
  ((void)0)

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

		if (UNLIKELY(face_normal ? is_quad_flip_v3_first_third_fast_with_normal(
			/* Simpler calculation (using the normal). */
			positions[mloop[mlt_a->tri[0]].v],
			positions[mloop[mlt_a->tri[1]].v],
			positions[mloop[mlt_a->tri[2]].v],
			positions[mloop[mlt_b->tri[2]].v],
			normal_precalc) :
			is_quad_flip_v3_first_third_fast(
				/* Expensive calculation (no normal). */
				positions[mloop[mlt_a->tri[0]].v],
				positions[mloop[mlt_a->tri[1]].v],
				positions[mloop[mlt_a->tri[2]].v],
				positions[mloop[mlt_b->tri[2]].v]))) {
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
			co_prev = positions[ml[mp_totloop - 1].v];
			for (uint j = 0; j < mp_totloop; j++, ml++) {
				co_curr = positions[ml->v];
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
		if (UNLIKELY(pf_arena == nullptr)) {
			pf_arena = *pf_arena_p = BLI_memarena_new(BLI_MEMARENA_STD_BUFSIZE, __func__);
		}

		uint(*tris)[3] = static_cast<uint(*)[3]>(
			BLI_memarena_alloc(pf_arena, sizeof(*tris) * size_t(totfilltri)));
		float(*projverts)[2] = static_cast<float(*)[2]>(
			BLI_memarena_alloc(pf_arena, sizeof(*projverts) * size_t(mp_totloop)));

		ml = mloop + mp_loopstart;
		for (uint j = 0; j < mp_totloop; j++, ml++) {
			mul_v2_m3v3(projverts[j], axis_mat, positions[ml->v]);
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
	const MPoly* mpoly,
	const float(*positions)[3],
	uint poly_index,
	MLoopTri* mlt,
	MemArena** pf_arena_p)
{
	mesh_calc_tessellation_for_face_impl(
		mloop, mpoly, positions, poly_index, mlt, pf_arena_p, false, nullptr);
}

static void mesh_calc_tessellation_for_face_with_normal(const MLoop* mloop,
	const MPoly* mpoly,
	const float(*positions)[3],
	uint poly_index,
	MLoopTri* mlt,
	MemArena** pf_arena_p,
	const float normal_precalc[3])
{
	mesh_calc_tessellation_for_face_impl(
		mloop, mpoly, positions, poly_index, mlt, pf_arena_p, true, normal_precalc);
}

static void mesh_recalc_looptri__single_threaded(const MLoop* mloop,
	const MPoly* mpoly,
	const float(*positions)[3],
	int totloop,
	int totpoly,
	MLoopTri* mlooptri,
	const float(*poly_normals)[3])
{
	MemArena* pf_arena = nullptr;
	const MPoly* mp = mpoly;
	uint tri_index = 0;

	if (poly_normals != nullptr) {
		for (uint poly_index = 0; poly_index < uint(totpoly); poly_index++, mp++) {
			mesh_calc_tessellation_for_face_with_normal(mloop,
				mpoly,
				positions,
				poly_index,
				&mlooptri[tri_index],
				&pf_arena,
				poly_normals[poly_index]);
			tri_index += uint(mp->totloop - 2);
		}
	}
	else {
		for (uint poly_index = 0; poly_index < uint(totpoly); poly_index++, mp++) {
			mesh_calc_tessellation_for_face(
				mloop, mpoly, positions, poly_index, &mlooptri[tri_index], &pf_arena);
			tri_index += uint(mp->totloop - 2);
		}
	}

	if (pf_arena) {
		BLI_memarena_free(pf_arena);
		pf_arena = nullptr;
	}
	BLI_assert(tri_index == uint(poly_to_tri_count(totpoly, totloop)));
	UNUSED_VARS_NDEBUG(totloop);
}

static void mesh_calc_tessellation_for_face_free_fn(const void* __restrict /*userdata*/,
	void* __restrict tls_v)
{
	TessellationUserTLS* tls_data = static_cast<TessellationUserTLS*>(tls_v);
	if (tls_data->pf_arena) {
		BLI_memarena_free(tls_data->pf_arena);
	}
}

static void mesh_calc_tessellation_for_face_with_normal_fn(void* __restrict userdata,
	const int index,
	const TaskParallelTLS* __restrict tls)
{
	auto data = static_cast<const TessellationUserData*>(userdata);
	TessellationUserTLS* tls_data = static_cast<TessellationUserTLS*>(tls->userdata_chunk);
	const int tri_index = poly_to_tri_count(index, data->mpoly[index].loopstart);
	mesh_calc_tessellation_for_face_impl(data->mloop,
		data->mpoly,
		data->positions,
		uint(index),
		&data->mlooptri[tri_index],
		&tls_data->pf_arena,
		true,
		data->poly_normals[index]);
}

static void mesh_calc_tessellation_for_face_fn(void* __restrict userdata,
	const int index,
	const TaskParallelTLS* __restrict tls)
{
	const TessellationUserData* data = static_cast<const TessellationUserData*>(userdata);
	TessellationUserTLS* tls_data = static_cast<TessellationUserTLS*>(tls->userdata_chunk);
	const int tri_index = poly_to_tri_count(index, data->mpoly[index].loopstart);
	mesh_calc_tessellation_for_face_impl(data->mloop,
		data->mpoly,
		data->positions,
		uint(index),
		&data->mlooptri[tri_index],
		&tls_data->pf_arena,
		false,
		nullptr);
}

static void mesh_recalc_looptri__multi_threaded(const MLoop* mloop,
	const MPoly* mpoly,
	const float(*positions)[3],
	int totpoly,
	MLoopTri* mlooptri,
	const float(*poly_normals)[3])
{
	struct TessellationUserTLS tls_data_dummy = { nullptr };

	struct TessellationUserData data {};
	data.mloop = mloop;
	data.mpoly = mpoly;
	data.positions = positions;
	data.mlooptri = mlooptri;
	data.poly_normals = poly_normals;

	TaskParallelSettings settings;
	BLI_parallel_range_settings_defaults(&settings);

	settings.userdata_chunk = &tls_data_dummy;
	settings.userdata_chunk_size = sizeof(tls_data_dummy);

	settings.func_free = mesh_calc_tessellation_for_face_free_fn;

	BLI_task_parallel_range(0,
		totpoly,
		&data,
		poly_normals ? mesh_calc_tessellation_for_face_with_normal_fn :
		mesh_calc_tessellation_for_face_fn,
		&settings);
}

void BKE_mesh_recalc_looptri(const MLoop* mloop,
	const MPoly* mpoly, const float(*vert_positions)[3],
	int totloop, int totpoly, MLoopTri* mlooptri)
{
	if (totloop < MESH_FACE_TESSELLATE_THREADED_LIMIT) 
	{
		mesh_recalc_looptri__single_threaded(mloop, mpoly, vert_positions, totloop, totpoly, mlooptri, nullptr);
	}
	else 
	{
		mesh_recalc_looptri__multi_threaded(mloop, mpoly, vert_positions, totpoly, mlooptri, nullptr);
	}
}

void BKE_mesh_runtime_looptri_recalc(Mesh* mesh)
{
	mesh_ensure_looptri_data(mesh);
	BLI_assert(mesh->totpoly == 0 || mesh->runtime->looptris.array_wip != nullptr);

	BKE_mesh_recalc_looptri(mesh->mloop, mesh->mpoly, (float(*)[3])mesh->mvert, mesh->totloop, mesh->totpoly, mesh->runtime->looptris.array_wip);

	BLI_assert(mesh->runtime->looptris.array == nullptr);
	atomic_cas_ptr((void**)&mesh->runtime->looptris.array, mesh->runtime->looptris.array, mesh->runtime->looptris.array_wip);
	mesh->runtime->looptris.array_wip = nullptr;
}

const MLoopTri* BKE_mesh_runtime_looptri_ensure(const Mesh* mesh)
{
	MLoopTri* looptri = mesh->runtime->looptris.array;

	if (looptri != nullptr) 
	{
		BLI_assert(BKE_mesh_runtime_looptri_len(mesh) == mesh->runtime->looptris.len);
	}
	else 
	{
		BKE_mesh_runtime_looptri_recalc(const_cast<Mesh*>(mesh));
		looptri = mesh->runtime->looptris.array;
	}

	return looptri;
}

static void mesh_ensure_cdlayers_primary(Mesh* mesh, bool do_tessface)
{
	auto a = CustomData_get_layer_named(&mesh->vdata, CD_PROP_FLOAT3, "position");
	if (!a)
	{
		CustomData_add_layer_named(&mesh->vdata, CD_PROP_FLOAT3, CD_CONSTRUCT, nullptr, mesh->totvert, "position");
	}

	if (!CustomData_get_layer(&mesh->edata, CD_MEDGE)) 
	{
		CustomData_add_layer(&mesh->edata, CD_MEDGE, CD_SET_DEFAULT, nullptr, mesh->totedge);
	}

	if (!CustomData_get_layer(&mesh->ldata, CD_MLOOP)) 
	{
		CustomData_add_layer(&mesh->ldata, CD_MLOOP, CD_SET_DEFAULT, nullptr, mesh->totloop);
	}

	if (!CustomData_get_layer(&mesh->pdata, CD_MPOLY)) 
	{
		CustomData_add_layer(&mesh->pdata, CD_MPOLY, CD_SET_DEFAULT, nullptr, mesh->totpoly);
	}

	if (do_tessface && !CustomData_get_layer(&mesh->fdata, CD_MFACE)) 
	{
		CustomData_add_layer(&mesh->fdata, CD_MFACE, CD_SET_DEFAULT, nullptr, mesh->totface);
	}
}

void cloth_from_mesh(ClothModifierData* clmd, Mesh* mesh)
{
	// Обнуляем временные данные
	CustomData_reset(&mesh->vdata);
	CustomData_reset(&mesh->edata);
	CustomData_reset(&mesh->ldata);
	CustomData_reset(&mesh->pdata);
	CustomData_reset(&mesh->fdata);

	// получаем кол-во индексов вершин, представляющих треугольники
	const uint looptri_num = poly_to_tri_count(mesh->totpoly, mesh->totloop);

	// Создаём временные данные, если их нет
	if (!mesh->runtime)
	{
		mesh->runtime = new MeshRuntime();

		// Делаем перерасчёт 
		MLoopTri* looptris = (MLoopTri*)MEM_malloc_arrayN(looptri_num, sizeof(*looptris), "runtime looptris");

		BKE_mesh_recalc_looptri(mesh->mloop, mesh->mpoly, (const float(*)[3])mesh->mvert, mesh->totloop, mesh->totpoly, looptris);

		mesh->runtime->looptris.array = looptris;
		mesh->runtime->looptris.len = looptri_num;
	}

	mesh_ensure_cdlayers_primary(mesh, true);

	BKE_mesh_debug_info(mesh);

	// Вычисление и кэширование MLoopTri для меша
	BKE_mesh_runtime_looptri_ensure(mesh);
	
	const MLoop* mloop = mesh->mloop;
	// Получение указателя на MLoopTri
	const MLoopTri* looptri = BKE_mesh_runtime_looptri_ensure(mesh);
	BLI_assert(looptri);

	/* Allocate our vertices. */
	clmd->clothObject->mvert_num = mesh->totvert;
	clmd->clothObject->verts = (ClothVertex*)MEM_callocN(sizeof(ClothVertex) * clmd->clothObject->mvert_num, "clothVertex");
	if (clmd->clothObject->verts == NULL) 
	{
		cloth_free_modifier(clmd);
		//BKE_modifier_set_error(ob, &(clmd->modifier), "Out of memory on allocating clmd->clothObject->verts");
		printf("Out of memory on allocating clmd->clothObject->verts");
		printf("cloth_free_modifier clmd->clothObject->verts\n");
		return;
	}

	/* save face information */
	if (clmd->hairdata == NULL) 
	{
		clmd->clothObject->primitive_num = looptri_num;
	}
	else 
	{
		clmd->clothObject->primitive_num = mesh->totedge;
	}

	clmd->clothObject->tri = (MVertTri*)MEM_mallocN(sizeof(MVertTri) * looptri_num, "clothLoopTris");
	if (clmd->clothObject->tri == NULL) 
	{
		cloth_free_modifier(clmd);
		//BKE_modifier_set_error( ob, &(clmd->modifier), "Out of memory on allocating clmd->clothObject->looptri");
		printf("Out of memory on allocating clmd->clothObject->looptri");
		printf("cloth_free_modifier clmd->clothObject->looptri\n");
		return;
	}
	BKE_mesh_runtime_verttri_from_looptri(clmd->clothObject->tri, mloop, looptri, looptri_num);

	clmd->clothObject->edges = mesh->medge;

	/* Free the springs since they can't be correct if the vertices changed. */
	if (clmd->clothObject->springs != NULL) 
	{
		MEM_freeN(clmd->clothObject->springs);
	}
}