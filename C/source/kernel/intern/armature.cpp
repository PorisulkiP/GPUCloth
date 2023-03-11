#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "MEM_guardedalloc.cuh"

#include "BLI_alloca.h"
#include "BLI_ghash.h"
#include "listbase.cuh"
#include "math.cuh"
#include "BLI_string.h"
#include "utildefines.h"
#include "BLT_translation.h"

#include "defaults.cuh"

#include "DNA_armature_types.h"
#include "DNA_constraint_types.h"
#include "listBase.cuh"
#include "object_types.cuh"
#include "scene_types.cuh"

#include "BKE_action.h"
#include "BKE_anim_data.h"
#include "BKE_anim_visualization.h"
#include "BKE_armature.h"
#include "BKE_constraint.h"
#include "BKE_curve.h"
#include "BKE_idprop.h"
#include "BKE_idtype.h"
#include "BKE_lib_id.h"
#include "BKE_lib_query.h"
#include "BKE_main.h"
#include "BKE_object.h"
#include "BKE_scene.h"

#include "DEG_depsgraph_build.h"
#include "DEG_depsgraph_query.h"

static void armature_bone_from_name_insert_recursive(GHash *bone_hash, ListBase *lb)
{
  LISTBASE_FOREACH (Bone *, bone, lb) {
    BLI_ghash_insert(bone_hash, bone->name, bone);
    armature_bone_from_name_insert_recursive(bone_hash, &bone->childbase);
  }
}

/**
 * Create a (name -> bone) map.
 *
 * \note typically #bPose.chanhash us used via #BKE_pose_channel_find_name
 * this is for the cases we can't use pose channels.
 */
static GHash *armature_bone_from_name_map(bArmature *arm)
{
  const int bones_count = BKE_armature_bonelist_count(&arm->bonebase);
  GHash *bone_hash = BLI_ghash_str_new_ex(__func__, bones_count);
  armature_bone_from_name_insert_recursive(bone_hash, &arm->bonebase);
  return bone_hash;
}

void BKE_armature_bone_hash_make(bArmature *arm)
{
  if (!arm->bonehash) {
    arm->bonehash = armature_bone_from_name_map(arm);
  }
}

void BKE_armature_bone_hash_free(bArmature *arm)
{
  if (arm->bonehash) {
    BLI_ghash_free(arm->bonehash, NULL, NULL);
    arm->bonehash = NULL;
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Armature Bone Flags
 * \{ */

bool BKE_armature_bone_flag_test_recursive(const Bone *bone, int flag)
{
  if (bone->flag & flag) {
    return true;
  }
  if (bone->parent) {
    return BKE_armature_bone_flag_test_recursive(bone->parent, flag);
  }
  return false;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Armature Layer Refresh Used
 * \{ */

static void armature_refresh_layer_used_recursive(bArmature *arm, ListBase *bones)
{
  LISTBASE_FOREACH (Bone *, bone, bones) {
    arm->layer_used |= bone->layer;
    armature_refresh_layer_used_recursive(arm, &bone->childbase);
  }
}

void BKE_armature_refresh_layer_used(struct Depsgraph *depsgraph, struct bArmature *arm)
{
  if (arm->edbo != NULL) {
    /* Don't perform this update when the armature is in edit mode. In that case it should be
     * handled by ED_armature_edit_refresh_layer_used(). */
    return;
  }

  arm->layer_used = 0;
  armature_refresh_layer_used_recursive(arm, &arm->bonebase);

  if (depsgraph == NULL || DEG_is_active(depsgraph)) {
    bArmature *arm_orig = (bArmature *)DEG_get_original_id(&arm->id);
    arm_orig->layer_used = arm->layer_used;
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Armature Layer Refresh Used
 * \{ */

/* Finds the best possible extension to the name on a particular axis. (For renaming, check for
 * unique names afterwards) strip_number: removes number extensions  (TODO: not used)
 * axis: the axis to name on
 * head/tail: the head/tail co-ordinate of the bone on the specified axis */
bool bone_autoside_name(
    char name[MAXBONENAME], int UNUSED(strip_number), short axis, float head, float tail)
{
  unsigned int len;
  char basename[MAXBONENAME] = "";
  char extension[5] = "";

  len = strlen(name);
  if (len == 0) {
    return false;
  }
  BLI_strncpy(basename, name, sizeof(basename));

  /* Figure out extension to append:
   * - The extension to append is based upon the axis that we are working on.
   * - If head happens to be on 0, then we must consider the tail position as well to decide
   *   which side the bone is on
   *   -> If tail is 0, then its bone is considered to be on axis, so no extension should be added
   *   -> Otherwise, extension is added from perspective of object based on which side tail goes to
   * - If head is non-zero, extension is added from perspective of object based on side head is on
   */
  if (axis == 2) {
    /* z-axis - vertical (top/bottom) */
    if (IS_EQF(head, 0.0f)) {
      if (tail < 0) {
        strcpy(extension, "Bot");
      }
      else if (tail > 0) {
        strcpy(extension, "Top");
      }
    }
    else {
      if (head < 0) {
        strcpy(extension, "Bot");
      }
      else {
        strcpy(extension, "Top");
      }
    }
  }
  else if (axis == 1) {
    /* y-axis - depth (front/back) */
    if (IS_EQF(head, 0.0f)) {
      if (tail < 0) {
        strcpy(extension, "Fr");
      }
      else if (tail > 0) {
        strcpy(extension, "Bk");
      }
    }
    else {
      if (head < 0) {
        strcpy(extension, "Fr");
      }
      else {
        strcpy(extension, "Bk");
      }
    }
  }
  else {
    /* x-axis - horizontal (left/right) */
    if (IS_EQF(head, 0.0f)) {
      if (tail < 0) {
        strcpy(extension, "R");
      }
      else if (tail > 0) {
        strcpy(extension, "L");
      }
    }
    else {
      if (head < 0) {
        strcpy(extension, "R");
        /* XXX Shouldn't this be simple else, as for z and y axes? */
      }
      else if (head > 0) {
        strcpy(extension, "L");
      }
    }
  }

  /* Simple name truncation
   * - truncate if there is an extension and it wouldn't be able to fit
   * - otherwise, just append to end
   */
  if (extension[0]) {
    bool changed = true;

    while (changed) { /* remove extensions */
      changed = false;
      if (len > 2 && basename[len - 2] == '.') {
        if (ELEM(basename[len - 1], 'L', 'R')) { /* L R */
          basename[len - 2] = '\0';
          len -= 2;
          changed = true;
        }
      }
      else if (len > 3 && basename[len - 3] == '.') {
        if ((basename[len - 2] == 'F' && basename[len - 1] == 'r') || /* Fr */
            (basename[len - 2] == 'B' && basename[len - 1] == 'k'))   /* Bk */
        {
          basename[len - 3] = '\0';
          len -= 3;
          changed = true;
        }
      }
      else if (len > 4 && basename[len - 4] == '.') {
        if ((basename[len - 3] == 'T' && basename[len - 2] == 'o' &&
             basename[len - 1] == 'p') || /* Top */
            (basename[len - 3] == 'B' && basename[len - 2] == 'o' &&
             basename[len - 1] == 't')) /* Bot */
        {
          basename[len - 4] = '\0';
          len -= 4;
          changed = true;
        }
      }
    }

    if ((MAXBONENAME - len) < strlen(extension) + 1) { /* add 1 for the '.' */
      strncpy(name, basename, len - strlen(extension));
    }

    BLI_snprintf(name, MAXBONENAME, "%s.%s", basename, extension);

    return true;
  }
  return false;
}

/* Computes the bezier handle vectors and rolls coming from custom handles. */
void BKE_pchan_bbone_handles_compute(const BBoneSplineParameters *param,
                                     float h1[3],
                                     float *r_roll1,
                                     float h2[3],
                                     float *r_roll2,
                                     bool ease,
                                     bool offsets)
{
  float mat3[3][3];
  float length = param->length;
  float epsilon = 1e-5 * length;

  if (param->do_scale) {
    length *= param->scale[1];
  }

  *r_roll1 = *r_roll2 = 0.0f;

  if (param->use_prev) {
    copy_v3_v3(h1, param->prev_h);

    if (param->prev_bbone) {
      /* If previous bone is B-bone too, use average handle direction. */
      h1[1] -= length;
    }

    if (normalize_v3(h1) < epsilon) {
      copy_v3_fl3(h1, 0.0f, -1.0f, 0.0f);
    }

    negate_v3(h1);

    if (!param->prev_bbone) {
      /* Find the previous roll to interpolate. */
      copy_m3_m4(mat3, param->prev_mat);
      mat3_vec_to_roll(mat3, h1, r_roll1);
    }
  }
  else {
    h1[0] = 0.0f;
    h1[1] = 1.0;
    h1[2] = 0.0f;
  }

  if (param->use_next) {
    copy_v3_v3(h2, param->next_h);

    /* If next bone is B-bone too, use average handle direction. */
    if (param->next_bbone) {
      /* pass */
    }
    else {
      h2[1] -= length;
    }

    if (normalize_v3(h2) < epsilon) {
      copy_v3_fl3(h2, 0.0f, 1.0f, 0.0f);
    }

    /* Find the next roll to interpolate as well. */
    copy_m3_m4(mat3, param->next_mat);
    mat3_vec_to_roll(mat3, h2, r_roll2);
  }
  else {
    h2[0] = 0.0f;
    h2[1] = 1.0f;
    h2[2] = 0.0f;
  }

  if (ease) {
    const float circle_factor = length * (cubic_tangent_factor_circle_v3(h1, h2) / 0.75f);

    const float hlength1 = param->ease1 * circle_factor;
    const float hlength2 = param->ease2 * circle_factor;

    /* and only now negate h2 */
    mul_v3_fl(h1, hlength1);
    mul_v3_fl(h2, -hlength2);
  }

  /* Add effects from bbone properties over the top
   * - These properties allow users to hand-animate the
   *   bone curve/shape, without having to resort to using
   *   extra bones
   * - The "bone" level offsets are for defining the rest-pose
   *   shape of the bone (e.g. for curved eyebrows for example).
   *   -> In the viewport, it's needed to define what the rest pose
   *      looks like
   *   -> For "rest == 0", we also still need to have it present
   *      so that we can "cancel out" this rest-pose when it comes
   *      time to deform some geometry, it won't cause double transforms.
   * - The "pchan" level offsets are the ones that animators actually
   *   end up animating
   */
  if (offsets) {
    /* Add extra rolls. */
    *r_roll1 += param->roll1;
    *r_roll2 += param->roll2;

    /* Extra curve x / y */
    /* NOTE:
     * Scale correction factors here are to compensate for some random floating-point glitches
     * when scaling up the bone or its parent by a factor of approximately 8.15/6, which results
     * in the bone length getting scaled up too (from 1 to 8), causing the curve to flatten out.
     */
    const float xscale_correction = (param->do_scale) ? param->scale[0] : 1.0f;
    const float yscale_correction = (param->do_scale) ? param->scale[2] : 1.0f;

    h1[0] += param->curve_in_x * xscale_correction;
    h1[2] += param->curve_in_y * yscale_correction;

    h2[0] += param->curve_out_x * xscale_correction;
    h2[2] += param->curve_out_y * yscale_correction;
  }
}

static void make_bbone_spline_matrix(BBoneSplineParameters *param,
                                     const float scalemats[2][4][4],
                                     const float pos[3],
                                     const float axis[3],
                                     float roll,
                                     float scalex,
                                     float scaley,
                                     float result[4][4])
{
  float mat3[3][3];

  vec_roll_to_mat3(axis, roll, mat3);

  copy_m4_m3(result, mat3);
  copy_v3_v3(result[3], pos);

  if (param->do_scale) {
    /* Correct for scaling when this matrix is used in scaled space. */
    mul_m4_series(result, scalemats[0], result, scalemats[1]);
  }

  /* BBone scale... */
  mul_v3_fl(result[0], scalex);
  mul_v3_fl(result[2], scaley);
}

/* Fade from first to second derivative when the handle is very short. */
static void ease_handle_axis(const float deriv1[3], const float deriv2[3], float r_axis[3])
{
  const float gap = 0.1f;

  copy_v3_v3(r_axis, deriv1);

  float len1 = len_squared_v3(deriv1), len2 = len_squared_v3(deriv2);
  float ratio = len1 / len2;

  if (ratio < gap * gap) {
    madd_v3_v3fl(r_axis, deriv2, gap - sqrtf(ratio));
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Bone Space to Space Conversion API
 * \{ */

void get_objectspace_bone_matrix(struct Bone *bone,
                                 float M_accumulatedMatrix[4][4],
                                 int UNUSED(root),
                                 int UNUSED(posed))
{
  copy_m4_m4(M_accumulatedMatrix, bone->arm_mat);
}

/* Convert World-Space Matrix to Pose-Space Matrix */
void BKE_armature_mat_world_to_pose(Object *ob, const float inmat[4][4], float outmat[4][4])
{
  float obmat[4][4];

  /* prevent crashes */
  if (ob == NULL) {
    return;
  }

  /* get inverse of (armature) object's matrix  */
  invert_m4_m4(obmat, ob->obmat);

  /* multiply given matrix by object's-inverse to find pose-space matrix */
  mul_m4_m4m4(outmat, inmat, obmat);
}

/* Convert World-Space Location to Pose-Space Location
 * NOTE: this cannot be used to convert to pose-space location of the supplied
 *       pose-channel into its local space (i.e. 'visual'-keyframing) */
void BKE_armature_loc_world_to_pose(Object *ob, const float inloc[3], float outloc[3])
{
  float xLocMat[4][4];
  float nLocMat[4][4];

  /* build matrix for location */
  unit_m4(xLocMat);
  copy_v3_v3(xLocMat[3], inloc);

  /* get bone-space cursor matrix and extract location */
  BKE_armature_mat_world_to_pose(ob, xLocMat, nLocMat);
  copy_v3_v3(outloc, nLocMat[3]);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Bone Matrix Calculation API
 * \{ */

/* Simple helper, computes the offset bone matrix.
 *     offs_bone = yoffs(b-1) + root(b) + bonemat(b). */
void BKE_bone_offset_matrix_get(const Bone *bone, float offs_bone[4][4])
{
  BLI_assert(bone->parent != NULL);

  /* Bone transform itself. */
  copy_m4_m3(offs_bone, bone->bone_mat);

  /* The bone's root offset (is in the parent's coordinate system). */
  copy_v3_v3(offs_bone[3], bone->head);

  /* Get the length translation of parent (length along y axis). */
  offs_bone[3][1] += bone->parent->length;
}


/* Compute the parent transform using data decoupled from specific data structures.
 *
 * bone_flag: Bone->flag containing settings
 * offs_bone: delta from parent to current arm_mat (or just arm_mat if no parent)
 * parent_arm_mat, parent_pose_mat: arm_mat and pose_mat of parent, or NULL
 * r_bpt: OUTPUT parent transform */
void BKE_bone_parent_transform_calc_from_matrices(int bone_flag,
                                                  int inherit_scale_mode,
                                                  const float offs_bone[4][4],
                                                  const float parent_arm_mat[4][4],
                                                  const float parent_pose_mat[4][4],
                                                  BoneParentTransform *r_bpt)
{
  copy_v3_fl(r_bpt->post_scale, 1.0f);

  if (parent_pose_mat) {
    const bool use_rotation = (bone_flag & BONE_HINGE) == 0;
    const bool full_transform = use_rotation && inherit_scale_mode == BONE_INHERIT_SCALE_FULL;

    /* Compose the rotscale matrix for this bone. */
    if (full_transform) {
      /* Parent pose rotation and scale. */
      mul_m4_m4m4(r_bpt->rotscale_mat, parent_pose_mat, offs_bone);
    }
    else {
      float tmat[4][4], tscale[3];

      /* If using parent pose rotation: */
      if (use_rotation) {
        copy_m4_m4(tmat, parent_pose_mat);

        /* Normalize the matrix when needed. */
        switch (inherit_scale_mode) {
          case BONE_INHERIT_SCALE_FULL:
          case BONE_INHERIT_SCALE_FIX_SHEAR:
            /* Keep scale and shear. */
            break;

          case BONE_INHERIT_SCALE_NONE:
          case BONE_INHERIT_SCALE_AVERAGE:
            /* Remove scale and shear from parent. */
            orthogonalize_m4_stable(tmat, 1, true);
            break;

          case BONE_INHERIT_SCALE_ALIGNED:
            /* Remove shear and extract scale. */
            orthogonalize_m4_stable(tmat, 1, false);
            normalize_m4_ex(tmat, r_bpt->post_scale);
            break;

          case BONE_INHERIT_SCALE_NONE_LEGACY:
            /* Remove only scale - bad legacy way. */
            normalize_m4(tmat);
            break;

          default:
            BLI_assert(false);
        }
      }
      /* If removing parent pose rotation: */
      else {
        copy_m4_m4(tmat, parent_arm_mat);

        /* Copy the parent scale when needed. */
        switch (inherit_scale_mode) {
          case BONE_INHERIT_SCALE_FULL:
            /* Ignore effects of shear. */
            mat4_to_size(tscale, parent_pose_mat);
            rescale_m4(tmat, tscale);
            break;

          case BONE_INHERIT_SCALE_FIX_SHEAR:
            /* Take the effects of parent shear into account to get exact volume. */
            mat4_to_size_fix_shear(tscale, parent_pose_mat);
            rescale_m4(tmat, tscale);
            break;

          case BONE_INHERIT_SCALE_ALIGNED:
            mat4_to_size_fix_shear(r_bpt->post_scale, parent_pose_mat);
            break;

          case BONE_INHERIT_SCALE_NONE:
          case BONE_INHERIT_SCALE_AVERAGE:
          case BONE_INHERIT_SCALE_NONE_LEGACY:
            /* Keep unscaled. */
            break;

          default:
            BLI_assert(false);
        }
      }

      /* Apply the average parent scale when needed. */
      if (inherit_scale_mode == BONE_INHERIT_SCALE_AVERAGE) {
        mul_mat3_m4_fl(tmat, cbrtf(fabsf(mat4_to_volume_scale(parent_pose_mat))));
      }

      mul_m4_m4m4(r_bpt->rotscale_mat, tmat, offs_bone);

      /* Remove remaining shear when needed, preserving volume. */
      if (inherit_scale_mode == BONE_INHERIT_SCALE_FIX_SHEAR) {
        orthogonalize_m4_stable(r_bpt->rotscale_mat, 1, false);
      }
    }

    /* Compose the loc matrix for this bone. */
    /* NOTE: That version does not modify bone's loc when HINGE/NO_SCALE options are set. */

    /* In this case, use the object's space *orientation*. */
    if (bone_flag & BONE_NO_LOCAL_LOCATION) {
      /* XXX I'm sure that code can be simplified! */
      float bone_loc[4][4], bone_rotscale[3][3], tmat4[4][4], tmat3[3][3];
      unit_m4(bone_loc);
      unit_m4(r_bpt->loc_mat);
      unit_m4(tmat4);

      mul_v3_m4v3(bone_loc[3], parent_pose_mat, offs_bone[3]);

      unit_m3(bone_rotscale);
      copy_m3_m4(tmat3, parent_pose_mat);
      mul_m3_m3m3(bone_rotscale, tmat3, bone_rotscale);

      copy_m4_m3(tmat4, bone_rotscale);
      mul_m4_m4m4(r_bpt->loc_mat, bone_loc, tmat4);
    }
    /* Those flags do not affect position, use plain parent transform space! */
    else if (!full_transform) {
      mul_m4_m4m4(r_bpt->loc_mat, parent_pose_mat, offs_bone);
    }
    /* Else (i.e. default, usual case),
     * just use the same matrix for rotation/scaling, and location. */
    else {
      copy_m4_m4(r_bpt->loc_mat, r_bpt->rotscale_mat);
    }
  }
  /* Root bones. */
  else {
    /* Rotation/scaling. */
    copy_m4_m4(r_bpt->rotscale_mat, offs_bone);
    /* Translation. */
    if (bone_flag & BONE_NO_LOCAL_LOCATION) {
      /* Translation of arm_mat, without the rotation. */
      unit_m4(r_bpt->loc_mat);
      copy_v3_v3(r_bpt->loc_mat[3], offs_bone[3]);
    }
    else {
      copy_m4_m4(r_bpt->loc_mat, r_bpt->rotscale_mat);
    }
  }
}

void BKE_bone_parent_transform_clear(struct BoneParentTransform *bpt)
{
  unit_m4(bpt->rotscale_mat);
  unit_m4(bpt->loc_mat);
  copy_v3_fl(bpt->post_scale, 1.0f);
}

void BKE_bone_parent_transform_invert(struct BoneParentTransform *bpt)
{
  invert_m4(bpt->rotscale_mat);
  invert_m4(bpt->loc_mat);
  invert_v3(bpt->post_scale);
}

void BKE_bone_parent_transform_combine(const struct BoneParentTransform *in1,
                                       const struct BoneParentTransform *in2,
                                       struct BoneParentTransform *result)
{
  mul_m4_m4m4(result->rotscale_mat, in1->rotscale_mat, in2->rotscale_mat);
  mul_m4_m4m4(result->loc_mat, in1->loc_mat, in2->loc_mat);
  mul_v3_v3v3(result->post_scale, in1->post_scale, in2->post_scale);
}

void BKE_bone_parent_transform_apply(const struct BoneParentTransform *bpt,
                                     const float inmat[4][4],
                                     float outmat[4][4])
{
  /* in case inmat == outmat */
  float tmploc[3];
  copy_v3_v3(tmploc, inmat[3]);

  mul_m4_m4m4(outmat, bpt->rotscale_mat, inmat);
  mul_v3_m4v3(outmat[3], bpt->loc_mat, tmploc);
  rescale_m4(outmat, bpt->post_scale);
}

/* Convert Pose-Space Matrix to Bone-Space Matrix.
 * NOTE: this cannot be used to convert to pose-space transforms of the supplied
 *       pose-channel into its local space (i.e. 'visual'-keyframing) */
void BKE_armature_mat_pose_to_bone(bPoseChannel *pchan,
                                   const float inmat[4][4],
                                   float outmat[4][4])
{
  BoneParentTransform bpt;

  BKE_bone_parent_transform_calc_from_pchan(pchan, &bpt);
  BKE_bone_parent_transform_invert(&bpt);
  BKE_bone_parent_transform_apply(&bpt, inmat, outmat);
}

/* Convert Bone-Space Matrix to Pose-Space Matrix. */
void BKE_armature_mat_bone_to_pose(bPoseChannel *pchan,
                                   const float inmat[4][4],
                                   float outmat[4][4])
{
  BoneParentTransform bpt;

  BKE_bone_parent_transform_calc_from_pchan(pchan, &bpt);
  BKE_bone_parent_transform_apply(&bpt, inmat, outmat);
}

/* Convert Pose-Space Location to Bone-Space Location
 * NOTE: this cannot be used to convert to pose-space location of the supplied
 *       pose-channel into its local space (i.e. 'visual'-keyframing) */
void BKE_armature_loc_pose_to_bone(bPoseChannel *pchan, const float inloc[3], float outloc[3])
{
  float xLocMat[4][4];
  float nLocMat[4][4];

  /* build matrix for location */
  unit_m4(xLocMat);
  copy_v3_v3(xLocMat[3], inloc);

  /* get bone-space cursor matrix and extract location */
  BKE_armature_mat_pose_to_bone(pchan, xLocMat, nLocMat);
  copy_v3_v3(outloc, nLocMat[3]);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Bone Matrix Read/Write API
 *
 * High level functions for transforming bones and reading the transform values.
 * \{ */

/* Computes vector and roll based on a rotation.
 * "mat" must contain only a rotation, and no scaling. */
void mat3_to_vec_roll(const float mat[3][3], float r_vec[3], float *r_roll)
{
  if (r_vec) {
    copy_v3_v3(r_vec, mat[1]);
  }

  if (r_roll) {
    mat3_vec_to_roll(mat, mat[1], r_roll);
  }
}

/* Computes roll around the vector that best approximates the matrix.
 * If vec is the Y vector from purely rotational mat, result should be exact. */
void mat3_vec_to_roll(const float mat[3][3], const float vec[3], float *r_roll)
{
  float vecmat[3][3], vecmatinv[3][3], rollmat[3][3], q[4];

  /* Compute the orientation relative to the vector with zero roll. */
  vec_roll_to_mat3(vec, 0.0f, vecmat);
  invert_m3_m3(vecmatinv, vecmat);
  mul_m3_m3m3(rollmat, vecmatinv, mat);

  /* Extract the twist angle as the roll value. */
  mat3_to_quat(q, rollmat);

  *r_roll = quat_split_swing_and_twist(q, 1, NULL, NULL);
}

/* Calculates the rest matrix of a bone based on its vector and a roll around that vector. */
/**
 * Given `v = (v.x, v.y, v.z)` our (normalized) bone vector, we want the rotation matrix M
 * from the Y axis (so that `M * (0, 1, 0) = v`).
 * - The rotation axis a lays on XZ plane, and it is orthonormal to v,
 *   hence to the projection of v onto XZ plane.
 * - `a = (v.z, 0, -v.x)`
 *
 * We know a is eigenvector of M (so M * a = a).
 * Finally, we have w, such that M * w = (0, 1, 0)
 * (i.e. the vector that will be aligned with Y axis once transformed).
 * We know w is symmetric to v by the Y axis.
 * - `w = (-v.x, v.y, -v.z)`
 *
 * Solving this, we get (x, y and z being the components of v):
 * <pre>
 *     ┌ (x^2 * y + z^2) / (x^2 + z^2),   x,   x * z * (y - 1) / (x^2 + z^2) ┐
 * M = │  x * (y^2 - 1)  / (x^2 + z^2),   y,    z * (y^2 - 1)  / (x^2 + z^2) │
 *     └ x * z * (y - 1) / (x^2 + z^2),   z,   (x^2 + z^2 * y) / (x^2 + z^2) ┘
 * </pre>
 *
 * This is stable as long as v (the bone) is not too much aligned with +/-Y
 * (i.e. x and z components are not too close to 0).
 *
 * Since v is normalized, we have `x^2 + y^2 + z^2 = 1`,
 * hence `x^2 + z^2 = 1 - y^2 = (1 - y)(1 + y)`.
 *
 * This allows to simplifies M like this:
 * <pre>
 *     ┌ 1 - x^2 / (1 + y),   x,     -x * z / (1 + y) ┐
 * M = │                -x,   y,                   -z │
 *     └  -x * z / (1 + y),   z,    1 - z^2 / (1 + y) ┘
 * </pre>
 *
 * Written this way, we see the case v = +Y is no more a singularity.
 * The only one
 * remaining is the bone being aligned with -Y.
 *
 * Let's handle
 * the asymptotic behavior when bone vector is reaching the limit of y = -1.
 * Each of the four corner elements can vary from -1 to 1,
 * depending on the axis a chosen for doing the rotation.
 * And the "rotation" here is in fact established by mirroring XZ plane by that given axis,
 * then inversing the Y-axis.
 * For sufficiently small x and z, and with y approaching -1,
 * all elements but the four corner ones of M will degenerate.
 * So let's now focus on these corner elements.
 *
 * We rewrite M so that it only contains its four corner elements,
 * and combine the `1 / (1 + y)` factor:
 * <pre>
 *                    ┌ 1 + y - x^2,        -x * z ┐
 * M* = 1 / (1 + y) * │                            │
 *                    └      -x * z,   1 + y - z^2 ┘
 * </pre>
 *
 * When y is close to -1, computing 1 / (1 + y) will cause severe numerical instability,
 * so we ignore it and normalize M instead.
 * We know `y^2 = 1 - (x^2 + z^2)`, and `y < 0`, hence `y = -sqrt(1 - (x^2 + z^2))`.
 *
 * Since x and z are both close to 0, we apply the binomial expansion to the first order:
 * `y = -sqrt(1 - (x^2 + z^2)) = -1 + (x^2 + z^2) / 2`. Which gives:
 * <pre>
 *                        ┌  z^2 - x^2,  -2 * x * z ┐
 * M* = 1 / (x^2 + z^2) * │                         │
 *                        └ -2 * x * z,   x^2 - z^2 ┘
 * </pre>
 */
void vec_roll_to_mat3_normalized(const float nor[3], const float roll, float r_mat[3][3])
{
  const float THETA_SAFE = 1.0e-5f;     /* theta above this value are always safe to use. */
  const float THETA_CRITICAL = 1.0e-9f; /* above this is safe under certain conditions. */

  const float x = nor[0];
  const float y = nor[1];
  const float z = nor[2];

  const float theta = 1.0f + y;
  const float theta_alt = x * x + z * z;
  float rMatrix[3][3], bMatrix[3][3];

  BLI_ASSERT_UNIT_V3(nor);

  /* When theta is close to zero (nor is aligned close to negative Y Axis),
   * we have to check we do have non-null X/Z components as well.
   * Also, due to float precision errors, nor can be (0.0, -0.99999994, 0.0) which results
   * in theta being close to zero. This will cause problems when theta is used as divisor.
   */
  if (theta > THETA_SAFE || ((x || z) && theta > THETA_CRITICAL)) {
    /* nor is *not* aligned to negative Y-axis (0,-1,0).
     * We got these values for free... so be happy with it... ;)
     */

    bMatrix[0][1] = -x;
    bMatrix[1][0] = x;
    bMatrix[1][1] = y;
    bMatrix[1][2] = z;
    bMatrix[2][1] = -z;

    if (theta > THETA_SAFE) {
      /* nor differs significantly from negative Y axis (0,-1,0): apply the general case. */
      bMatrix[0][0] = 1 - x * x / theta;
      bMatrix[2][2] = 1 - z * z / theta;
      bMatrix[2][0] = bMatrix[0][2] = -x * z / theta;
    }
    else {
      /* nor is close to negative Y axis (0,-1,0): apply the special case. */
      bMatrix[0][0] = (x + z) * (x - z) / -theta_alt;
      bMatrix[2][2] = -bMatrix[0][0];
      bMatrix[2][0] = bMatrix[0][2] = 2.0f * x * z / theta_alt;
    }
  }
  else {
    /* nor is very close to negative Y axis (0,-1,0): use simple symmetry by Z axis. */
    unit_m3(bMatrix);
    bMatrix[0][0] = bMatrix[1][1] = -1.0;
  }

  /* Make Roll matrix */
  axis_angle_normalized_to_mat3(rMatrix, nor, roll);

  /* Combine and output result */
  mul_m3_m3m3(r_mat, rMatrix, bMatrix);
}

void vec_roll_to_mat3(const float vec[3], const float roll, float r_mat[3][3])
{
  float nor[3];

  normalize_v3_v3(nor, vec);
  vec_roll_to_mat3_normalized(nor, roll, r_mat);
}