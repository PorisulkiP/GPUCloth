/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup modifiers
 */

#pragma once

#include "modifier.h"

/* ****************** Type structures for all modifiers ****************** */

inline ModifierTypeInfo modifierType_None;
inline ModifierTypeInfo modifierType_Subsurf;
inline ModifierTypeInfo modifierType_Lattice;
inline ModifierTypeInfo modifierType_Curve;
inline ModifierTypeInfo modifierType_Build;
inline ModifierTypeInfo modifierType_Mirror;
inline ModifierTypeInfo modifierType_Decimate;
inline ModifierTypeInfo modifierType_Wave;
inline ModifierTypeInfo modifierType_Armature;
inline ModifierTypeInfo modifierType_Hook;
inline ModifierTypeInfo modifierType_Softbody;
inline ModifierTypeInfo modifierType_Boolean;
inline ModifierTypeInfo modifierType_Array;
inline ModifierTypeInfo modifierType_EdgeSplit;
inline ModifierTypeInfo modifierType_Displace;
inline ModifierTypeInfo modifierType_UVProject;
inline ModifierTypeInfo modifierType_Smooth;
inline ModifierTypeInfo modifierType_Cast;
inline ModifierTypeInfo modifierType_MeshDeform;
inline ModifierTypeInfo modifierType_ParticleSystem;
inline ModifierTypeInfo modifierType_ParticleInstance;
inline ModifierTypeInfo modifierType_Explode;
extern ModifierTypeInfo modifierType_Cloth;
inline ModifierTypeInfo modifierType_Collision;
inline ModifierTypeInfo modifierType_Bevel;
inline ModifierTypeInfo modifierType_Shrinkwrap;
inline ModifierTypeInfo modifierType_Fluidsim;
inline ModifierTypeInfo modifierType_Mask;
inline ModifierTypeInfo modifierType_SimpleDeform;
inline ModifierTypeInfo modifierType_Multires;
inline ModifierTypeInfo modifierType_Surface;
inline ModifierTypeInfo modifierType_Fluid;
inline ModifierTypeInfo modifierType_ShapeKey;
inline ModifierTypeInfo modifierType_Solidify;
inline ModifierTypeInfo modifierType_Screw;
inline ModifierTypeInfo modifierType_Ocean;
inline ModifierTypeInfo modifierType_Warp;
inline ModifierTypeInfo modifierType_NavMesh;
inline ModifierTypeInfo modifierType_WeightVGEdit;
inline ModifierTypeInfo modifierType_WeightVGMix;
inline ModifierTypeInfo modifierType_WeightVGProximity;
inline ModifierTypeInfo modifierType_DynamicPaint;
inline ModifierTypeInfo modifierType_Remesh;
inline ModifierTypeInfo modifierType_Skin;
inline ModifierTypeInfo modifierType_LaplacianSmooth;
inline ModifierTypeInfo modifierType_Triangulate;
inline ModifierTypeInfo modifierType_UVWarp;
inline ModifierTypeInfo modifierType_MeshCache;
inline ModifierTypeInfo modifierType_LaplacianDeform;
inline ModifierTypeInfo modifierType_Wireframe;
inline ModifierTypeInfo modifierType_Weld;
inline ModifierTypeInfo modifierType_DataTransfer;
inline ModifierTypeInfo modifierType_NormalEdit;
inline ModifierTypeInfo modifierType_CorrectiveSmooth;
inline ModifierTypeInfo modifierType_MeshSequenceCache;
inline ModifierTypeInfo modifierType_SurfaceDeform;
inline ModifierTypeInfo modifierType_WeightedNormal;
inline ModifierTypeInfo modifierType_Nodes;
inline ModifierTypeInfo modifierType_MeshToVolume;
inline ModifierTypeInfo modifierType_VolumeDisplace;
inline ModifierTypeInfo modifierType_VolumeToMesh;

/* MOD_util.c */

/**
 * Only called by `BKE_modifier.h/modifier.c`
 */
inline void modifier_type_init(ModifierTypeInfo *types[]);