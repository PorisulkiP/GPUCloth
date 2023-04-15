/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup modifiers
 */

#pragma once

#include "modifier.h"

/* ****************** Type structures for all modifiers ****************** */

ModifierTypeInfo modifierType_None;
ModifierTypeInfo modifierType_Subsurf;
ModifierTypeInfo modifierType_Lattice;
ModifierTypeInfo modifierType_Curve;
ModifierTypeInfo modifierType_Build;
ModifierTypeInfo modifierType_Mirror;
ModifierTypeInfo modifierType_Decimate;
ModifierTypeInfo modifierType_Wave;
ModifierTypeInfo modifierType_Armature;
ModifierTypeInfo modifierType_Hook;
ModifierTypeInfo modifierType_Softbody;
ModifierTypeInfo modifierType_Boolean;
ModifierTypeInfo modifierType_Array;
ModifierTypeInfo modifierType_EdgeSplit;
ModifierTypeInfo modifierType_Displace;
ModifierTypeInfo modifierType_UVProject;
ModifierTypeInfo modifierType_Smooth;
ModifierTypeInfo modifierType_Cast;
ModifierTypeInfo modifierType_MeshDeform;
ModifierTypeInfo modifierType_ParticleSystem;
ModifierTypeInfo modifierType_ParticleInstance;
ModifierTypeInfo modifierType_Explode;
extern ModifierTypeInfo modifierType_Cloth;
ModifierTypeInfo modifierType_Collision;
ModifierTypeInfo modifierType_Bevel;
ModifierTypeInfo modifierType_Shrinkwrap;
ModifierTypeInfo modifierType_Fluidsim;
ModifierTypeInfo modifierType_Mask;
ModifierTypeInfo modifierType_SimpleDeform;
ModifierTypeInfo modifierType_Multires;
ModifierTypeInfo modifierType_Surface;
ModifierTypeInfo modifierType_Fluid;
ModifierTypeInfo modifierType_ShapeKey;
ModifierTypeInfo modifierType_Solidify;
ModifierTypeInfo modifierType_Screw;
ModifierTypeInfo modifierType_Ocean;
ModifierTypeInfo modifierType_Warp;
ModifierTypeInfo modifierType_NavMesh;
ModifierTypeInfo modifierType_WeightVGEdit;
ModifierTypeInfo modifierType_WeightVGMix;
ModifierTypeInfo modifierType_WeightVGProximity;
ModifierTypeInfo modifierType_DynamicPaint;
ModifierTypeInfo modifierType_Remesh;
ModifierTypeInfo modifierType_Skin;
ModifierTypeInfo modifierType_LaplacianSmooth;
ModifierTypeInfo modifierType_Triangulate;
ModifierTypeInfo modifierType_UVWarp;
ModifierTypeInfo modifierType_MeshCache;
ModifierTypeInfo modifierType_LaplacianDeform;
ModifierTypeInfo modifierType_Wireframe;
ModifierTypeInfo modifierType_Weld;
ModifierTypeInfo modifierType_DataTransfer;
ModifierTypeInfo modifierType_NormalEdit;
ModifierTypeInfo modifierType_CorrectiveSmooth;
ModifierTypeInfo modifierType_MeshSequenceCache;
ModifierTypeInfo modifierType_SurfaceDeform;
ModifierTypeInfo modifierType_WeightedNormal;
ModifierTypeInfo modifierType_Nodes;
ModifierTypeInfo modifierType_MeshToVolume;
ModifierTypeInfo modifierType_VolumeDisplace;
ModifierTypeInfo modifierType_VolumeToMesh;

/* MOD_util.c */

/**
 * Only called by `BKE_modifier.h/modifier.c`
 */
void modifier_type_init(ModifierTypeInfo *types[]);