/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup DNA
 */

#pragma once

#include "ID.h"
#include "customdata_types.cuh"


typedef struct Simulation {
  
  ID id;
  struct AnimData *adt; /* animation data (must be immediately after id) */

  /* This nodetree is embedded into the data block. */
  struct bNodeTree *nodetree;

  uint32_t flag;
  char _pad[4];
} Simulation;

/** #Simulation.flag */
enum {
  SIM_DS_EXPAND = (1 << 0),
};
