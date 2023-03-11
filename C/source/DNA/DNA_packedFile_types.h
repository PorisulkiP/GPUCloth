/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2001-2002 NaN Holding BV. All rights reserved. */

/** \file
 * \ingroup DNA
 */

#pragma once


typedef struct PackedFile {
  int size;
  int seek;
  void *data;
} PackedFile;
