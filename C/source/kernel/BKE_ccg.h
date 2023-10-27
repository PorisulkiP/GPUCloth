#pragma once

/** \file
 * \ingroup bke
 */

/* defines  */
#include "compiler_compat.cuh"

/* declares fprintf() and abort(), needed for BLI_assert */
#include <stdio.h>
#include <stdlib.h>


struct CCGSubSurf;

/* Each CCGElem is CCGSubSurf's representation of a subdivided
 * vertex. All CCGElems in a particular CCGSubSurf have the same
 * layout, but the layout can vary from one CCGSubSurf to another. For
 * this reason, CCGElem is presented as an opaque pointer, and
 * elements should always be accompanied by a CCGKey, which provides
 * the necessary offsets to access components of a CCGElem.
 */
typedef struct CCGElem CCGElem;

typedef struct CCGKey {
  int level;

  /* number of bytes in each element (one float per layer, plus
   * three floats for normals if enabled) */
  int elem_size;

  /* number of elements along each side of grid */
  int grid_size;
  /* number of elements in the grid (grid size squared) */
  int grid_area;
  /* number of bytes in each grid (grid_area * elem_size) */
  int grid_bytes;

  /* currently always the last three floats, unless normals are
   * disabled */
  int normal_offset;

  /* offset in bytes of mask value; only valid if 'has_mask' is
   * true */
  int mask_offset;

  int has_normals;
  int has_mask;
} CCGKey;

/* initialize 'key' at the specified level */
void CCG_key(CCGKey *key, const struct CCGSubSurf *ss, int level);
void CCG_key_top_level(CCGKey *key, const struct CCGSubSurf *ss);

/* get a pointer to the coordinate, normal, or mask components */
 float *CCG_elem_co(CCGElem *elem);
 float *CCG_elem_no(const CCGKey *key, CCGElem *elem);
 float *CCG_elem_mask(const CCGKey *key, CCGElem *elem);

/* get the element at 'offset' in an array */
 CCGElem *CCG_elem_offset(const CCGKey *key, CCGElem *elem, int offset);

/* get the element at coordinate (x,y) in a face-grid array */
 CCGElem *CCG_grid_elem(const CCGKey *key, CCGElem *elem, int x, int y);

/* combinations of above functions */
 float *CCG_grid_elem_co(const CCGKey *key, CCGElem *elem, int x, int y);
 float *CCG_grid_elem_no(const CCGKey *key, CCGElem *elem, int x, int y);
 float *CCG_grid_elem_mask(const CCGKey *key, CCGElem *elem, int x, int y);
 float *CCG_elem_offset_co(const CCGKey *key, CCGElem *elem, int offset);
 float *CCG_elem_offset_no(const CCGKey *key, CCGElem *elem, int offset);
 float *CCG_elem_offset_mask(const CCGKey *key, CCGElem *elem, int offset);

/* for iteration, get a pointer to the next element in an array */
 CCGElem *CCG_elem_next(const CCGKey *key, CCGElem *elem);

/* inline definitions follow */

 float *CCG_elem_co(CCGElem *elem)
{
  return (float *)elem;
}

 float *CCG_elem_no(const CCGKey *key, CCGElem *elem)
{
  BLI_assert(key->has_normals);
  return (float *)((char *)elem + key->normal_offset);
}

 float *CCG_elem_mask(const CCGKey *key, CCGElem *elem)
{
  BLI_assert(key->has_mask);
  return (float *)((char *)elem + (key->mask_offset));
}

 CCGElem *CCG_elem_offset(const CCGKey *key, CCGElem *elem, int offset)
{
  return (CCGElem *)(((char *)elem) + key->elem_size * offset);
}

 CCGElem *CCG_grid_elem(const CCGKey *key, CCGElem *elem, int x, int y)
{
  //  BLI_assert(x < key->grid_size && y < key->grid_size);
  return CCG_elem_offset(key, elem, (y * key->grid_size + x));
}

 float *CCG_grid_elem_co(const CCGKey *key, CCGElem *elem, int x, int y)
{
  return CCG_elem_co(CCG_grid_elem(key, elem, x, y));
}

 float *CCG_grid_elem_no(const CCGKey *key, CCGElem *elem, int x, int y)
{
  return CCG_elem_no(key, CCG_grid_elem(key, elem, x, y));
}

 float *CCG_grid_elem_mask(const CCGKey *key, CCGElem *elem, int x, int y)
{
  return CCG_elem_mask(key, CCG_grid_elem(key, elem, x, y));
}

 float *CCG_elem_offset_co(const CCGKey *key, CCGElem *elem, int offset)
{
  return CCG_elem_co(CCG_elem_offset(key, elem, offset));
}

 float *CCG_elem_offset_no(const CCGKey *key, CCGElem *elem, int offset)
{
  return CCG_elem_no(key, CCG_elem_offset(key, elem, offset));
}

 float *CCG_elem_offset_mask(const CCGKey *key, CCGElem *elem, int offset)
{
  return CCG_elem_mask(key, CCG_elem_offset(key, elem, offset));
}

 CCGElem *CCG_elem_next(const CCGKey *key, CCGElem *elem)
{
  return CCG_elem_offset(key, elem, 1);
}