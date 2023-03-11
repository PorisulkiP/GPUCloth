/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#pragma once

/** \file
 * \ingroup bke
 */

#include "compiler_attrs.cuh"


struct BlendDataReader;
struct BlendExpander;
struct BlendLibReader;
struct BlendWriter;
struct ID;
struct ID;

typedef union IDTemplate {
  int i;
  float f;
  double d;
  struct {
    const char *str;
    int len;
    char subtype;
  } string;
  struct ID *id;
  struct {
    int len;
    char type;
  } array;
  struct {
    int matvec_size;
    const float *example;
  } matrix_or_vector;
} IDTemplate;

/* ----------- Property Array Type ---------- */

struct ID *IDP_NewIDPArray(const char *name) ATTR_WARN_UNUSED_RESULT ATTR_NONNULL();
struct ID *IDP_CopyIDPArray(const struct ID *array,
                                    const int flag) ATTR_WARN_UNUSED_RESULT ATTR_NONNULL();

/* shallow copies item */
void IDP_SetIndexArray(struct ID *prop, int index, struct ID *item) ATTR_NONNULL();
struct ID *IDP_GetIndexArray(struct ID *prop, int index) ATTR_WARN_UNUSED_RESULT
    ATTR_NONNULL();
void IDP_AppendArray(struct ID *prop, struct ID *item);
void IDP_ResizeIDPArray(struct ID *prop, int len);

/* ----------- Numeric Array Type ----------- */
/*this function works for strings too!*/
void IDP_ResizeArray(struct ID *prop, int newlen);
void IDP_FreeArray(struct ID *prop);

/* ---------- String Type ------------ */
struct ID *IDP_NewString(const char *st,
                                 const char *name,
                                 int maxlen) ATTR_WARN_UNUSED_RESULT
    ATTR_NONNULL(2 /* 'name 'arg */); /* maxlen excludes '\0' */
void IDP_AssignString(struct ID *prop, const char *st, int maxlen)
    ATTR_NONNULL(); /* maxlen excludes '\0' */
void IDP_ConcatStringC(struct ID *prop, const char *st) ATTR_NONNULL();
void IDP_ConcatString(struct ID *str1, struct ID *append) ATTR_NONNULL();
void IDP_FreeString(struct ID *prop) ATTR_NONNULL();

/*-------- ID Type -------*/

typedef void (*IDPWalkFunc)(void *userData, struct ID *idp);

void IDP_AssignID(struct ID *prop, struct ID *id, const int flag);

/*-------- Group Functions -------*/

/** Sync values from one group to another, only where they match */
void IDP_SyncGroupValues(struct ID *dest, const struct ID *src) ATTR_NONNULL();
void IDP_SyncGroupTypes(struct ID *dest,
                        const struct ID *src,
                        const bool do_arraylen) ATTR_NONNULL();
void IDP_ReplaceGroupInGroup(struct ID *dest, const struct ID *src) ATTR_NONNULL();
void IDP_ReplaceInGroup(struct ID *group, struct ID *prop) ATTR_NONNULL();
void IDP_ReplaceInGroup_ex(struct ID *group,
                           struct ID *prop,
                           struct ID *prop_exist);
void IDP_MergeGroup(struct ID *dest, const struct ID *src, const bool do_overwrite)
    ATTR_NONNULL();
void IDP_MergeGroup_ex(struct ID *dest,
                       const struct ID *src,
                       const bool do_overwrite,
                       const int flag) ATTR_NONNULL();
bool IDP_AddToGroup(struct ID *group, struct ID *prop) ATTR_NONNULL();
bool IDP_InsertToGroup(struct ID *group,
                       struct ID *previous,
                       struct ID *pnew) ATTR_NONNULL(1 /* group */, 3 /* pnew */);
void IDP_RemoveFromGroup(struct ID *group, struct ID *prop) ATTR_NONNULL();
void IDP_FreeFromGroup(struct ID *group, struct ID *prop) ATTR_NONNULL();

struct ID *IDP_GetPropertyFromGroup(const struct ID *prop,
                                            const char *name) ATTR_WARN_UNUSED_RESULT
    ATTR_NONNULL();
struct ID *IDP_GetPropertyTypeFromGroup(const struct ID *prop,
                                                const char *name,
                                                const char type) ATTR_WARN_UNUSED_RESULT
    ATTR_NONNULL();

/*-------- Main Functions --------*/
struct ID *IDP_GetProperties(struct ID *id,
                                     const bool create_if_needed) ATTR_WARN_UNUSED_RESULT
    ATTR_NONNULL();
struct ID *IDP_CopyProperty(const struct ID *prop) ATTR_WARN_UNUSED_RESULT
    ATTR_NONNULL();
struct ID *IDP_CopyProperty_ex(const struct ID *prop,
                                       const int flag) ATTR_WARN_UNUSED_RESULT ATTR_NONNULL();
void IDP_CopyPropertyContent(struct ID *dst, struct ID *src) ATTR_NONNULL();

bool IDP_EqualsProperties_ex(struct ID *prop1,
                             struct ID *prop2,
                             const bool is_strict) ATTR_WARN_UNUSED_RESULT;

bool IDP_EqualsProperties(struct ID *prop1,
                          struct ID *prop2) ATTR_WARN_UNUSED_RESULT;

struct ID *IDP_New(const char type,
                           const IDTemplate *val,
                           const char *name) ATTR_WARN_UNUSED_RESULT ATTR_NONNULL();

void IDP_FreePropertyContent_ex(struct ID *prop, const bool do_id_user);
void IDP_FreePropertyContent(struct ID *prop);
void IDP_FreeProperty_ex(struct ID *prop, const bool do_id_user);
void IDP_FreeProperty(struct ID *prop);

void IDP_ClearProperty(struct ID *prop);

void IDP_Reset(struct ID *prop, const struct ID *reference);

#define IDP_Int(prop) ((prop)->data.val)
#define IDP_Array(prop) ((prop)->data.pointer)
/* C11 const correctness for casts */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#  define IDP_Float(prop) \
    _Generic((prop), \
  struct ID *:             (*(float *)&(prop)->data.val), \
  const struct ID *: (*(const float *)&(prop)->data.val))
#  define IDP_Double(prop) \
    _Generic((prop), \
  struct ID *:             (*(double *)&(prop)->data.val), \
  const struct ID *: (*(const double *)&(prop)->data.val))
#  define IDP_String(prop) \
    _Generic((prop), \
  struct ID *:             ((char *) (prop)->data.pointer), \
  const struct ID *: ((const char *) (prop)->data.pointer))
#  define IDP_IDPArray(prop) \
    _Generic((prop), \
  struct ID *:             ((struct ID *) (prop)->data.pointer), \
  const struct ID *: ((const struct ID *) (prop)->data.pointer))
#  define IDP_Id(prop) \
    _Generic((prop), \
  struct ID *:             ((ID *) (prop)->data.pointer), \
  const struct ID *: ((const ID *) (prop)->data.pointer))
#else
#  define IDP_Float(prop) (*(float *)&(prop)->data.val)
#  define IDP_Double(prop) (*(double *)&(prop)->data.val)
#  define IDP_String(prop) ((char *)(prop)->data.pointer)
#  define IDP_IDPArray(prop) ((struct ID *)(prop)->data.pointer)
#  define IDP_Id(prop) ((ID *)(prop)->data.pointer)
#endif

/**
 * Call a callback for each ID in the hierarchy under given root one (included).
 *
 */
typedef void (*IDPForeachPropertyCallback)(struct ID *id_property, void *user_data);

void IDP_foreach_property(struct ID *id_property_root,
                          const int type_filter,
                          IDPForeachPropertyCallback callback,
                          void *user_data);

/* Format ID as strings */
char *IDP_reprN(const struct ID *prop, uint *r_len);
void IDP_repr_fn(const struct ID *prop,
                 void (*str_append_fn)(void *user_data, const char *str, uint str_len),
                 void *user_data);
void IDP_print(const struct ID *prop);

void IDP_BlendWrite(struct BlendWriter *writer, const struct ID *prop);
void IDP_BlendReadData_impl(struct BlendDataReader *reader,
                            struct ID **prop,
                            const char *caller_func_id);
#define IDP_BlendDataRead(reader, prop) IDP_BlendReadData_impl(reader, prop, __func__)
void IDP_BlendReadLib(struct BlendLibReader *reader, struct ID *prop);
void IDP_BlendReadExpand(struct BlendExpander *expander, struct ID *prop);
