#pragma once
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ID;
struct bNodeSocketType;
struct bNodeTreeType;
struct bNodeType;

struct IDFilterEnumPropertyItem {
  const uint64_t flag;
  const char *identifier;
  const int icon;
  const char *name;
  const char *description;
};
extern const struct IDFilterEnumPropertyItem rna_enum_id_type_filter_items[];

struct PointerRNA;
struct PropertyRNA;
struct bContext;

#ifdef __cplusplus
}
#endif
