#include <string.h>

#include "MEM_guardedalloc.cuh"

#include "ID.h"
#include "customdata_types.cuh"
#include "DNA_mesh_types.h"
#include "meshdata_types.cuh"
#include "DNA_pointcloud_types.h"

#include "BLI_string_utf8.h"

#include "BKE_attribute.h"
#include "BKE_customdata.h"
#include "BKE_hair.h"
#include "BKE_pointcloud.h"
#include "BKE_report.h"

#include "RNA_access.h"

typedef struct DomainInfo {
  CustomData *customdata;
  int length;
} DomainInfo;

