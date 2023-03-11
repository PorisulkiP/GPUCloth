#include "MEM_guardedalloc.cuh"

#include "DNA_brush_types.h"
#include "defaults.cuh"
#include "DNA_gpencil_types.h"
#include "DNA_material_types.h"
#include "object_types.cuh"
#include "scene_types.cuh"

#include "listbase.cuh"
#include "math.cuh"
#include "BLI_rand.h"

#include "BLT_translation.h"

#include "BKE_brush.h"
#include "BKE_colortools.h"
#include "BKE_context.h"
#include "BKE_gpencil.h"
#include "BKE_icons.h"
#include "BKE_idtype.h"
#include "BKE_lib_id.h"
#include "BKE_lib_query.h"
#include "BKE_lib_remap.h"
#include "BKE_main.h"
#include "BKE_material.h"
#include "BKE_paint.h"
#include "BKE_texture.h"

#include "IMB_colormanagement.h"
#include "IMB_imbuf.h"
#include "IMB_imbuf_types.h"

#include "RE_texture.h" /* RE_texture_evaluate */


