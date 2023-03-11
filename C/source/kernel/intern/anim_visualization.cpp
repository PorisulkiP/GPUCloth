#include "MEM_guardedalloc.cuh"

#include "DNA_action_types.h"
#include "anim_types.h"
#include "object_types.cuh"
#include "scene_types.cuh"

#include "BKE_anim_visualization.h"
#include "BKE_report.h"

/* ******************************************************************** */
/* Animation Visualization */

/* Initialize the default settings for animation visualization */
void animviz_settings_init(bAnimVizSettings *avs)
{
  /* sanity check */
  if (avs == NULL) {
    return;
  }

  /* path settings */
  avs->path_bc = avs->path_ac = 10;

  avs->path_sf = 1;   /* xxx - take from scene instead? */
  avs->path_ef = 250; /* xxx - take from scene instead? */

  avs->path_viewflag = (MOTIONPATH_VIEW_KFRAS | MOTIONPATH_VIEW_KFNOS);

  avs->path_step = 1;

  avs->path_bakeflag |= MOTIONPATH_BAKE_HEADS;
}


/* Free the given motion path instance and its data
 * NOTE: this frees the motion path given!
 */
void animviz_free_motionpath(bMotionPath *mpath)
{
  /* sanity check */
  if (mpath == NULL) {
    return;
  }

  /* free the cache first */
  animviz_free_motionpath_cache(mpath);

  /* now the instance itself */
  MEM_freeN(mpath);
}
