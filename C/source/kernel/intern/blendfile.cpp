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

/** \file
 * \ingroup bke
 *
 * High level `.blend` file read/write,
 * and functions for writing *partial* files (only selected data-blocks).
 */

#include <stdlib.h>
#include <string.h>

#include "MEM_guardedalloc.cuh"

#include "scene_types.cuh"
#include "DNA_screen_types.h"
#include "DNA_workspace_types.h"

#include "listbase.cuh"
#include "BLI_path_util.h"
#include "BLI_string.h"
#include "BLI_system.h"
#include "utildefines.h"

#include "BKE_addon.h"
#include "BKE_appdir.h"
#include "BKE_blender.h"
#include "BKE_blender_version.h"
#include "BKE_blendfile.h"
#include "BKE_bpath.h"
#include "BKE_colorband.h"
#include "BKE_context.h"
#include "BKE_global.h"
#include "BKE_ipo.h"
#include "BKE_keyconfig.h"
#include "BKE_layer.h"
#include "BKE_lib_id.h"
#include "BKE_main.h"
#include "BKE_preferences.h"
#include "BKE_report.h"
#include "BKE_scene.h"
#include "BKE_screen.h"
#include "BKE_studiolight.h"
#include "BKE_undo_system.h"
#include "BKE_workspace.h"
#include "RNA_access.h"

/* -------------------------------------------------------------------- */
/** \name High Level `.blend` file read/write.
 * \{ */

static bool clean_paths_visit_cb(void *UNUSED(userdata), char *path_dst, const char *path_src)
{
  strcpy(path_dst, path_src);
  BLI_path_slash_native(path_dst);
  return !STREQ(path_dst, path_src);
}

/* make sure path names are correct for OS */
static void clean_paths(Main *main)
{
  Scene *scene;

  BKE_bpath_traverse_main(main, clean_paths_visit_cb, BKE_BPATH_TRAVERSE_SKIP_MULTIFILE, NULL);

  for (scene = main->scenes.first; scene; scene = scene->id.next) {
    BLI_path_slash_native(scene->r.pic);
  }
}

static bool wm_scene_is_visible(wmWindowManager *wm, Scene *scene)
{
  wmWindow *win;
  for (win = wm->windows.first; win; win = win->next) {
    if (win->scene == scene) {
      return true;
    }
  }
  return false;
}

bool BKE_blendfile_userdef_write_app_template(const char *filepath, ReportList *reports)
{
  /* if it fails, overwrite is OK. */
  UserDef *userdef_default = BKE_blendfile_userdef_read(filepath, NULL);
  if (userdef_default == NULL) {
    return BKE_blendfile_userdef_write(filepath, reports);
  }

  BKE_blender_userdef_app_template_data_swap(&U, userdef_default);
  bool ok = BKE_blendfile_userdef_write(filepath, reports);
  BKE_blender_userdef_app_template_data_swap(&U, userdef_default);
  BKE_blender_userdef_data_free(userdef_default, false);
  MEM_freeN(userdef_default);
  return ok;
}

bool BKE_blendfile_userdef_write_all(ReportList *reports)
{
  char filepath[FILE_MAX];
  const char *cfgdir;
  bool ok = true;
  const bool use_template_userpref = BKE_appdir_app_template_has_userpref(U.app_template);

  if ((cfgdir = BKE_appdir_folder_id_create(BLENDER_USER_CONFIG, NULL))) {
    bool ok_write;
    BLI_path_join(filepath, sizeof(filepath), cfgdir, BLENDER_USERPREF_FILE, NULL);

    printf("Writing userprefs: '%s' ", filepath);
    if (use_template_userpref) {
      ok_write = BKE_blendfile_userdef_write_app_template(filepath, reports);
    }
    else {
      ok_write = BKE_blendfile_userdef_write(filepath, reports);
    }

    if (ok_write) {
      printf("ok\n");
    }
    else {
      printf("fail\n");
      ok = false;
    }
  }
  else {
    BKE_report(reports, RPT_ERROR, "Unable to create userpref path");
  }

  if (use_template_userpref) {
    if ((cfgdir = BKE_appdir_folder_id_create(BLENDER_USER_CONFIG, U.app_template))) {
      /* Also save app-template prefs */
      BLI_path_join(filepath, sizeof(filepath), cfgdir, BLENDER_USERPREF_FILE, NULL);

      printf("Writing userprefs app-template: '%s' ", filepath);
      if (BKE_blendfile_userdef_write(filepath, reports) != 0) {
        printf("ok\n");
      }
      else {
        printf("fail\n");
        ok = false;
      }
    }
    else {
      BKE_report(reports, RPT_ERROR, "Unable to create app-template userpref path");
      ok = false;
    }
  }

  if (ok) {
    U.runtime.is_dirty = false;
  }
  return ok;
}
