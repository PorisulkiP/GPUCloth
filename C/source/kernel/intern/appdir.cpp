#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>

#include "BLI_fileops.h"
#include "BLI_fileops_types.h"
#include "listbase.cuh"
#include "BLI_path_util.h"
#include "BLI_string.h"
#include "BLI_string_utf8.h"
#include "BLI_string_utils.h"
#include "utildefines.h"

#include "BKE_appdir.h" /* own include */
#include "BKE_blender_version.h"

#include "BLT_translation.h"

#include "MEM_guardedalloc.cuh"

#ifdef WIN32
#  include <io.h>
#  ifdef _WIN32_IE
#    undef _WIN32_IE
#  endif
#  define _WIN32_IE 0x0501
#  include "BLI_winstuff.h"
#  include <shlobj.h>
#  include <windows.h>
#else /* non windows */
#  ifdef WITH_BINRELOC
#    include "binreloc.h"
#  endif
/* #mkdtemp on OSX (and probably all *BSD?), not worth making specific check for this OS. */
#  include <unistd.h>
#endif /* WIN32 */

static const char _str_null[] = "(null)";
#define STR_OR_FALLBACK(a) ((a) ? (a) : _str_null)


static struct {
  /** Full path to program executable. */
  char program_filename[FILE_MAX];
  /** Full path to directory in which executable is located. */
  char program_dirname[FILE_MAX];
  /** Persistent temporary directory (defined by the preferences or OS). */
  char temp_dirname_base[FILE_MAX];
  /** Volatile temporary directory (owned by Blender, removed on exit). */
  char temp_dirname_session[FILE_MAX];
} g_app = {
    .temp_dirname_session = "",
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Initialization
 * \{ */

#ifndef NDEBUG
static bool is_appdir_init = false;
#  define ASSERT_IS_INIT() BLI_assert(is_appdir_init)
#else
#  define ASSERT_IS_INIT() ((void)0)
#endif

/**
 * Sanity check to ensure correct API use in debug mode.
 *
 * Run this once the first level of arguments has been passed so we can be sure
 * `--env-system-datafiles`, and other `--env-*` arguments has been passed.
 *
 * Without this any callers to this module that run early on,
 * will miss out on changes from parsing arguments.
 */
void BKE_appdir_init(void)
{
#ifndef NDEBUG
  BLI_assert(is_appdir_init == false);
  is_appdir_init = true;
#endif
}

void BKE_appdir_exit(void)
{
#ifndef NDEBUG
  BLI_assert(is_appdir_init == true);
  is_appdir_init = false;
#endif
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Internal Utilities
 * \{ */

/**
 * \returns a formatted representation of the specified version number. Non-re-entrant!
 */
static char *blender_version_decimal(const int version)
{
  static char version_str[5];
  BLI_assert(version < 1000);
  BLI_snprintf(version_str, sizeof(version_str), "%d.%02d", version / 100, version % 100);
  return version_str;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Default Directories
 * \{ */

/**
 * Get the folder that's the "natural" starting point for browsing files on an OS. On Unix that is
 * $HOME, on Windows it is %userprofile%/Documents.
 *
 * \note On Windows `Users/{MyUserName}/Documents` is used as it's the default location to save
 *       documents.
 */
const char *BKE_appdir_folder_default(void)
{
#ifndef WIN32
  return BLI_getenv("HOME");
#else  /* Windows */
  static char documentfolder[MAXPATHLEN];

  if (BKE_appdir_folder_documents(documentfolder)) {
    return documentfolder;
  }

  return NULL;
#endif /* WIN32 */
}

/**
 * Get the user's home directory, i.e. $HOME on UNIX, %userprofile% on Windows.
 */
const char *BKE_appdir_folder_home(void)
{
#ifndef WIN32
  return BLI_getenv("HOME");
#else /* Windows */
  return BLI_getenv("userprofile");
#endif
}

/**
 * Gets a good default directory for fonts.
 */
bool BKE_appdir_font_folder_default(
    /* This parameter can only be `const` on non-windows platforms.
     * NOLINTNEXTLINE: readability-non-const-parameter. */
    char *dir)
{
  bool success = false;
#ifdef WIN32
  wchar_t wpath[FILE_MAXDIR];
  success = SHGetSpecialFolderPathW(0, wpath, CSIDL_FONTS, 0);
  if (success) {
    wcscat(wpath, L"\\");
    BLI_strncpy_wchar_as_utf8(dir, wpath, FILE_MAXDIR);
  }
#endif
  /* TODO: Values for other platforms. */
  UNUSED_VARS(dir);
  return success;
}

const char *BKE_appdir_folder_id(const int folder_id, const char *subfolder)
{
  static char path[FILE_MAX] = "";
  if (BKE_appdir_folder_id_ex(folder_id, subfolder, path, sizeof(path))) {
    return path;
  }
  return NULL;
}

/**
 * Returns the path to a folder in the user area without checking that it actually exists first.
 */
const char *BKE_appdir_folder_id_user_notest(const int folder_id, const char *subfolder)
{
  const int version = BLENDER_VERSION;
  static char path[FILE_MAX] = "";
  const bool check_is_dir = false;

  switch (folder_id) {
    case BLENDER_USER_DATAFILES:
      if (get_path_environment_ex(
              path, sizeof(path), subfolder, "BLENDER_USER_DATAFILES", check_is_dir)) {
        break;
      }
      get_path_user_ex(path, sizeof(path), "datafiles", subfolder, version, check_is_dir);
      break;
    case BLENDER_USER_CONFIG:
      if (get_path_environment_ex(
              path, sizeof(path), subfolder, "BLENDER_USER_CONFIG", check_is_dir)) {
        break;
      }
      get_path_user_ex(path, sizeof(path), "config", subfolder, version, check_is_dir);
      break;
    case BLENDER_USER_AUTOSAVE:
      if (get_path_environment_ex(
              path, sizeof(path), subfolder, "BLENDER_USER_AUTOSAVE", check_is_dir)) {
        break;
      }
      get_path_user_ex(path, sizeof(path), "autosave", subfolder, version, check_is_dir);
      break;
    case BLENDER_USER_SCRIPTS:
      if (get_path_environment_ex(
              path, sizeof(path), subfolder, "BLENDER_USER_SCRIPTS", check_is_dir)) {
        break;
      }
      get_path_user_ex(path, sizeof(path), "scripts", subfolder, version, check_is_dir);
      break;
    default:
      BLI_assert(0);
      break;
  }

  if ('\0' == path[0]) {
    return NULL;
  }
  return path;
}

/**
 * Returns the path to a folder in the user area, creating it if it doesn't exist.
 */
const char *BKE_appdir_folder_id_create(const int folder_id, const char *subfolder)
{
  const char *path;

  /* Only for user folders. */
  if (!ELEM(folder_id,
            BLENDER_USER_DATAFILES,
            BLENDER_USER_CONFIG,
            BLENDER_USER_SCRIPTS,
            BLENDER_USER_AUTOSAVE)) {
    return NULL;
  }

  path = BKE_appdir_folder_id(folder_id, subfolder);

  if (!path) {
    path = BKE_appdir_folder_id_user_notest(folder_id, subfolder);
    if (path) {
      BLI_dir_create_recursive(path);
    }
  }

  return path;
}

/**
 * Returns the path of the top-level version-specific local, user or system directory.
 * If check_is_dir, then the result will be NULL if the directory doesn't exist.
 */
const char *BKE_appdir_folder_id_version(const int folder_id,
                                         const int version,
                                         const bool check_is_dir)
{
  static char path[FILE_MAX] = "";
  bool ok;
  switch (folder_id) {
    case BLENDER_RESOURCE_PATH_USER:
      ok = get_path_user_ex(path, sizeof(path), NULL, NULL, version, check_is_dir);
      break;
    case BLENDER_RESOURCE_PATH_LOCAL:
      ok = get_path_local_ex(path, sizeof(path), NULL, NULL, version, check_is_dir);
      break;
    case BLENDER_RESOURCE_PATH_SYSTEM:
      ok = get_path_system_ex(path, sizeof(path), NULL, NULL, version, check_is_dir);
      break;
    default:
      path[0] = '\0'; /* in case check_is_dir is false */
      ok = false;
      BLI_assert(!"incorrect ID");
      break;
  }
  return ok ? path : NULL;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Program Path Queries
 *
 * Access locations of Blender & Python.
 * \{ */

void BKE_appdir_program_path_init(const char *argv0)
{
  where_am_i(g_app.program_filename, sizeof(g_app.program_filename), argv0);
  BLI_split_dir_part(g_app.program_filename, g_app.program_dirname, sizeof(g_app.program_dirname));
}

/**
 * Path to executable
 */
const char *BKE_appdir_program_path(void)
{
  BLI_assert(g_app.program_filename[0]);
  return g_app.program_filename;
}

/** Keep in sync with `bpy.utils.app_template_paths()` */
static const char *app_template_directory_search[2] = {
    "startup" SEP_STR "bl_app_templates_user",
    "startup" SEP_STR "bl_app_templates_system",
};

static const int app_template_directory_id[2] = {
    /* Only 'USER' */
    BLENDER_USER_SCRIPTS,
    /* Covers 'LOCAL' & 'SYSTEM'. */
    BLENDER_SYSTEM_SCRIPTS,
};

/**
 * Return true if templates exist
 */
bool BKE_appdir_app_template_any(void)
{
  char temp_dir[FILE_MAX];
  for (int i = 0; i < ARRAY_SIZE(app_template_directory_id); i++) {
    if (BKE_appdir_folder_id_ex(app_template_directory_id[i],
                                app_template_directory_search[i],
                                temp_dir,
                                sizeof(temp_dir))) {
      return true;
    }
  }
  return false;
}

bool BKE_appdir_app_template_has_userpref(const char *app_template)
{
  /* Test if app template provides a `userpref.blend`.
   * If not, we will share user preferences with the rest of Blender. */
  if (app_template[0] == '\0') {
    return false;
  }

  char app_template_path[FILE_MAX];
  if (!BKE_appdir_app_template_id_search(
          app_template, app_template_path, sizeof(app_template_path))) {
    return false;
  }

  char userpref_path[FILE_MAX];
  BLI_path_join(
      userpref_path, sizeof(userpref_path), app_template_path, BLENDER_USERPREF_FILE, NULL);
  return BLI_exists(userpref_path);
}

void BKE_appdir_app_templates(ListBase *templates)
{
  BLI_listbase_clear(templates);

  for (int i = 0; i < ARRAY_SIZE(app_template_directory_id); i++) {
    char subdir[FILE_MAX];
    if (!BKE_appdir_folder_id_ex(app_template_directory_id[i],
                                 app_template_directory_search[i],
                                 subdir,
                                 sizeof(subdir))) {
      continue;
    }

    struct direntry *dir;
    uint totfile = BLI_filelist_dir_contents(subdir, &dir);
    for (int f = 0; f < totfile; f++) {
      if (!FILENAME_IS_CURRPAR(dir[f].relname) && S_ISDIR(dir[f].type)) {
        char *template = BLI_strdup(dir[f].relname);
        BLI_addtail(templates, BLI_genericNodeN(template));
      }
    }

    BLI_filelist_free(dir, totfile);
  }
}

/**
 * Sets #g_app.temp_dirname_base to \a userdir if specified and is a valid directory,
 * otherwise chooses a suitable OS-specific temporary directory.
 * Sets #g_app.temp_dirname_session to a #mkdtemp
 * generated sub-dir of #g_app.temp_dirname_base.
 */
void BKE_tempdir_init(const char *userdir)
{
  where_is_temp(g_app.temp_dirname_base, sizeof(g_app.temp_dirname_base), userdir);

  /* Clear existing temp dir, if needed. */
  BKE_tempdir_session_purge();
  /* Now that we have a valid temp dir, add system-generated unique sub-dir. */
  tempdir_session_create(
      g_app.temp_dirname_session, sizeof(g_app.temp_dirname_session), g_app.temp_dirname_base);
}

/**
 * Path to temporary directory (with trailing slash)
 */
const char *BKE_tempdir_session(void)
{
  return g_app.temp_dirname_session[0] ? g_app.temp_dirname_session : BKE_tempdir_base();
}

/**
 * Path to persistent temporary directory (with trailing slash)
 */
const char *BKE_tempdir_base(void)
{
  return g_app.temp_dirname_base;
}

/**
 * Delete content of this instance's temp dir.
 */
void BKE_tempdir_session_purge(void)
{
  if (g_app.temp_dirname_session[0] && BLI_is_dir(g_app.temp_dirname_session)) {
    BLI_delete(g_app.temp_dirname_session, true, true);
  }
}

/** \} */
