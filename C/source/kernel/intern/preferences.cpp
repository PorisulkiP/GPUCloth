#include <string.h>

#include "MEM_guardedalloc.cuh"

#include "listbase.cuh"

#define U BLI_STATIC_ASSERT(false, "Global 'U' not allowed, only use arguments passed in!")

/* -------------------------------------------------------------------- */
/** \name Asset Libraries
 * \{ */

bUserAssetLibrary *BKE_preferences_asset_library_add(UserDef *userdef,
                                                     const char *name,
                                                     const char *path)
{
  bUserAssetLibrary *library = (bUserAssetLibrary*)MEM_callocN(sizeof(*library), "bUserAssetLibrary");

  BLI_addtail(&userdef->asset_libraries, library);

  if (name) {
    BKE_preferences_asset_library_name_set(userdef, library, name);
  }
  if (path) {
    BLI_strncpy(library->path, path, sizeof(library->path));
  }

  return library;
}

void BKE_preferences_asset_library_name_set(UserDef *userdef,
                                            bUserAssetLibrary *library,
                                            const char *name)
{
  BLI_strncpy_utf8(library->name, name, sizeof(library->name));
  BLI_uniquename(&userdef->asset_libraries,
                 library,
                 name,
                 '.',
                 offsetof(bUserAssetLibrary, name),
                 sizeof(library->name));
}

/**
 * Unlink and free a library preference member.
 * \note Free's \a library itself.
 */
void BKE_preferences_asset_library_remove(UserDef *userdef, bUserAssetLibrary *library)
{
  BLI_freelinkN(&userdef->asset_libraries, library);
}

bUserAssetLibrary *BKE_preferences_asset_library_find_from_index(const UserDef *userdef, int index)
{
  return (bUserAssetLibrary*)BLI_findlink(&userdef->asset_libraries, index);
}

bUserAssetLibrary *BKE_preferences_asset_library_find_from_name(const UserDef *userdef,
                                                                const char *name)
{
  return (bUserAssetLibrary*)BLI_findstring(&userdef->asset_libraries, name, offsetof(bUserAssetLibrary, name));
}

int BKE_preferences_asset_library_get_index(const UserDef *userdef,
                                            const bUserAssetLibrary *library)
{
  return BLI_findindex(&userdef->asset_libraries, library);
}

void BKE_preferences_asset_library_default_add(UserDef *userdef)
{
  char documents_path[FILE_MAXDIR];

  /* No home or documents path found, not much we can do. */
  if (!BKE_appdir_folder_documents(documents_path) || !documents_path[0]) {
    return;
  }

  bUserAssetLibrary *library = BKE_preferences_asset_library_add(userdef, DATA_("Default"), NULL);

  /* Add new "Default" library under '[doc_path]/Blender/Assets'. */
  BLI_path_join(
      library->path, sizeof(library->path), documents_path, N_("Blender"), N_("Assets"), NULL);
}

/** \} */
