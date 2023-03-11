#include "eval/deg_eval_runtime_backup.h"
#include "eval/deg_eval_copy_on_write.h"

#include "utildefines.h"

namespace blender::deg {

RuntimeBackup::RuntimeBackup(const Depsgraph *depsgraph)
    : have_backup(false),
      scene_backup(depsgraph),
      object_backup(depsgraph),
      drawdata_ptr(nullptr),
      movieclip_backup(depsgraph),
      volume_backup(depsgraph)
{
  drawdata_backup.first = drawdata_backup.last = nullptr;
}

void RuntimeBackup::init_from_id(ID *id)
{
}

void RuntimeBackup::restore_to_id(ID *id)
{
}

}  // namespace blender::deg
