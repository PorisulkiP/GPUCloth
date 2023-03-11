#include "eval/deg_eval_runtime_backup_scene.h"
#include "scene_types.cuh"

namespace blender::deg {

SceneBackup::SceneBackup(const Depsgraph *depsgraph) : sequencer_backup(depsgraph)
{
  reset();
}

void SceneBackup::reset()
{
  sound_scene = nullptr;
  playback_handle = nullptr;
  sound_scrub_handle = nullptr;
  speaker_handles = nullptr;
  rigidbody_last_time = -1;
}

void SceneBackup::init_from_scene(Scene *scene)
{

  sound_scene = scene->sound_scene;
  playback_handle = scene->playback_handle;
  sound_scrub_handle = scene->sound_scrub_handle;
  speaker_handles = scene->speaker_handles;

  /* Clear pointers stored in the scene, so they are not freed when copied-on-written datablock
   * is freed for re-allocation. */
  scene->sound_scene = nullptr;
  scene->playback_handle = nullptr;
  scene->sound_scrub_handle = nullptr;
  scene->speaker_handles = nullptr;

  sequencer_backup.init_from_scene(scene);
}

void SceneBackup::restore_to_scene(Scene *scene)
{
  scene->sound_scene = sound_scene;
  scene->playback_handle = playback_handle;
  scene->sound_scrub_handle = sound_scrub_handle;
  scene->speaker_handles = speaker_handles;

  sequencer_backup.restore_to_scene(scene);
  reset();
}

}  // namespace blender::deg
