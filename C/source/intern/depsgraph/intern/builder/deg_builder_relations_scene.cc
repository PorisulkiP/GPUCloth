#include "builder/deg_builder_relations.h"

#include "scene_types.cuh"

namespace blender::deg {

void DepsgraphRelationBuilder::build_scene_render(Scene *scene, ViewLayer *view_layer)
{
  scene_ = scene;
  const bool build_compositor = (scene->r.scemode & R_DOCOMP);
  const bool build_sequencer = (scene->r.scemode & R_DOSEQ);
  build_scene_parameters(scene);
  build_scene_audio(scene);
  if (build_compositor) {
    build_scene_compositor(scene);
  }
  if (build_sequencer) {
    build_scene_sequencer(scene);
    build_scene_speakers(scene, view_layer);
  }
  if (scene->camera != nullptr) {
    build_object(scene->camera);
  }
}

void DepsgraphRelationBuilder::build_scene_parameters(Scene *scene)
{}

void DepsgraphRelationBuilder::build_scene_compositor(Scene *scene)
{
  if (built_map_.checkIsBuiltAndTag(scene, BuilderMap::TAG_SCENE_COMPOSITOR)) {
    return;
  }
  if (scene->nodetree == nullptr) {
    return;
  }
  build_nodetree(scene->nodetree);
}

}  // namespace blender::deg
