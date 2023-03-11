#pragma once

#include "deg_builder_cache.h"

#include "depsgraph_type.h"

struct Depsgraph;
struct Main;
struct Scene;
struct ViewLayer;

namespace blender {
namespace deg {

struct Depsgraph;
class DepsgraphNodeBuilder;
class DepsgraphRelationBuilder;

/* Base class for Depsgraph Builder pipelines.
 *
 * Basically it runs through the following steps:
 * - sanity check
 * - build nodes
 * - build relations
 * - finalize
 */
class AbstractBuilderPipeline {
 public:
  AbstractBuilderPipeline(::Depsgraph *graph);
  virtual ~AbstractBuilderPipeline();

  void build();

 protected:
  Depsgraph *deg_graph_;
  Main *bmain_;
  Scene *scene_;
  ViewLayer *view_layer_;
  DepsgraphBuilderCache builder_cache_;

  virtual unique_ptr<DepsgraphNodeBuilder> construct_node_builder();
  virtual unique_ptr<DepsgraphRelationBuilder> construct_relation_builder();

  virtual void build_step_sanity_check();
  void build_step_nodes();
  void build_step_relations();

  virtual void build_nodes(DepsgraphNodeBuilder &node_builder) = 0;
  virtual void build_relations(DepsgraphRelationBuilder &relation_builder) = 0;
};

}  // namespace deg
}  // namespace blender
