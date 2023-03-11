#include "eval/deg_eval_flush.h"

#include <cmath>

#include "listBase.h"
#include "math_vector.cuh"
#include "utildefines.h"
#include "object_types.cuh"
#include "scene_types.cuh"


#include "DEG_depsgraph.h"

#include "debug/deg_debug.h"
#include "depsgraph.h"
#include "depsgraph_relation.h"
#include "depsgraph_update.h"
#include "node/deg_node.h"
#include "node/deg_node_component.h"
#include "node/deg_node_factory.h"
#include "node/deg_node_id.h"
#include "node/deg_node_operation.h"
#include "node/deg_node_time.h"

#include "eval/deg_eval_copy_on_write.h"

#undef INVALIDATE_ON_FLUSH

namespace blender::deg {

enum {
  ID_STATE_NONE = 0,
  ID_STATE_MODIFIED = 1,
};

enum {
  COMPONENT_STATE_NONE = 0,
  COMPONENT_STATE_SCHEDULED = 1,
  COMPONENT_STATE_DONE = 2,
};

using FlushQueue = deque<OperationNode *>;

namespace {

inline void flush_schedule_entrypoints(Depsgraph *graph, FlushQueue *queue)
{}

inline void flush_handle_id_node(IDNode *id_node)
{
  id_node->custom_flags = ID_STATE_MODIFIED;
}

/* TODO(sergey): We can reduce number of arguments here. */
inline void flush_handle_component_node(IDNode *id_node,
                                        ComponentNode *comp_node,
                                        FlushQueue *queue)
{
  /* We only handle component once. */
  if (comp_node->custom_flags == COMPONENT_STATE_DONE) {
    return;
  }
  comp_node->custom_flags = COMPONENT_STATE_DONE;
  /* Tag all required operations in component for update, unless this is a
   * special component where we don't want all operations to be tagged.
   *
   * TODO(sergey): Make this a more generic solution. */
  if (!ELEM(comp_node->type, NodeType::PARTICLE_SETTINGS, NodeType::PARTICLE_SYSTEM)) {
    for (OperationNode *op : comp_node->operations) {
      op->flag |= DEPSOP_FLAG_NEEDS_UPDATE;
    }
  }
  /* when some target changes bone, we might need to re-run the
   * whole IK solver, otherwise result might be unpredictable. */
  if (comp_node->type == NodeType::BONE) {
    ComponentNode *pose_comp = id_node->find_component(NodeType::EVAL_POSE);
    BLI_assert(pose_comp != nullptr);
    if (pose_comp->custom_flags == COMPONENT_STATE_NONE) {
      queue->push_front(pose_comp->get_entry_operation());
      pose_comp->custom_flags = COMPONENT_STATE_SCHEDULED;
    }
  }
}

/* Schedule children of the given operation node for traversal.
 *
 * One of the children will by-pass the queue and will be returned as a function
 * return value, so it can start being handled right away, without building too
 * much of a queue.
 */
inline OperationNode *flush_schedule_children(OperationNode *op_node, FlushQueue *queue)
{
  if (op_node->flag & DEPSOP_FLAG_USER_MODIFIED) {
    IDNode *id_node = op_node->owner->owner;
    id_node->is_user_modified = true;
  }

  OperationNode *result = nullptr;
  for (Relation *rel : op_node->outlinks) {
    /* Flush is forbidden, completely. */
    if (rel->flag & RELATION_FLAG_NO_FLUSH) {
      continue;
    }
    /* Relation only allows flushes on user changes, but the node was not
     * affected by user. */
    if ((rel->flag & RELATION_FLAG_FLUSH_USER_EDIT_ONLY) &&
        (op_node->flag & DEPSOP_FLAG_USER_MODIFIED) == 0) {
      continue;
    }
    OperationNode *to_node = (OperationNode *)rel->to;
    /* Always flush flushable flags, so children always know what happened
     * to their parents. */
    to_node->flag |= (op_node->flag & DEPSOP_FLAG_FLUSH);
    /* Flush update over the relation, if it was not flushed yet. */
    if (to_node->scheduled) {
      continue;
    }
    if (result != nullptr) {
      queue->push_front(to_node);
    }
    else {
      result = to_node;
    }
    to_node->scheduled = true;
  }
  return result;
}

}  // namespace

/* Flush updates from tagged nodes outwards until all affected nodes
 * are tagged.
 */
void deg_graph_flush_updates(Depsgraph *graph)
{
  /* Sanity checks. */
  BLI_assert(graph != nullptr);
  Main *bmain = graph->bmain;

  graph->time_source->flush_update_tag(graph);

  /* Nothing to update, early out. */
  if (graph->entry_tags.is_empty()) {
    return;
  }
  /* Starting from the tagged "entry" nodes, flush outwards. */
  FlushQueue queue;
  flush_schedule_entrypoints(graph, &queue);
  /* Prepare update context for editors. */
  DEGEditorUpdateContext update_ctx;
  update_ctx.bmain = bmain;
  update_ctx.depsgraph = (::Depsgraph *)graph;
  update_ctx.scene = graph->scene;
  update_ctx.view_layer = graph->view_layer;
  /* Do actual flush. */
  while (!queue.empty()) {
    OperationNode *op_node = queue.front();
    queue.pop_front();
    while (op_node != nullptr) {
      /* Tag operation as required for update. */
      op_node->flag |= DEPSOP_FLAG_NEEDS_UPDATE;
      /* Inform corresponding ID and component nodes about the change. */
      ComponentNode *comp_node = op_node->owner;
      IDNode *id_node = comp_node->owner;
      flush_handle_id_node(id_node);
      flush_handle_component_node(id_node, comp_node, &queue);
      /* Flush to nodes along links. */
      op_node = flush_schedule_children(op_node, &queue);
    }
  }
}

/* Clear tags from all operation nodes. */
void deg_graph_clear_tags(Depsgraph *graph)
{
  /* Go over all operation nodes, clearing tags. */
  for (OperationNode *node : graph->operations) {
    node->flag &= ~(DEPSOP_FLAG_DIRECTLY_MODIFIED | DEPSOP_FLAG_NEEDS_UPDATE |
                    DEPSOP_FLAG_USER_MODIFIED);
  }
  /* Clear any entry tags which haven't been flushed. */
  graph->entry_tags.clear();

  graph->time_source->tagged_for_update = false;
}

}  // namespace blender::deg
