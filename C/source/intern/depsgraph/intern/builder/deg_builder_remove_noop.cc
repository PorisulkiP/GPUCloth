#include "builder/deg_builder_remove_noop.h"

#include "MEM_guardedalloc.cuh"

#include "node/deg_node.h"
#include "node/deg_node_operation.h"

#include "debug/deg_debug.h"
#include "depsgraph.h"
#include "depsgraph_relation.h"
#include "depsgraph_type.h"

namespace blender::deg {

static inline bool is_unused_noop(OperationNode *op_node)
{
  if (op_node == nullptr) {
    return false;
  }
  if (op_node->flag & OperationFlag::DEPSOP_FLAG_PINNED) {
    return false;
  }
  return op_node->is_noop() && op_node->outlinks.is_empty();
}

void deg_graph_remove_unused_noops(Depsgraph *graph)
{
  int num_removed_relations = 0;
  deque<OperationNode *> queue;

  for (OperationNode *node : graph->operations) {
    if (is_unused_noop(node)) {
      queue.push_back(node);
    }
  }

  while (!queue.empty()) {
    OperationNode *to_remove = queue.front();
    queue.pop_front();

    while (!to_remove->inlinks.is_empty()) {
      Relation *rel_in = to_remove->inlinks[0];
      Node *dependency = rel_in->from;

      /* Remove the relation. */
      rel_in->unlink();
      delete rel_in;
      num_removed_relations++;

      /* Queue parent no-op node that has now become unused. */
      OperationNode *operation = dependency->get_exit_operation();
      if (is_unused_noop(operation)) {
        queue.push_back(operation);
      }
    }
  }
}

}  // namespace blender::deg
