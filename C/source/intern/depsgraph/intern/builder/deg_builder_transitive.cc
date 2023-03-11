#include "builder/deg_builder_transitive.h"

#include "MEM_guardedalloc.cuh"

#include "node/deg_node.h"
#include "node/deg_node_component.h"
#include "node/deg_node_operation.h"

#include "debug/deg_debug.h"
#include "depsgraph.h"
#include "depsgraph_relation.h"

namespace blender::deg {
enum {
  OP_VISITED = 1,
  OP_REACHABLE = 2,
};

static void deg_graph_tag_paths_recursive(Node *node)
{
  if (node->custom_flags & OP_VISITED) {
    return;
  }
  node->custom_flags |= OP_VISITED;
  for (Relation *rel : node->inlinks) {
    deg_graph_tag_paths_recursive(rel->from);
    /* Do this only in inlinks loop, so the target node does not get
     * flagged. */
    rel->from->custom_flags |= OP_REACHABLE;
  }
}

void deg_graph_transitive_reduction(Depsgraph *graph)
{
  int num_removed_relations = 0;
  Vector<Relation *> relations_to_remove;

  for (OperationNode *target : graph->operations) {
    /* Clear tags. */
    for (OperationNode *node : graph->operations) {
      node->custom_flags = 0;
    }
    /* Mark nodes from which we can reach the target
     * start with children, so the target node and direct children are not
     * flagged. */
    target->custom_flags |= OP_VISITED;
    for (Relation *rel : target->inlinks) {
      deg_graph_tag_paths_recursive(rel->from);
    }
    /* Remove redundant paths to the target. */
    for (Relation *rel : target->inlinks) {
      if (rel->from->type == NodeType::TIMESOURCE) {
        /* HACK: time source nodes don't get "custom_flags" flag
         * set/cleared. */
        /* TODO: there will be other types in future, so iterators above
         * need modifying. */
        continue;
      }
      if (rel->from->custom_flags & OP_REACHABLE) {
        relations_to_remove.append(rel);
      }
    }
    for (Relation *rel : relations_to_remove) {
      rel->unlink();
      delete rel;
    }
    num_removed_relations += relations_to_remove.size();
    relations_to_remove.clear();
  }}

}  // namespace blender::deg
